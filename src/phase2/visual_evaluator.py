"""
===============================================================================
phase2/visual_evaluator.py — Evaluador Visual Fase 2 (ACECOM · BOLD5000)
===============================================================================

Pipeline end-to-end para cerrar el loop cualitativo de la tesis:

    embeds_test.pt  (adapter Ridge fMRI→CLIP-ViT-L/14)
         │
         ▼
    SD 2.1 unCLIP (bfloat16, xformers, vae_slicing)
         │
         ▼
    Reconstrucción PNG + GT (BOLD5000 stimuli) → collage lado a lado
         │
         ▼
    Grid agregado para inspección rápida en la tesis

USOS
----
    # Por defecto resuelve TODO desde config + env ACECOM_*
    python -m phase2.visual_evaluator --subject CSI1

    # Solo primeros 16 estímulos, pasos reducidos
    python -m phase2.visual_evaluator --subject CSI1 --limit 16 --steps 25

    # Override del .pt del adapter
    python -m phase2.visual_evaluator --subject CSI1 \\
        --embeds /mnt/scratch/adapter/CSI1/embeds_test.pt

PORTABILIDAD
------------
Todas las rutas se derivan de `config.DATA_DIRS` / `config.BOLD5000_CONFIG`,
controladas por las variables de entorno `ACECOM_*` definidas en config.py:

    ACECOM_BOLD5000_STIMULI_ROOT  → raíz de COCO/ImageNet/Scene
    ACECOM_PHASE2_OUTPUTS         → raíz donde está adapter/{subject}/
    ACECOM_EVAL_OUTPUT            → raíz de collages/grids (este script)
    ACECOM_HF_CACHE               → cache SD 2.1 unCLIP

Ninguna ruta local hardcodeada: corre idéntico en RTX 2070 (dev) y 4070 Ti
(inferencia remota) con sólo exportar los paths correctos.

VRAM
----
- `load_sd_unclip_pipeline` ya activa bf16 + xformers + vae slicing.
- `reconstruct_from_embedding` corre dentro de `torch.no_grad`.
- Este script añade: `torch.inference_mode`, `empty_cache` cada K imágenes,
  embedding trasladado al device una sola vez por trial y liberado tras uso.
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from phase2.bold5000_loader import get_ordered_test_stems

logger = logging.getLogger("phase2.visual_evaluator")

VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EMBED_DIM = int(config.SD_CONFIG["embedding_dim"])


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def _default_embeds_path(subject: str) -> Path:
    """`{phase2_outputs}/adapter/{subject}/embeds_test.pt` — mismo sitio que escribe train_adapter."""
    return config.DATA_DIRS["phase2_outputs"] / "adapter" / subject / "embeds_test.pt"


def load_adapter_embeddings(embed_path: Path) -> tuple[list[int], torch.Tensor]:
    """
    Lee el payload dumpeado por `phase2/train_adapter.py`:
        {"trial_ids": list[int], "embeddings": tensor(N, 768)}

    Acepta también dict {trial_id: tensor(768,)} como fallback legacy.
    """
    if not embed_path.exists():
        raise FileNotFoundError(f"Embeddings no encontrados: {embed_path}")

    blob = torch.load(embed_path, map_location="cpu")

    if isinstance(blob, dict) and "trial_ids" in blob and "embeddings" in blob:
        trial_ids = list(blob["trial_ids"])
        emb = blob["embeddings"]
        if isinstance(emb, torch.Tensor) is False:
            emb = torch.as_tensor(emb)
    elif isinstance(blob, dict):
        trial_ids = list(blob.keys())
        emb = torch.stack([torch.as_tensor(blob[k]).flatten() for k in trial_ids])
    else:
        raise ValueError(f"Formato no reconocido en {embed_path}")

    emb = emb.float()
    if emb.ndim != 2 or emb.shape[1] != EMBED_DIM:
        raise ValueError(f"Embeddings shape inválido: {tuple(emb.shape)}, esperado (N, {EMBED_DIM})")
    if len(trial_ids) != emb.shape[0]:
        raise ValueError(f"trial_ids ({len(trial_ids)}) != filas de embeddings ({emb.shape[0]})")
    
    # Fix: Efecto Shrinkage de Ridge.
    # Ridge aplasta la magnitud del vector (norma). Debemos restaurar la norma 
    # para que SD 2.1 unCLIP no ignore el vector considerándolo "vacío".
    import torch.nn.functional as F
    emb = F.normalize(emb, p=2, dim=-1) * 12.0 # ~12.0 es la norma promedio de CLIP ViT-L/14

    return trial_ids, emb


def align_stems_to_embeddings(subject: str, n_rows: int) -> list[str]:
    """
    Re-deriva los stems del test set y verifica alineación con las filas del
    tensor de embeddings (que siguen el orden de `Split.trial_ids_test` =
    índices densos 0..N-1 sobre `sorted(test_idx_by_stem)`).
    """
    stems = get_ordered_test_stems(subject)
    if len(stems) != n_rows:
        raise ValueError(
            f"Mismatch embeds vs stems test: filas={n_rows} "
            f"pero get_ordered_test_stems({subject}) devolvió {len(stems)}. "
            f"Re-entrena el adapter con los mismos paths que resolviste aquí."
        )
    return stems


# ---------------------------------------------------------------------------
# Ground Truth lookup
# ---------------------------------------------------------------------------

def find_ground_truth(stimuli_root: Path, stem: str) -> Path | None:
    """Recursivo en COCO/ImageNet/Scene: primer match con extensión válida."""
    for cand in stimuli_root.rglob(f"{stem}.*"):
        if cand.is_file() and cand.suffix.lower() in VALID_IMG_EXT:
            return cand
    return None


# ---------------------------------------------------------------------------
# Render (par individual + grid agregado)
# ---------------------------------------------------------------------------

def render_pair(gt_path: Path, recon: Image.Image, out_path: Path, stem: str, dpi: int) -> None:
    # API orientada a objetos (Figure, no pyplot): pyplot mantiene estado global
    # NO thread-safe. Como este render se ejecuta dentro del ThreadPoolExecutor
    # (concurrente con la GPU), usamos Figure + FigureCanvasAgg aislados por hilo.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    gt = Image.open(gt_path).convert("RGB")
    fig = Figure(figsize=(8, 4.4))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    axes[0].imshow(gt)
    axes[0].set_title("Estímulo Original", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(recon)
    axes[1].set_title("Reconstrucción (SD 2.1 unCLIP)", fontsize=11)
    axes[1].axis("off")
    fig.suptitle(stem, fontsize=9, y=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")


def render_grid(
    items: list[tuple[str, Path, Path]],
    out_path: Path,
    max_rows: int,
    dpi: int,
) -> None:
    """
    items: [(stem, gt_path, recon_path), ...]  (ambas existen en disco)

    Grid 2 columnas (GT | Recon) × N filas. Cap a `max_rows`.
    """
    items = items[:max_rows]
    if not items:
        logger.warning("Grid vacío, se omite.")
        return

    n = len(items)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, (stem, gt_path, recon_path) in enumerate(items):
        gt = Image.open(gt_path).convert("RGB")
        rc = Image.open(recon_path).convert("RGB")
        axes[i, 0].imshow(gt); axes[i, 0].axis("off")
        axes[i, 1].imshow(rc); axes[i, 1].axis("off")
        if i == 0:
            axes[i, 0].set_title("Estímulo Original", fontsize=11)
            axes[i, 1].set_title("Reconstrucción", fontsize=11)
        axes[i, 0].set_ylabel(stem, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def _dummy_recon_from_embed(embed: torch.Tensor, size: int = 256, seed: int = 0) -> Image.Image:
    """Stub determinista — PIL RGB ruidoso condicionado por el hash del embed. Solo dry-run."""
    sig = int((embed.detach().float().flatten().sum() * 1e3).item()) ^ seed
    rng = np.random.default_rng(abs(sig) & 0xFFFFFFFF)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


# Clase de excepcion de OOM (no existe en torch muy viejos -> fallback a RuntimeError).
_OOM_ERR = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)


def _is_oom(exc: Exception) -> bool:
    return isinstance(exc, _OOM_ERR) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def auto_batch_size(device: torch.device) -> int:
    """Elige un batch agresivo-pero-seguro segun la VRAM total detectada.
    Pensado para SD 2.1 unCLIP a 768px en bf16. Escala por tier para llenar la
    GPU sin conocer el modelo exacto (4050 6GB, 4070 8/12GB, 4070Ti Super 16GB...)."""
    if device.type != "cuda":
        return 2
    try:
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    except Exception:
        return 2
    if total_gb >= 15:   # 16GB+  (4070 Ti Super, 4080...)
        return 6
    if total_gb >= 11:   # 12GB   (4070 / 4070 Ti desktop)
        return 4
    if total_gb >= 7:    # 8GB    (4070 laptop)
        return 3
    return 2             # 6GB    (4050 laptop)


def _generate_oom_safe(
    reconstruct_fn,
    pipeline,
    embeds: list[torch.Tensor],
    prompts: list[str],
    *,
    device: torch.device,
    num_inference_steps: int,
    guidance_scale: float,
    noise_level: int,
    seed: int,
    subject: str,
) -> list[Image.Image]:
    """Genera un (sub)lote en GPU con red anti-OOM: si CUDA se queda sin memoria,
    vacia cache y reintenta partiendo el lote a la mitad, recursivamente, hasta 1.
    Asi un batch demasiado agresivo se auto-corrige en vez de matar el proceso a
    mitad de la grabacion."""
    try:
        stacked = torch.stack(embeds).to(device=device, dtype=pipeline.unet.dtype)
        imgs = reconstruct_fn(
            pipeline,
            stacked,
            prompt=prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_level=noise_level,
            seed=seed,
        )
        del stacked
        return imgs if isinstance(imgs, list) else [imgs]
    except Exception as exc:
        if not _is_oom(exc) or len(embeds) == 1:
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        mid = len(embeds) // 2
        logger.warning(
            f"[{subject}] OOM en sub-lote de {len(embeds)} -> reintento partido "
            f"{mid}+{len(embeds) - mid}"
        )
        common = dict(device=device, num_inference_steps=num_inference_steps,
                      guidance_scale=guidance_scale, noise_level=noise_level,
                      seed=seed, subject=subject)
        left = _generate_oom_safe(reconstruct_fn, pipeline, embeds[:mid], prompts[:mid], **common)
        right = _generate_oom_safe(reconstruct_fn, pipeline, embeds[mid:], prompts[mid:], **common)
        return left + right


def run_evaluation(
    subject: str,
    embeds_path: Path,
    stimuli_root: Path,
    out_base: Path,
    num_inference_steps: int,
    guidance_scale: float,
    noise_level: int,
    seed: int,
    limit: int | None,
    empty_cache_every: int,
    dpi: int,
    grid_rows: int,
    use_cpu: bool,
    batch_size: int | None = None,
    save_workers: int = 4,
    dry_run: bool = False,
) -> dict:
    trial_ids, embeds = load_adapter_embeddings(embeds_path)
    stems = align_stems_to_embeddings(subject, embeds.shape[0])

    if limit is not None:
        trial_ids = trial_ids[:limit]
        embeds = embeds[:limit]
        stems = stems[:limit]

    device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")

    # Batch: None => auto por VRAM (throughput agresivo, con red anti-OOM abajo).
    if batch_size is None:
        batch_size = auto_batch_size(device)
        if device.type == "cuda":
            try:
                props = torch.cuda.get_device_properties(device)
                logger.info(f"GPU={props.name} | VRAM={props.total_memory / (1024**3):.1f}GB "
                            f"| batch_size auto={batch_size}")
            except Exception:
                logger.info(f"batch_size auto={batch_size}")
        else:
            logger.info(f"batch_size auto={batch_size} (CPU)")
    batch_size = max(1, int(batch_size))

    logger.info(f"device={device} | N={len(stems)} | steps={num_inference_steps} | cfg={guidance_scale} | dry_run={dry_run}")

    subj_root = out_base / subject
    recon_dir = subj_root / "reconstructions"
    pairs_dir = subj_root / "pairs"
    recon_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    pipeline = None
    reconstruct_fn = None
    sd_prior_prompts: list[str] = [""]
    if not dry_run:
        # Import tardío: dry-run no requiere diffusers/transformers/accelerate instalados.
        from diffusers import DPMSolverMultistepScheduler
        from sd_decoder import (
            SD_PRIOR_PROMPTS,
            load_sd_unclip_pipeline,
            reconstruct_from_embedding,
        )
        sd_prior_prompts = list(SD_PRIOR_PROMPTS)
        reconstruct_fn = reconstruct_from_embedding
        pipeline = load_sd_unclip_pipeline(device=device, seed=seed)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        logger.info(f"Scheduler: {type(pipeline.scheduler).__name__}")
    else:
        logger.warning("DRY-RUN activo: SD 2.1 unCLIP NO se carga. Recon = PIL sintético.")

    ok = missing_gt = failed = 0
    # (orden, stem, gt_path, recon_path) — se ordena al final para grid determinista
    collage_ordered: list[tuple[int, str, Path, Path]] = []
    t0 = time.perf_counter()

    def _postprocess(order: int, stem: str, recon_img: Image.Image, newly_generated: bool):
        """Corre en hilos de CPU EN PARALELO con la GPU: guarda el PNG de la
        reconstrucción, busca el ground-truth y renderiza el collage comparativo.
        Devolver el `order` permite reordenar el grid de forma determinista."""
        recon_path = recon_dir / f"{stem}_recon.png"
        if newly_generated:
            recon_img.save(recon_path)
        gt_path = find_ground_truth(stimuli_root, stem)
        if gt_path is not None:
            pair_path = pairs_dir / f"{stem}_compare.png"
            render_pair(gt_path, recon_img, pair_path, stem, dpi=dpi)
        return order, stem, gt_path, recon_path

    work = list(zip(stems, embeds))
    n = len(work)
    n_batches = math.ceil(n / batch_size) if n else 0
    logger.info(
        f"[{subject}] paralelizado: {n} imgs | batch_size={batch_size} | "
        f"save_workers={save_workers} | {n_batches} lotes GPU"
    )

    futures = []
    with ThreadPoolExecutor(max_workers=save_workers) as pool:
        # torch.inference_mode envuelve SOLO la generación en GPU (hilo principal).
        with torch.inference_mode():
            for b in range(n_batches):
                chunk = work[b * batch_size : (b + 1) * batch_size]
                # Los que ya existen en disco se releen; solo se generan los faltantes.
                gen_local = [
                    k for k, (stem, _) in enumerate(chunk)
                    if not (recon_dir / f"{stem}_recon.png").exists()
                ]
                recon_imgs: list[Image.Image | None] = [None] * len(chunk)

                for k, (stem, _) in enumerate(chunk):
                    if k not in gen_local:
                        recon_imgs[k] = Image.open(recon_dir / f"{stem}_recon.png").convert("RGB")

                # --- Generación en LOTE (paralelismo de datos en GPU / SIMD) ---
                if gen_local:
                    try:
                        if dry_run:
                            for k in gen_local:
                                stem, emb = chunk[k]
                                recon_imgs[k] = _dummy_recon_from_embed(
                                    emb, seed=seed + b * batch_size + k
                                )
                        else:
                            sub_prompts = [
                                sd_prior_prompts[(b * batch_size + k) % len(sd_prior_prompts)]
                                for k in gen_local
                            ]
                            # Generacion en lote con red anti-OOM (parte a la mitad si revienta).
                            imgs = _generate_oom_safe(
                                reconstruct_fn,
                                pipeline,
                                [chunk[k][1] for k in gen_local],
                                sub_prompts,
                                device=device,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                noise_level=noise_level,
                                seed=seed,
                                subject=subject,
                            )
                            for j, k in enumerate(gen_local):
                                recon_imgs[k] = imgs[j]
                    except Exception as exc:
                        failed += len(gen_local)
                        bad = [chunk[k][0] for k in gen_local]
                        logger.error(f"[{subject}] fallo en lote {b} {bad}: {exc}")

                # --- Encolar post-proceso (I/O concurrente con la siguiente GPU) ---
                for k, (stem, _) in enumerate(chunk):
                    if recon_imgs[k] is None:
                        continue  # la generación falló para este item
                    order = b * batch_size + k
                    futures.append(
                        pool.submit(_postprocess, order, stem, recon_imgs[k], k in gen_local)
                    )

                # empty_cache cada ~empty_cache_every imágenes (traducido a lotes)
                cadence = max(1, empty_cache_every // batch_size) if empty_cache_every > 0 else 0
                if device.type == "cuda" and cadence and (b + 1) % cadence == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

        # Drenar el pool: la GPU ya terminó, recolectamos los resultados de I/O.
        for fut in futures:
            try:
                order, stem, gt_path, recon_path = fut.result()
                if gt_path is None:
                    missing_gt += 1
                    logger.warning(f"[{subject}] GT no hallado: {stem}")
                else:
                    collage_ordered.append((order, stem, gt_path, recon_path))
                    ok += 1
            except Exception as exc:
                failed += 1
                logger.error(f"[{subject}] post-proceso falló: {exc}")

    # Grid determinista: reordenar por índice original antes de renderizar.
    collage_ordered.sort(key=lambda x: x[0])
    collage_items = [(stem, gt, rc) for _, stem, gt, rc in collage_ordered]
    grid_path = subj_root / f"{subject}_grid.png"
    render_grid(collage_items, grid_path, max_rows=grid_rows, dpi=dpi)

    dt = time.perf_counter() - t0
    per_img = dt / max(ok + failed, 1)
    summary = {
        "subject": subject,
        "n_total": len(stems),
        "ok": ok,
        "missing_gt": missing_gt,
        "failed": failed,
        "seconds": round(dt, 1),
        "sec_per_img": round(per_img, 2),
        "out_dir": str(subj_root),
        "grid": str(grid_path) if grid_path.exists() else None,
    }
    logger.info(f"[{subject}] done — {summary}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluador visual Fase 2: embeds adapter → SD 2.1 unCLIP → GT vs Recon."
    )
    ap.add_argument("--subject", required=True, choices=config.BOLD5000_SUBJECTS,
                    help="Sujeto BOLD5000 (CSI1..CSI4).")
    ap.add_argument("--embeds", type=Path, default=None,
                    help="Override del .pt del adapter. "
                         "Default: {phase2_outputs}/adapter/{subject}/embeds_test.pt")
    ap.add_argument("--stimuli-root", type=Path, default=None,
                    help="Override raíz de estímulos BOLD5000. "
                         "Default: config.BOLD5000_CONFIG['stimuli_images']")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Raíz de salida. Default: config.DATA_DIRS['eval_output']")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap en # de estímulos (debug).")
    ap.add_argument("--steps", type=int, default=int(config.SD_CONFIG["num_inference_steps"]))
    ap.add_argument("--guidance", type=float, default=float(config.SD_CONFIG["guidance_scale"]))
    ap.add_argument("--noise-level", type=int, default=int(config.SD_CONFIG["noise_level"]))
    ap.add_argument("--seed", type=int, default=int(config.SD_CONFIG["seed"]))
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--grid-rows", type=int, default=12,
                    help="Filas máximas en el grid agregado.")
    ap.add_argument("--empty-cache-every", type=int, default=4,
                    help="Cada N imágenes llama torch.cuda.empty_cache() (0 = off).")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Imágenes por lote GPU (paralelismo de datos). "
                         "Default: AUTO por VRAM detectada (6→2, 8→3, 12→4, 16→6). "
                         "Si el batch es muy alto y hay OOM, el lote se parte a la mitad "
                         "automáticamente. Pasa un entero para forzar.")
    ap.add_argument("--save-workers", type=int, default=4,
                    help="Hilos de CPU para guardado/collage async concurrente con la GPU.")
    ap.add_argument("--cpu", action="store_true", help="Fuerza CPU (fp32, lento).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip SD pipeline load. Usa stub PIL (valida IO y shapes sin diffusers).")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    embeds_path = args.embeds or _default_embeds_path(args.subject)
    stimuli_root = args.stimuli_root or config.BOLD5000_CONFIG["stimuli_images"]
    out_base = args.out_dir or config.DATA_DIRS["eval_output"]

    logger.info(f"subject     = {args.subject}")
    logger.info(f"embeds      = {embeds_path}")
    logger.info(f"stimuli_root= {stimuli_root}")
    logger.info(f"out_base    = {out_base}")

    summary = run_evaluation(
        subject=args.subject,
        embeds_path=Path(embeds_path),
        stimuli_root=Path(stimuli_root),
        out_base=Path(out_base),
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        noise_level=args.noise_level,
        seed=args.seed,
        limit=args.limit,
        empty_cache_every=args.empty_cache_every,
        dpi=args.dpi,
        grid_rows=args.grid_rows,
        use_cpu=args.cpu,
        batch_size=args.batch_size,
        save_workers=args.save_workers,
        dry_run=args.dry_run,
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
