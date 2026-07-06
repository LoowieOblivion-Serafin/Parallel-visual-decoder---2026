"""
===============================================================================
FASE 2 — INFERENCIA SD 2.1 unCLIP DESDE EMBEDDINGS NSD
===============================================================================

Carga embeddings CLIP ViT-L/14 (768-d) producidos por el adapter fMRI→CLIP
sobre NSD, y los pasa al pipeline SD 2.1 unCLIP para reconstruir imágenes.

PRE-REQUISITO
-------------
El adapter fMRI→CLIP-ViT-L/14 debe estar entrenado y haber producido un
archivo .pt o .npy con la matriz (n_trials, 768) por sujeto. Ese paso vive
en `phase2/train_adapter.py` (pendiente; depende del acceso NSD).

USO
---
    python phase2_run_sd.py --subject sub01 --embeds path/to/embeds.pt
    python phase2_run_sd.py --subject sub01 --embeds path/to/embeds.pt --limit 5
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import sys
import time
from pathlib import Path

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from diffusers import DPMSolverMultistepScheduler

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from sd_decoder import (
    load_sd_unclip_pipeline,
    reconstruct_from_embedding,
    SD_PRIOR_PROMPTS,
)

GLOBAL_SEED = config.SD_CONFIG["seed"]
INFERENCE_STEPS = config.SD_CONFIG["num_inference_steps"]
GUIDANCE_SCALE = config.SD_CONFIG["guidance_scale"]
EMBED_DIM = config.SD_CONFIG["embedding_dim"]
OUTPUT_ROOT = config.DATA_DIRS["output"]

logger = logging.getLogger("phase2_run_sd")


def load_embeddings(embed_path: Path) -> dict[str, torch.Tensor]:
    """
    Carga embeddings 768-d producidos por el adapter fMRI→CLIP-ViT-L/14.

    Formato esperado:
        - .pt con dict {trial_id: tensor (768,)}, o
        - .pt con tensor (N, 768) + lista paralela de trial_ids almacenada
          como `trial_ids` en el mismo dict.
    """
    if not embed_path.exists():
        raise FileNotFoundError(f"Embeddings no encontrados: {embed_path}")

    data = torch.load(embed_path, map_location="cpu")

    if isinstance(data, dict) and "trial_ids" in data and "embeddings" in data:
        ids = data["trial_ids"]
        emb = data["embeddings"]
        if emb.shape[1] != EMBED_DIM:
            raise ValueError(f"Shape inválido: {emb.shape}, esperado (N, {EMBED_DIM})")
        out = {tid: emb[i] for i, tid in enumerate(ids)}
    elif isinstance(data, dict):
        out = {tid: t.flatten() for tid, t in data.items()}
        bad = [tid for tid, t in out.items() if t.numel() != EMBED_DIM]
        if bad:
            raise ValueError(f"{len(bad)} embeddings con dim != {EMBED_DIM}")
    else:
        raise ValueError(f"Formato no reconocido en {embed_path}")

    logger.info(f"Cargados {len(out)} embeddings desde {embed_path}")
    return out


def run_subject(
    pipeline,
    subject_id: str,
    embeddings: dict[str, torch.Tensor],
    output_dir: Path,
    num_inference_steps: int = INFERENCE_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    limit: int | None = None,
    batch_size: int = 4,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = list(embeddings.items())
    if limit is not None:
        items = items[:limit]

    # Filtrar solo los trials que no se han procesado aún
    to_process = []
    skipped_count = 0
    for trial_id, embed in items:
        out_path = output_dir / f"{subject_id}_{trial_id}_sd_unclip.png"
        if out_path.exists():
            skipped_count += 1
        else:
            to_process.append((trial_id, embed))

    if skipped_count > 0:
        logger.info(f"[{subject_id}] {skipped_count} imágenes ya existen. Saltando reconstrucción.")

    if not to_process:
        logger.info(f"[{subject_id}] Todas las imágenes ({len(items)}) ya existen.")
        return len(items)

    saved = skipped_count
    t_start = time.perf_counter()

    def _save_image_async(img, path, tid):
        try:
            img.save(path)
        except Exception as exc:
            logger.error(f"[{subject_id}] Fallo al guardar la imagen para {tid}: {exc}")

    # Creamos un pool de hilos para guardar las imágenes de forma asíncrona en disco
    # mientras la GPU sigue calculando el siguiente batch.
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i in range(0, len(to_process), batch_size):
            batch = to_process[i : i + batch_size]
            batch_trial_ids = [item[0] for item in batch]
            batch_embeds = torch.stack([item[1] for item in batch])  # Forma: (B, 768)

            try:
                # Mapeamos los prompts correspondientes
                batch_prompts = [
                    SD_PRIOR_PROMPTS[(skipped_count + i + k) % len(SD_PRIOR_PROMPTS)]
                    for k in range(len(batch))
                ]

                # Inferencia en lote en la GPU (paralelismo de datos)
                imgs = reconstruct_from_embedding(
                    pipeline,
                    batch_embeds,
                    prompt=batch_prompts,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=GLOBAL_SEED,
                )

                # Si es un lote de tamaño 1, la función devuelve una imagen sola.
                # Lo convertimos en lista para procesar de forma uniforme.
                if not isinstance(imgs, list):
                    imgs = [imgs]

                # Encolamos la escritura en disco de forma concurrente
                for img, trial_id in zip(imgs, batch_trial_ids):
                    out_path = output_dir / f"{subject_id}_{trial_id}_sd_unclip.png"
                    executor.submit(_save_image_async, img, out_path, trial_id)
                    saved += 1
                    logger.info(f"[{subject_id}] ({saved}/{len(items)}) {trial_id} → {out_path.name} (encolado para guardar)")

            except Exception as exc:
                logger.error(f"[{subject_id}] Fallo en el lote {batch_trial_ids}: {exc}")
                continue

    dt = time.perf_counter() - t_start
    processed_count = saved - skipped_count
    per_img = dt / max(processed_count, 1)
    logger.info(f"[{subject_id}] {saved}/{len(items)} completadas. "
                f"Procesadas {processed_count} en {dt:.1f}s ({per_img:.1f}s/img)")
    return saved


def main() -> int:
    ap = argparse.ArgumentParser(description="Fase 2 — inferencia SD 2.1 unCLIP desde NSD adapter")
    ap.add_argument("--subject", required=True, help="Identificador NSD (e.g. sub01)")
    ap.add_argument("--embeds", required=True, type=Path,
                    help="Ruta al .pt de embeddings 768-d")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--steps", type=int, default=INFERENCE_STEPS)
    ap.add_argument("--guidance", type=float, default=GUIDANCE_SCALE)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Tamaño de lote (batch size) para inferencia paralela en GPU (default: 4)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    logger.info(f"Device: {device} | seed={GLOBAL_SEED} | steps={args.steps} | cfg={args.guidance} | batch_size={args.batch_size}")

    pipeline = load_sd_unclip_pipeline(device=device, seed=GLOBAL_SEED)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    logger.info(f"Scheduler: {type(pipeline.scheduler).__name__}")

    embeddings = load_embeddings(args.embeds)

    subject_out_dir = OUTPUT_ROOT / args.subject
    total_saved = run_subject(
        pipeline,
        args.subject,
        embeddings,
        subject_out_dir,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        limit=args.limit,
        batch_size=args.batch_size,
    )

    logger.info(f"Fase 2 completada — {total_saved} imágenes en {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
