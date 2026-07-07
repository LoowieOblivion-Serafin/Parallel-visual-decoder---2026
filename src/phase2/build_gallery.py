"""
===============================================================================
phase2/build_gallery.py — Presentacion HTML autocontenida de pares reconstruidos
===============================================================================

Recorre la salida del `visual_evaluator` y arma un unico archivo HTML con las
parejas [Estimulo Original] vs [Reconstruccion SD 2.1 unCLIP] embebidas como
data-URI base64. El resultado abre en cualquier navegador sin servidor ni rutas
externas: ideal para exponer la tesis / proyecto de Programacion Paralela.

ENTRADA (estructura que escribe visual_evaluator.py)
----------------------------------------------------
    {eval_dir}/{subject}/reconstructions/{stem}_recon.png
    {eval_dir}/{subject}/{subject}_grid.png            (opcional)

El ground truth se resuelve con rglob sobre `stimuli_root` (COCO/ImageNet/Scene),
igual que el evaluador. Si no se halla, la tarjeta muestra solo la reconstruccion.

USO
---
    # Todos los sujetos presentes en eval_dir, resuelve paths desde config
    python -m phase2.build_gallery

    # Un sujeto, thumbnails mas grandes, no abrir navegador
    python -m phase2.build_gallery --subject CSI1 --thumb 512 --no-open

    # Overrides explicitos
    python -m phase2.build_gallery --eval-dir output_reconstruccions_test2 \\
        --stimuli-root ../BOLD5000_Stimuli/.../Presented_Stimuli
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import logging
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

logger = logging.getLogger("phase2.build_gallery")

VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RECON_SUFFIX = "_recon.png"


# ---------------------------------------------------------------------------
# Resolucion de datos
# ---------------------------------------------------------------------------

def find_ground_truth(stimuli_root: Path, stem: str) -> Path | None:
    """Primer match con extension valida en COCO/ImageNet/Scene (recursivo)."""
    if not stimuli_root or not stimuli_root.is_dir():
        return None
    for cand in stimuli_root.rglob(f"{stem}.*"):
        if cand.is_file() and cand.suffix.lower() in VALID_IMG_EXT:
            return cand
    return None


def iter_subjects(eval_dir: Path, subject: str | None) -> list[str]:
    """Sujetos a incluir: el pedido, o todos los subdirs con reconstructions/."""
    if subject:
        return [subject]
    found = []
    for child in sorted(eval_dir.iterdir()):
        if child.is_dir() and (child / "reconstructions").is_dir():
            found.append(child.name)
    return found


def iter_recon_stems(subject_dir: Path) -> list[tuple[str, Path]]:
    """[(stem, recon_path)] ordenado, para cada {stem}_recon.png."""
    recon_dir = subject_dir / "reconstructions"
    out: list[tuple[str, Path]] = []
    if not recon_dir.is_dir():
        return out
    for f in sorted(recon_dir.glob(f"*{RECON_SUFFIX}")):
        stem = f.name[: -len(RECON_SUFFIX)]
        if stem:
            out.append((stem, f))
    return out


# ---------------------------------------------------------------------------
# Embebido base64 (con thumbnail para acotar el tamano del HTML)
# ---------------------------------------------------------------------------

def img_to_data_uri(path: Path, thumb: int) -> str:
    """Carga la imagen, la reescala a `thumb` px (lado mayor) y la devuelve como
    data-URI PNG base64. `thumb <= 0` conserva el tamano original."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if thumb and thumb > 0:
            im.thumbnail((thumb, thumb), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Render HTML
# ---------------------------------------------------------------------------

_CSS = """
:root { --bg:#0d1117; --card:#161b22; --border:#30363d; --fg:#e6edf3; --muted:#8b949e; --accent:#58a6ff; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font-family:system-ui,Segoe UI,Roboto,sans-serif; }
header { padding:28px 32px 12px; border-bottom:1px solid var(--border); position:sticky; top:0; background:var(--bg); z-index:10; }
h1 { margin:0 0 4px; font-size:22px; }
.meta { color:var(--muted); font-size:13px; }
.subject { padding:24px 32px 8px; }
.subject h2 { margin:0 0 4px; font-size:18px; color:var(--accent); }
.subject .stat { color:var(--muted); font-size:13px; margin-bottom:12px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:16px; padding:0 32px 8px; }
.card { background:var(--card); border:1px solid var(--border); border-radius:10px; overflow:hidden; }
.pair { display:grid; grid-template-columns:1fr 1fr; }
.pair figure { margin:0; }
.pair img { width:100%; display:block; aspect-ratio:1/1; object-fit:cover; background:#000; }
.pair figcaption { font-size:11px; text-align:center; padding:4px; color:var(--muted); }
.pair .gt figcaption { color:#3fb950; }
.pair .rc figcaption { color:var(--accent); }
.card .stem { font-size:11px; color:var(--muted); padding:6px 8px; border-top:1px solid var(--border); word-break:break-all; }
.gridimg { padding:8px 32px 32px; }
.gridimg img { max-width:100%; border:1px solid var(--border); border-radius:10px; }
footer { padding:24px 32px 40px; color:var(--muted); font-size:12px; border-top:1px solid var(--border); }
"""


def build_card(stem: str, gt_uri: str | None, rc_uri: str) -> str:
    gt_fig = (
        f'<figure class="gt"><img src="{gt_uri}" alt="GT {html.escape(stem)}">'
        f"<figcaption>Original</figcaption></figure>"
        if gt_uri
        else '<figure class="gt"><figcaption>sin GT</figcaption></figure>'
    )
    rc_fig = (
        f'<figure class="rc"><img src="{rc_uri}" alt="Recon {html.escape(stem)}">'
        f"<figcaption>Reconstruccion</figcaption></figure>"
    )
    return (
        '<div class="card"><div class="pair">'
        + gt_fig + rc_fig
        + f'</div><div class="stem">{html.escape(stem)}</div></div>'
    )


def build_html(
    eval_dir: Path,
    stimuli_root: Path | None,
    subjects: list[str],
    thumb: int,
    limit: int | None,
) -> tuple[str, int, int]:
    """Devuelve (html, total_pares, total_sin_gt)."""
    sections: list[str] = []
    total = 0
    total_missing = 0

    for subject in subjects:
        subj_dir = eval_dir / subject
        stems = iter_recon_stems(subj_dir)
        if limit is not None:
            stems = stems[:limit]
        if not stems:
            logger.warning(f"[{subject}] sin reconstrucciones en {subj_dir}")
            continue

        cards: list[str] = []
        missing = 0
        for stem, recon_path in stems:
            gt_path = find_ground_truth(stimuli_root, stem)
            try:
                rc_uri = img_to_data_uri(recon_path, thumb)
                gt_uri = img_to_data_uri(gt_path, thumb) if gt_path else None
                if gt_uri is None:
                    missing += 1
                cards.append(build_card(stem, gt_uri, rc_uri))
            except Exception as exc:
                logger.error(f"[{subject}] fallo embebiendo {stem}: {exc}")

        total += len(cards)
        total_missing += missing

        grid_png = subj_dir / f"{subject}_grid.png"
        grid_block = ""
        if grid_png.is_file():
            try:
                grid_uri = img_to_data_uri(grid_png, thumb=0)
                grid_block = (
                    f'<div class="gridimg"><img src="{grid_uri}" '
                    f'alt="grid {subject}"></div>'
                )
            except Exception as exc:
                logger.warning(f"[{subject}] grid no embebido: {exc}")

        sections.append(
            f'<section class="subject"><h2>{html.escape(subject)}</h2>'
            f'<div class="stat">{len(cards)} pares · {missing} sin GT</div></section>'
            f'<div class="grid">{"".join(cards)}</div>'
            + grid_block
        )

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    doc = (
        "<!doctype html><html lang='es'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Reconstruccion Visual — Pares GT vs SD 2.1 unCLIP</title>"
        f"<style>{_CSS}</style></head><body>"
        "<header><h1>Reconstruccion Visual en Paralelo — BOLD5000 → SD 2.1 unCLIP</h1>"
        f"<div class='meta'>Generado {stamp} · {len(subjects)} sujeto(s) · "
        f"{total} pares · {total_missing} sin ground truth</div></header>"
        + "".join(sections)
        + "<footer>visual_evaluator (batch GPU + I/O async) · "
        "build_gallery.py · V1sual_DecoderV2</footer>"
        "</body></html>"
    )
    return doc, total, total_missing


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Genera una presentacion HTML autocontenida de los pares reconstruidos."
    )
    ap.add_argument("--subject", default=None, choices=config.BOLD5000_SUBJECTS,
                    help="Un solo sujeto. Default: todos los presentes en eval-dir.")
    ap.add_argument("--eval-dir", type=Path, default=None,
                    help="Raiz de salida del evaluador. Default: config.DATA_DIRS['eval_output'].")
    ap.add_argument("--stimuli-root", type=Path, default=None,
                    help="Raiz de estimulos para el GT. Default: config.BOLD5000_CONFIG['stimuli_images'].")
    ap.add_argument("--out", type=Path, default=None,
                    help="Ruta del HTML. Default: {eval-dir}/presentacion_pares.html")
    ap.add_argument("--thumb", type=int, default=384,
                    help="Lado mayor del thumbnail embebido en px (0 = tamano original). Default 384.")
    ap.add_argument("--limit", type=int, default=None, help="Cap de pares por sujeto.")
    ap.add_argument("--no-open", action="store_true", help="No abrir el navegador al terminar.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    eval_dir = (args.eval_dir or config.DATA_DIRS["eval_output"]).resolve()
    stimuli_root = args.stimuli_root or config.BOLD5000_CONFIG["stimuli_images"]
    stimuli_root = Path(stimuli_root) if stimuli_root else None
    out_path = args.out or (eval_dir / "presentacion_pares.html")

    if not eval_dir.is_dir():
        logger.error(f"eval-dir no existe: {eval_dir}. Corre primero visual_evaluator / render_rtx4050.ps1.")
        return 1

    subjects = iter_subjects(eval_dir, args.subject)
    if not subjects:
        logger.error(f"Sin sujetos con reconstrucciones bajo {eval_dir}.")
        return 1

    logger.info(f"eval_dir     = {eval_dir}")
    logger.info(f"stimuli_root = {stimuli_root}")
    logger.info(f"subjects     = {subjects}")

    doc, total, missing = build_html(eval_dir, stimuli_root, subjects, args.thumb, args.limit)
    if total == 0:
        logger.error("No se embebio ningun par. Nada que presentar.")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Presentacion escrita: {out_path} ({size_mb:.1f} MB · {total} pares · {missing} sin GT)")

    if not args.no_open:
        try:
            webbrowser.open(out_path.as_uri())
        except Exception as exc:
            logger.warning(f"No se pudo abrir el navegador: {exc}. Abre manualmente: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
