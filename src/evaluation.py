"""
===============================================================================
EVALUACIÓN CUANTITATIVA DE RECONSTRUCCIONES — Fase 2 (NSD + SD 2.1 unCLIP)
===============================================================================

Calcula métricas estándar sobre reconstrucciones SD 2.1 unCLIP frente a
estímulos COCO referenciados por NSD trial_id.

Métricas:
    - SSIM           : similitud estructural
    - PixCorr        : correlación Pearson sobre píxeles
    - PSNR           : relación señal/ruido (dB)
    - LPIPS          : distancia perceptual aprendida
    - CLIP cosine    : similitud semántica (CLIP ViT-L/14)
    - Pairwise ID    : accuracy 2-vías (Scotti et al. 2023)

Uso:
    py -3.12 evaluation.py --recon-dir output_sd_reconstructions --gt-dir nsd/stimuli
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluation")

try:
    # pyrefly: ignore [missing-import]
    from skimage.metrics import structural_similarity as _ssim
    # pyrefly: ignore [missing-import]
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    log.warning("scikit-image no instalado; SSIM/PSNR se omitirán.")

try:
    # pyrefly: ignore [missing-import]
    from scipy.stats import pearsonr as _pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    log.warning("scipy no instalado; PixCorr se omitirá.")

try:
    # pyrefly: ignore [missing-import]
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    log.error("PyTorch no instalado. CLIP/LPIPS no disponibles.")

try:
    # pyrefly: ignore [missing-import]
    import lpips as _lpips_lib
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

HAS_CLIP = False
if HAS_TORCH:
    try:
        # pyrefly: ignore [missing-import]
        from transformers import CLIPModel, CLIPProcessor
        HAS_CLIP = True
    except ImportError:
        log.warning("transformers no disponible; CLIP cos omitido.")


IMG_SIZE = config.SD_CONFIG["image_size"]


@dataclass
class StimulusPair:
    label: str
    subject: str
    gt: np.ndarray
    recon: np.ndarray


def load_gt_images(gt_dir: Path) -> Dict[str, np.ndarray]:
    """Lee PNG/JPG de stimuli como {label: ndarray uint8 HxWx3}."""
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Directorio GT no encontrado: {gt_dir}")
    out: Dict[str, np.ndarray] = {}
    for img_path in sorted(gt_dir.glob("*")):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        out[img_path.stem] = np.asarray(Image.open(img_path).convert("RGB"))
    log.info("Cargados %d estímulos GT desde %s", len(out), gt_dir)
    return out


def load_reconstructions(subject: str, recon_root: Path) -> Dict[str, np.ndarray]:
    """Lee PNG `<subject>_<label>_sd_unclip.png` de `recon_root/<subject>/`."""
    subject_dir = recon_root / subject
    if not subject_dir.is_dir():
        log.warning("No encontrado: %s", subject_dir)
        return {}
    suffix = "_sd_unclip.png"
    prefix = f"{subject}_"
    out: Dict[str, np.ndarray] = {}
    for png in sorted(subject_dir.glob(f"*{suffix}")):
        name = png.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        label = name[len(prefix):-len(suffix)]
        out[label] = np.asarray(Image.open(png).convert("RGB"))
    log.info("Sujeto %s: %d reconstrucciones cargadas", subject, len(out))
    return out


def align_pairs(subject: str,
                stimuli: Dict[str, np.ndarray],
                recons: Dict[str, np.ndarray]) -> List[StimulusPair]:
    pairs: List[StimulusPair] = []
    for label, recon in recons.items():
        if label not in stimuli:
            continue
        gt = stimuli[label]
        if gt.shape[:2] != (IMG_SIZE, IMG_SIZE):
            gt = np.asarray(Image.fromarray(gt).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))
        if recon.shape[:2] != (IMG_SIZE, IMG_SIZE):
            recon = np.asarray(Image.fromarray(recon).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))
        pairs.append(StimulusPair(label=label, subject=subject, gt=gt, recon=recon))
    return pairs


@dataclass
class PerceptualModels:
    device: str
    lpips_fn: Optional[object] = None
    clip_model: Optional[object] = None
    clip_processor: Optional[object] = None


def build_models(device: Optional[str] = None) -> PerceptualModels:
    if not HAS_TORCH:
        return PerceptualModels(device="cpu")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    models = PerceptualModels(device=device)
    if HAS_LPIPS:
        log.info("Cargando LPIPS (AlexNet) en %s", device)
        models.lpips_fn = _lpips_lib.LPIPS(net="alex", verbose=False).to(device).eval()
    if HAS_CLIP:
        repo = config.SD_CONFIG["clip_target_repo"]
        log.info("Cargando CLIP %s en %s", repo, device)
        models.clip_model = CLIPModel.from_pretrained(repo).to(device).eval()
        models.clip_processor = CLIPProcessor.from_pretrained(repo)
    return models


def pixel_metrics(recon: np.ndarray, gt: np.ndarray) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {"ssim": None, "pixcorr": None, "psnr": None}
    if HAS_SKIMAGE:
        out["ssim"] = float(_ssim(gt, recon, channel_axis=2, data_range=255))
        psnr_val = float(_psnr(gt, recon, data_range=255))
        out["psnr"] = psnr_val if np.isfinite(psnr_val) else float("nan")
    if HAS_SCIPY:
        r, _ = _pearsonr(gt.flatten().astype(np.float64),
                         recon.flatten().astype(np.float64))
        out["pixcorr"] = float(r)
    return out


def _to_lpips_tensor(img: np.ndarray, device: str) -> "torch.Tensor":
    arr = np.ascontiguousarray(img)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 127.5 - 1.0
    return t.unsqueeze(0).to(device)


def lpips_metric(pair: StimulusPair, models: PerceptualModels) -> Optional[float]:
    if models.lpips_fn is None:
        return None
    with torch.no_grad():
        d = models.lpips_fn(
            _to_lpips_tensor(pair.recon, models.device),
            _to_lpips_tensor(pair.gt, models.device),
        )
    return float(d.item())


def _clip_embed(imgs: List[np.ndarray], models: PerceptualModels) -> "torch.Tensor":
    assert models.clip_model is not None and models.clip_processor is not None
    inputs = models.clip_processor(images=[Image.fromarray(im) for im in imgs],
                                   return_tensors="pt").to(models.device)
    with torch.no_grad():
        feats = models.clip_model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return feats


def clip_similarity(pairs: List[StimulusPair],
                    models: PerceptualModels) -> Dict[str, float]:
    if models.clip_model is None or not pairs:
        return {p.label: float("nan") for p in pairs}
    recons_feat = _clip_embed([p.recon for p in pairs], models)
    gts_feat = _clip_embed([p.gt for p in pairs], models)
    cos = (recons_feat * gts_feat).sum(dim=-1).detach().cpu().numpy()
    return {pair.label: float(c) for pair, c in zip(pairs, cos)}


def pairwise_identification(pairs: List[StimulusPair],
                            models: PerceptualModels) -> Dict[str, float]:
    n = len(pairs)
    if n < 2 or models.clip_model is None:
        return {"n": n, "pairwise_acc": float("nan")}
    recons_feat = _clip_embed([p.recon for p in pairs], models)
    gts_feat = _clip_embed([p.gt for p in pairs], models)
    sim = (recons_feat @ gts_feat.T).detach().cpu().numpy()
    diag = np.diag(sim)
    mask = np.ones_like(sim, dtype=bool)
    np.fill_diagonal(mask, False)
    wins = (diag[:, None] > sim).astype(np.float64)
    wins = wins[mask].reshape(n, n - 1)
    return {"n": n, "pairwise_acc": float(wins.mean())}


@dataclass
class PerImageRow:
    subject: str
    label: str
    ssim: Optional[float] = None
    pixcorr: Optional[float] = None
    psnr: Optional[float] = None
    lpips: Optional[float] = None
    clip_sim: Optional[float] = None


def evaluate_subject(subject: str,
                     stimuli: Dict[str, np.ndarray],
                     recon_root: Path,
                     models: PerceptualModels) -> Tuple[List[PerImageRow], Dict[str, float]]:
    recons = load_reconstructions(subject, recon_root)
    pairs = align_pairs(subject, stimuli, recons)
    rows: List[PerImageRow] = []
    clip_sims = clip_similarity(pairs, models)
    for p in pairs:
        px = pixel_metrics(p.recon, p.gt)
        rows.append(PerImageRow(
            subject=subject,
            label=p.label,
            ssim=px["ssim"],
            pixcorr=px["pixcorr"],
            psnr=px["psnr"],
            lpips=lpips_metric(p, models),
            clip_sim=clip_sims.get(p.label),
        ))
    summary = {
        "subject": subject,
        "n_pairs": len(pairs),
        **_aggregate(rows),
        **pairwise_identification(pairs, models),
    }
    return rows, summary


def _aggregate(rows: List[PerImageRow]) -> Dict[str, float]:
    keys = ["ssim", "pixcorr", "psnr", "lpips", "clip_sim"]
    out: Dict[str, float] = {}
    for k in keys:
        vals = [getattr(r, k) for r in rows
                if getattr(r, k) is not None and np.isfinite(getattr(r, k))]
        out[f"mean_{k}"] = float(np.mean(vals)) if vals else float("nan")
    return out


def write_csv(rows: List[PerImageRow], path: Path) -> None:
    if not rows:
        return
    header = list(asdict(rows[0]).keys())
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(
            "" if v is None else (f"{v:.6f}" if isinstance(v, float) else str(v))
            for v in asdict(r).values()
        ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Escrito %s", path)


def write_summary_csv(summaries: List[Dict[str, float]], path: Path) -> None:
    if not summaries:
        return
    header = list(summaries[0].keys())
    lines = [",".join(header)]
    for s in summaries:
        lines.append(",".join(
            f"{v:.6f}" if isinstance(v, float) and np.isfinite(v)
            else "" if v is None
            else str(v)
            for v in s.values()
        ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Escrito %s", path)


def main(subjects: Iterable[str],
         recon_root: Path,
         gt_dir: Path) -> None:
    recon_root.mkdir(parents=True, exist_ok=True)
    stimuli = load_gt_images(gt_dir)
    models = build_models()

    all_rows: List[PerImageRow] = []
    summaries: List[Dict[str, float]] = []

    for subject in subjects:
        rows, summary = evaluate_subject(subject, stimuli, recon_root, models)
        if not rows:
            log.warning("Sujeto %s: sin pares evaluables.", subject)
            continue
        all_rows.extend(rows)
        summaries.append(summary)
        write_csv(rows, recon_root / f"metrics_{subject}.csv")
        log.info(
            "[%s] n=%d · SSIM=%.3f · PixCorr=%.3f · CLIP=%.3f · pairwise=%.3f",
            subject, summary["n_pairs"],
            summary["mean_ssim"], summary["mean_pixcorr"],
            summary["mean_clip_sim"], summary["pairwise_acc"],
        )

    if summaries:
        write_summary_csv(summaries, recon_root / "metrics_summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluación SD 2.1 unCLIP sobre NSD")
    parser.add_argument("--subjects", nargs="+",
                        default=config.NSD_CONFIG["subjects"])
    parser.add_argument("--recon-dir", type=Path,
                        default=config.DATA_DIRS["output"])
    parser.add_argument("--gt-dir", type=Path,
                        default=config.NSD_CONFIG["stimuli_root"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.subjects, args.recon_dir, args.gt_dir)
