"""
===============================================================================
phase2/create_mock_assets.py — Ecosistema sintético para smoke-test CPU
===============================================================================

Genera un mini-BOLD5000 completo bajo `--root` para ejercitar el pipeline
visual_evaluator sin tocar datos reales ni requerir GPU.

ARTEFACTOS GENERADOS
--------------------
    {root}/stimuli/
        Scene_Stimuli/Presented_Stimuli/
            repeated_stimuli_113_list.txt
            COCO/    ImageNet/    Scene/mock_img_{i}.jpg   (N imágenes 64×64)
        Stimuli_Presentation_Lists/
            CSI1/CSI1_sess01/CSI1_sess01_run01.txt         (stims concatenados)
    {root}/outputs/
        clip_targets/bold5000_vitL14.pt                    (filenames + emb)
        adapter/CSI1/embeds_test.pt                        (trial_ids + emb)

El script imprime las variables `ACECOM_*` que se deben exportar para que
`config.py` resuelva rutas al sandbox en lugar del dataset real.

USO
---
    python -m phase2.create_mock_assets --root mock_assets --n 4 --subject CSI1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

EMBED_DIM = int(config.SD_CONFIG["embedding_dim"])


def _write_img(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path, format="JPEG", quality=85)


def create_mock_ecosystem(root: Path, n: int, subject: str, seed: int) -> dict[str, Path]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    stimuli_root = root / "stimuli"
    presented = stimuli_root / "Scene_Stimuli" / "Presented_Stimuli"
    lists_root = stimuli_root / "Stimuli_Presentation_Lists"
    outputs_root = root / "outputs"

    # Subdirs mandatorios (aunque no todos tengan imágenes)
    for sub in ("COCO", "ImageNet", "Scene"):
        (presented / sub).mkdir(parents=True, exist_ok=True)

    # Estímulos sintéticos (van bajo Scene/)
    stems = [f"mock_img_{i:04d}" for i in range(n)]
    filenames = [f"{stem}.jpg" for stem in stems]
    for i, fname in enumerate(filenames):
        _write_img(presented / "Scene" / fname, seed=seed + i)

    # Lista de repeated (loader tolera != 113 con warning).
    # OJO: config.bold5000_repeated_list_txt() la busca en Scene_Stimuli/ (un nivel
    # ARRIBA de Presented_Stimuli). Escribir aquí para que get_ordered_test_stems
    # la encuentre sin parches manuales.
    repeated_txt = stimuli_root / "Scene_Stimuli" / "repeated_stimuli_113_list.txt"
    repeated_txt.write_text("\n".join(filenames) + "\n", encoding="utf-8")

    # Presentation lists: una sesión, un run con todos los stems
    sess_dir = lists_root / subject / f"{subject}_sess01"
    sess_dir.mkdir(parents=True, exist_ok=True)
    (sess_dir / f"{subject}_sess01_run01.txt").write_text(
        "\n".join(filenames) + "\n", encoding="utf-8"
    )

    # CLIP targets (claves = filenames; stem se deriva dentro del loader)
    clip_embeds = torch.from_numpy(rng.standard_normal((n, EMBED_DIM)).astype("float32"))
    clip_path = outputs_root / "clip_targets" / "bold5000_vitL14.pt"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "filenames": filenames,
            "embeddings": clip_embeds,
            "model_id": "mock-openai/clip-vit-large-patch14",
            "dim": EMBED_DIM,
        },
        clip_path,
    )

    # Adapter embeddings (mismo formato que train_adapter.py)
    adapter_embeds = torch.from_numpy(rng.standard_normal((n, EMBED_DIM)).astype("float32"))
    embeds_path = outputs_root / "adapter" / subject / "embeds_test.pt"
    embeds_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"trial_ids": list(range(n)), "embeddings": adapter_embeds},
        embeds_path,
    )

    return {
        "stimuli_root": stimuli_root,
        "outputs_root": outputs_root,
        "clip_targets_pt": clip_path,
        "embeds_test_pt": embeds_path,
        "repeated_list_txt": repeated_txt,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Genera ecosistema mock BOLD5000 para smoke-test.")
    ap.add_argument("--root", type=Path, default=Path("mock_assets"),
                    help="Raíz del sandbox. Default: ./mock_assets")
    ap.add_argument("--n", type=int, default=4, help="# estímulos sintéticos.")
    ap.add_argument("--subject", default="CSI1", choices=config.BOLD5000_SUBJECTS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = args.root.resolve()
    paths = create_mock_ecosystem(root, n=args.n, subject=args.subject, seed=args.seed)

    eval_out = (root / "eval_out").resolve()
    print("Mock ecosystem ready:")
    for k, v in paths.items():
        print(f"  {k:<18} = {v}")
    print()
    print("Export (bash):")
    print(f"  export ACECOM_BOLD5000_STIMULI_ROOT={paths['stimuli_root']}")
    print(f"  export ACECOM_PHASE2_OUTPUTS={paths['outputs_root']}")
    print(f"  export ACECOM_EVAL_OUTPUT={eval_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
