"""
===============================================================================
DECODIFICADOR — STABLE DIFFUSION 2.1 unCLIP (Fase 2 NSD)
===============================================================================

Pipeline de inferencia condicionado por embedding CLIP ViT-L/14 (768-d)
predicho desde fMRI por el adapter NSD.

Arquitectura:

    fMRI (NSD) → Adapter (LoRA / Ridge) → z_CLIP ∈ R^768 → SD 2.1 unCLIP UNet → VAE dec → imagen

HARDWARE
--------
RTX 3070 (8 GB) o RTX 4070 Ti (12 GB). bf16 + xformers para ~5-6 GB en inferencia.

REFERENCIAS HF
--------------
- Modelo:   https://huggingface.co/diffusers/stable-diffusion-2-1-unclip-i2i-l
- Pipeline: diffusers.StableUnCLIPImg2ImgPipeline
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from diffusers import StableUnCLIPImg2ImgPipeline
from accelerate.utils import set_seed

import config

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES
# ============================================================================

SD_UNCLIP_REPO = config.SD_CONFIG["repo_id"]
SD_DTYPE = torch.bfloat16
SD_CACHE_DIR = config.DATA_DIRS["models_hf"]

# Negative prompt fijo para classifier-free guidance negativa.
SD_NEGATIVE_PROMPT = "blurry, noise, abstract, deformed, chaotic, multiple objects"

# Prompts de estabilización semántica vacíos para no sobreescribir el vector cerebral.
SD_PRIOR_PROMPTS = [
    "",
]


# ============================================================================
# CARGA DEL PIPELINE
# ============================================================================

def load_sd_unclip_pipeline(
    device: torch.device | str = "cuda",
    repo_id: str = SD_UNCLIP_REPO,
    cache_dir: Path | str | None = None,
    enable_xformers: bool = True,
    enable_vae_slicing: bool = True,
    seed: int | None = 42,
) -> StableUnCLIPImg2ImgPipeline:
    """
    Instancia el pipeline SD 2.1 unCLIP en bf16.

    Args:
        device: 'cuda' o torch.device. CPU fuerza fp32.
        repo_id: identificador HF Hub del modelo.
        cache_dir: cache local de pesos. Default: PROJECT_ROOT/models_hf/.
        enable_xformers: memory-efficient attention (CRÍTICO en 8/12 GB).
        enable_vae_slicing: decodifica latente en slices.
        seed: semilla global (set_seed). None para no fijar.

    Returns:
        StableUnCLIPImg2ImgPipeline en eval() sobre device, dtype bf16.
    """
    if isinstance(device, str):
        device = torch.device(device)

    if seed is not None:
        set_seed(seed)

    if cache_dir is None:
        cache_dir = SD_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dtype = SD_DTYPE if device.type == "cuda" else torch.float32

    logger.info(f"Cargando SD 2.1 unCLIP desde {repo_id} (dtype={dtype}, cache={cache_dir})")

    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
        repo_id,
        torch_dtype=dtype,
        cache_dir=str(cache_dir),
        variant="fp16" if dtype == torch.bfloat16 else None,
        use_safetensors=True,
    )

    if enable_xformers and device.type == "cuda":
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory-efficient attention habilitado")
        except Exception as exc:
            logger.warning(
                f"xformers no disponible ({exc}). Instalar con: pip install xformers."
            )

    if enable_vae_slicing:
        pipeline.enable_vae_slicing()

    pipeline.safety_checker = None
    pipeline = pipeline.to(device)

    pipeline.unet.eval()
    pipeline.vae.eval()
    pipeline.image_encoder.eval()

    logger.info(
        f"Pipeline listo | device={device} | dtype={dtype} | "
        f"xformers={enable_xformers} | vae_slicing={enable_vae_slicing}"
    )

    return pipeline


# ============================================================================
# INFERENCIA
# ============================================================================

def reconstruct_from_embedding(
    pipeline: StableUnCLIPImg2ImgPipeline,
    brain_clip_embedding: torch.Tensor,
    prompt: str = SD_PRIOR_PROMPTS[0],
    num_inference_steps: int = 30,
    guidance_scale: float = 10.0,
    noise_level: int = 0,
    negative_prompt: str = SD_NEGATIVE_PROMPT,
    seed: int | None = 42,
    output_height: int = 768,
    output_width: int = 768,
):
    """
    Reconstruye una imagen a partir de un embedding CLIP ViT-L/14 (768-d)
    predicho del fMRI.

    Por qué image_embeds y NO solo prompt:
        El objetivo científico es medir cuánta información visual se recupera
        del cerebro. Pasar el embedding del adapter por `image_embeds` salta
        el image_encoder de SD; el negative_prompt entra por el text_encoder
        sólo para CFG negativa (no aporta señal positiva externa).

    Args:
        pipeline: cargado por load_sd_unclip_pipeline().
        brain_clip_embedding: tensor (1, 768) o (768,) del adapter fMRI→CLIP.
        num_inference_steps: pasos del scheduler (mín. estable 25-30).
        guidance_scale: CFG. 10 es agresivo y estable.
        noise_level: ruido gaussiano sobre image_embed (0 = mínima varianza).
        seed: generador determinista por-llamada.
        output_height, output_width: nativo SD 2.1 unCLIP = 768.

    Returns:
        PIL.Image.Image
    """
    is_single_input = False
    if brain_clip_embedding.dim() == 1:
        brain_clip_embedding = brain_clip_embedding.unsqueeze(0)
        is_single_input = True
    elif brain_clip_embedding.dim() == 2 and brain_clip_embedding.shape[0] == 1:
        is_single_input = True

    if brain_clip_embedding.dim() != 2 or brain_clip_embedding.shape[1] != 768:
        raise ValueError(
            f"brain_clip_embedding debe tener shape (B, 768), (1, 768) o (768,); "
            f"recibido: {tuple(brain_clip_embedding.shape)}"
        )

    batch_size = brain_clip_embedding.shape[0]
    device = pipeline.unet.device
    target_dtype = pipeline.unet.dtype
    brain_clip_embedding = brain_clip_embedding.to(device=device, dtype=target_dtype)

    # Si prompt es un string, lo duplicamos para que coincida con el tamaño del lote.
    # Si ya es una lista, nos aseguramos de que coincida con el tamaño del lote.
    if isinstance(prompt, str):
        prompts = [prompt] * batch_size
    else:
        prompts = prompt
        if len(prompts) != batch_size:
            raise ValueError(f"La longitud de prompts ({len(prompts)}) no coincide con el batch_size ({batch_size})")

    if isinstance(negative_prompt, str):
        negative_prompts = [negative_prompt] * batch_size
    else:
        negative_prompts = negative_prompt
        if len(negative_prompts) != batch_size:
            raise ValueError(f"La longitud de negative_prompts ({len(negative_prompts)}) no coincide con el batch_size ({batch_size})")

    # Generamos semillas deterministas por-elemento en el lote
    generators = None
    if seed is not None:
        if batch_size == 1:
            generators = torch.Generator(device=device).manual_seed(int(seed))
        else:
            generators = [
                torch.Generator(device=device).manual_seed(int(seed) + i)
                for i in range(batch_size)
            ]

    with torch.no_grad():
        result = pipeline(
            prompt=prompts,
            image_embeds=brain_clip_embedding,
            negative_prompt=negative_prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_level=noise_level,
            generator=generators,
            height=output_height,
            width=output_width,
        )

    if is_single_input:
        return result.images[0]
    return result.images


# ============================================================================
# SMOKE TEST
# ============================================================================

def _smoke_test() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Smoke test en device={device}")

    pipeline = load_sd_unclip_pipeline(device=device)
    dummy_embed = torch.randn(1, 768, device=device)

    img = reconstruct_from_embedding(
        pipeline,
        dummy_embed,
        num_inference_steps=10,
        guidance_scale=10.0,
        seed=42,
    )

    logger.info(f"Smoke test OK — imagen shape={img.size}")


if __name__ == "__main__":
    _smoke_test()
