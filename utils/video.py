import torch
import pprint
from pathlib import Path
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature
from DeepCache import DeepCacheSDHelper

# ---------------------------
# Globals
# ---------------------------
PIPE = None
CONFIG = None
DTYPE = None
DEVICE = "cuda"

# ---------------------------
# Resolve paths robustly
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # /app
LATENTSYNC_DIR = BASE_DIR / "LatentSync"

CONFIG_PATH = LATENTSYNC_DIR / "configs" / "unet" / "stage2_512.yaml"
SCHEDULER_DIR = LATENTSYNC_DIR / "configs"
UNET_CKPT = BASE_DIR / "checkpoints" / "latentsync_unet.pt"
WHISPER_TINY = BASE_DIR / "checkpoints" / "whisper" / "whisper" / "tiny.pt"
WHISPER_SMALL = BASE_DIR / "checkpoints" / "whisper" / "small.pt"
MASK_PATH = LATENTSYNC_DIR / "latentsync" / "utils" / "mask.png"


def load_pipe():
    """
    Load LatentSync pipeline once (idempotent).
    Safe for RunPod serverless warm reuse.
    """
    global PIPE, CONFIG, DTYPE

    if PIPE is not None:
        return PIPE

    # ---- Load config (CORRECT PATH) ----
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"LatentSync config not found: {CONFIG_PATH}")

    CONFIG = OmegaConf.load(str(CONFIG_PATH))

    # ---- Precision selection ----
    is_fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    DTYPE = torch.float16 if is_fp16 else torch.float32

    # ---- Scheduler (CORRECT DIR) ----
    scheduler = DDIMScheduler.from_pretrained(str(SCHEDULER_DIR))

    # ---- Whisper model selection ----
    if CONFIG.model.cross_attention_dim == 768:
        whisper_path = WHISPER_SMALL
    else:
        whisper_path = WHISPER_TINY

    if not whisper_path.exists():
        raise FileNotFoundError(f"Whisper model not found: {whisper_path}")

    audio_encoder = Audio2Feature(
        model_path=str(whisper_path),
        device=DEVICE,
        num_frames=CONFIG.data.num_frames,
        audio_feat_length=CONFIG.data.audio_feat_length,
    )

    # ---- VAE ----
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=DTYPE,
    ).to(DEVICE)

    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    # ---- UNet ----
    if not UNET_CKPT.exists():
        raise FileNotFoundError(f"LatentSync UNet checkpoint not found: {UNET_CKPT}")

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(CONFIG.model),
        str(UNET_CKPT),
        device="cpu",
    )
    unet = unet.to(dtype=DTYPE)

    # ---- Fix mask path (important) ----
    CONFIG.data.mask_image_path = str(MASK_PATH)

    # ---- Pipeline ----
    PIPE = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to(DEVICE)

    # ---- DeepCache ----
    helper = DeepCacheSDHelper(pipe=PIPE)
    helper.set_params(cache_interval=3, cache_branch_id=0)
    helper.enable()

    return PIPE


def generate_lipsync(
    video_path: str,
    audio_path: str,
    output_path: str,
    temp_dir: str,
    seed: int = 1247,
):
    pipe = load_pipe()

    if seed != -1:
        torch.manual_seed(seed)

    payload = {
        "video_path": video_path,
        "audio_path": audio_path,
        "video_out_path": output_path,
        "num_frames": CONFIG.data.num_frames,
        "num_inference_steps": 20,
        "guidance_scale": 1.5,
        "weight_dtype": DTYPE,
        "width": CONFIG.data.resolution,
        "height": CONFIG.data.resolution,
        "mask_image_path": CONFIG.data.mask_image_path,
        "temp_dir": temp_dir,
    }

    print("🚀 Pipeline arguments:")
    pprint.pprint(payload)

    pipe(
        video_path=video_path,
        audio_path=audio_path,
        video_out_path=output_path,
        num_frames=CONFIG.data.num_frames,
        num_inference_steps=20,
        guidance_scale=1.5,
        weight_dtype=DTYPE,
        width=CONFIG.data.resolution,
        height=CONFIG.data.resolution,
        mask_image_path=CONFIG.data.mask_image_path,
        temp_dir=temp_dir,
    )
