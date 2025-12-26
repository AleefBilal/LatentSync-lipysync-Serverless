import torch
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature
from DeepCache import DeepCacheSDHelper

PIPE = None
CONFIG = None
DTYPE = None
DEVICE = "cuda"


def load_pipe():
    """
    Loads LatentSync pipeline once.
    Safe to call multiple times.
    """
    global PIPE, CONFIG, DTYPE

    if PIPE is not None:
        return PIPE

    CONFIG = OmegaConf.load("LatentSync/configs/unet.yaml")

    is_fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    DTYPE = torch.float16 if is_fp16 else torch.float32

    scheduler = DDIMScheduler.from_pretrained("LatentSync/configs")

    # Whisper model
    if CONFIG.model.cross_attention_dim == 768:
        whisper_path = "checkpoints/whisper/small.pt"
    else:
        whisper_path = "checkpoints/whisper/tiny.pt"

    audio_encoder = Audio2Feature(
        model_path=whisper_path,
        device=DEVICE,
        num_frames=CONFIG.data.num_frames,
        audio_feat_length=CONFIG.data.audio_feat_length,
    )

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=DTYPE,
    ).to(DEVICE)

    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(CONFIG.model),
        "checkpoints/latentsync_unet.pt",
        device="cpu",
    )
    unet = unet.to(dtype=DTYPE)

    PIPE = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to(DEVICE)

    # Enable DeepCache
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

    pipe(
        video_path=video_path,
        audio_path=audio_path,
        video_out_path=output_path,
        num_frames=CONFIG.data.num_frames,
        num_inference_steps=20,
        guidance_scale=1.0,
        weight_dtype=DTYPE,
        width=CONFIG.data.resolution,
        height=CONFIG.data.resolution,
        mask_image_path=CONFIG.data.mask_image_path,
        temp_dir=temp_dir,
    )