import runpod
import uuid
import os
import logging
import shutil
import torch
from pathlib import Path

from utils.s3 import download_file, upload_file
from utils.utllity import (
    load_environment,
    reset_runtime_env,
    classify_env,
)
from utils.video import load_pipe, generate_lipsync

logging.basicConfig(level=logging.INFO)

_ = load_pipe()  # preload LatentSync
SEED = 1247


def handler(event):
    workdir = None

    try:
        reset_runtime_env()
        inp = event["input"]

        # ---- Info mode ----
        if inp.get("aleef") is True:
            return {
                "service": "latentsync-1.6",
                "version": "1.0",
                "inputs": ["ref_video_path", "ref_audio_path", "level"],
            }

        ref_video = inp["ref_video_path"]
        ref_audio = inp["ref_audio_path"]
        level = inp.get("level", None)

        # ---- Environment selection ----
        if not level:
            try:
                _, _, bucket, *_ = ref_video.split("/")
                level = classify_env(bucket)
                load_environment(level)
            except Exception:
                load_environment()  # default

        # ---- Working directory ----
        workdir = Path("/tmp") / str(uuid.uuid4())
        workdir.mkdir(parents=True, exist_ok=True)

        local_video = workdir / "input.mp4"
        local_audio = workdir / "input.wav"
        output_video = workdir / "output.mp4"
        temp_dir = workdir / "temp"

        # ---- Download inputs ----
        download_file(ref_video, str(local_video))
        download_file(ref_audio, str(local_audio))

        logging.info("🎬 Starting lip-sync generation")

        generate_lipsync(
            video_path=str(local_video),
            audio_path=str(local_audio),
            output_path=str(output_video),
            temp_dir=str(temp_dir),
            seed=SEED,
        )

        # ---- Upload result ----
        key = f"video_gen/latentsync/{uuid.uuid4()}.mp4"
        s3_path = upload_file(str(output_video), key)

        return {"video_path": s3_path}

    except Exception as e:
        logging.exception("❌ Lip-sync generation failed")
        return {"error": str(e)}

    finally:
        # ---- Cleanup files ----
        if workdir and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)

        # ---- Cleanup GPU ----
        torch.cuda.empty_cache()


runpod.serverless.start({"handler": handler})
