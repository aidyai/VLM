import os
from pathlib import PurePosixPath
from typing import Union
import modal
from pathlib import Path


APP_NAME = "usem-ocr"
MINUTES = 60  # seconds
HOURS = 60 * MINUTES
ALLOW_WANDB = os.environ.get("ALLOW_WANDB", "false").lower() == "true"

app = modal.App()


# Image and volume setup
cuda_version = "12.1.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Container image setup
ocr_vlm = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "ninja", "packaging", "wheel", "torch", "accelerate",
        "datasets", "bitsandbytes", "peft", "trl", 
        "huggingface_hub", "wandb", "deepspeed", 
        "transformers", "einops"
    ).env(
        dict(
            HF_HOME="/pretrained",
            HF_HUB_ENABLE_HF_TRANSFER="1",
        )
    )
)


VOLUME_NAME = "pretrained-model"
outputs = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
MODEL_PATH = Path("/pretrained")  # remote path for saving video outputs

MODEL_VOLUME_NAME = "checkpoint-model"
model = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
OUTPUTS_PATH = Path("/checkpoint")  # remote path for saving model weights


with image.imports():
    import torch
    from huggingface_hub import snapshot_download


@app.function(
    image=ocr_vlm,
    volumes={
        MODEL_PATH: outputs,
    },
    timeout=20 * MINUTES,
)

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct" 
# def download_model(revision="83359d26a7e2bbe200ecbfda8ebff850fd03b545"):
#     # uses HF_HOME to point download to the model volume
#     AutoModelForVision2Seq.from_pretrained(
#         MODEL_NAME,
#         torch_dtype=torch.bfloat16,
#         revision=revision,
#     )
MODEL_REVISION = "a1d521368f8d353afa4da2ed2bb1bf646ef1ff5f"

# Ensure base model is downloaded
try:
    snapshot_download(
        MODEL_NAME,
        local_files=True,
    )
    print(f"Volume contains {MODEL_NAME}.")
except FileNotFoundError:
    print(f"Downloading {MODEL_NAME} ...")
    snapshot_download(MODEL_NAME, local_dir=MODEL_PATH)
    print("Committing /pretrained directory ...")
    outputs.commit()
