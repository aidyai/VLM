import os
from datetime import datetime
from pathlib import Path
import secrets
import yaml
from uuid import uuid4
from modal import App, Volume, Image, gpu, Retries
from VLM.config.common import (
    app,
    ocr_vlm,
    HOURS,
    MINUTES,
    model,
    outputs,
    MODEL_PATH,
    OUTPUTS_PATH,
)


# GPU Configuration
GPU_CONFIG = os.environ.get("GPU_CONFIG", "a100:2")
if len(GPU_CONFIG.split(":")) <= 1:
    N_GPUS = int(os.environ.get("N_GPUS", 2))
    GPU_CONFIG = f"{GPU_CONFIG}:{N_GPUS}"


@app.function(
    image=ocr_vlm,
    gpu=GPU_CONFIG,
    volumes={
        OUTPUTS_PATH: model,  
        MODEL_PATH: outputs,
    },
    timeout=24 * HOURS,
)

def train(experiment=None):
    # Generate a unique experiment name if not provided
    if experiment is None:
        experiment = uuid4().hex[:8]
    
    
    import torch
    from huggingface_hub import login
    import wandb
    import subprocess
    
    HF_TOKEN = "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"
    WANDB_APIKEY = "0d505324ba165d96687f3624d4310bf171485b9d"
    WANDB_PROJECT = "usem_ocr"

    # Ensure sensitive data is set
    if not HF_TOKEN or not WANDB_APIKEY:
        raise ValueError("Please set Hugging Face and WandB tokens in your environment.")

    current_dir = Path(__file__).parent
    config_file = current_dir/ 'config' / 'deepspeed_zero3.yaml'
    train_path = current_dir/ 'src' / 'main.py'


    login(token=HF_TOKEN)
    wandb.login(key=WANDB_APIKEY)
    wandb.init(
        project=WANDB_PROJECT,
        )

    print(f"Using {torch.cuda.device_count()} {torch.cuda.get_device_name()} GPU(s).")

    # Construct the accelerate launch command with the specified arguments
    ALLOW_WANDB = os.environ.get("ALLOW_WANDB", "true").lower() == "true"
    

    # Parse configuration
    DATASET_NAME = "aidystark/usem_ocr"
    MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct" 
    #####################################################
    output_dir=str(OUTPUTS_PATH / "usem_ocr")



    # Launch training
    print("Spawning container for training.")

    # Full training arguments
    train_command = [
        "accelerate", "launch",
        str(config_file),
        str(train_path),
        "--dataset_name", str(DATASET_NAME),
        "--model_name_or_path", str(MODEL_NAME),
        "--per_device_train_batch_size", "8",
        "--gradient_accumulation_steps", "8",
        "--output_dir", output_dir,
        "--bf16",
        "--torch_dtype", "bfloat16",
        "--gradient_checkpointing",
        f"{'--report_to wandb' if ALLOW_WANDB else ''}"
    ]


    model.commit()
    print("✅ done")

    # Run the command in a subprocess
    try:
        result = subprocess.run(
            train_command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print('Output:', result.stdout)
    except subprocess.CalledProcessError as e:
        print('Command failed. Return code:', e.returncode)
        print('Output:', e.stdout)
        print('Error:', e.stderr)

    print(f"Training completed for experiment: {experiment}")



# Local entrypoint to start the training job
@app.local_entrypoint()
def main(experiment: str = None):
    if experiment is None:
        experiment = uuid4().hex[:8]
    
    print(f"🚀 Starting OCR VLM training experiment: {experiment}")
    train.remote(experiment)