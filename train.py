import os
from datetime import datetime
from pathlib import Path
import secrets
import yaml
from modal import App, Volume, Image, gpu, Retries
from .common import (
    app,
    ocr_vlm,
    HOURS,
    MINUTES,
    MODEL_PATH,
    OUTPUT_PATH,
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

def train():

    import torch
    import wandb
    import subprocess
    


    current_dir = Path(__file__).parent
    config_file = current_dir/ 'config' / 'deepspeed_zero3.yaml'
    train_path = current_dir/ 'src' / 'main.py'


    login(token=HF_TOKEN)
    wandb.login(key=WANDB_APIKEY)
    wandb.init(
        project=WANDB_PROJECT,
        config=training_args,
        )

    print(f"Using {torch.cuda.device_count()} {torch.cuda.get_device_name()} GPU(s).")

    # Construct the accelerate launch command with the specified arguments
    ALLOW_WANDB = os.environ.get("ALLOW_WANDB", "true").lower() == "true"
    

    # Parse configuration
    MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct" 
    #####################################################
    output_dir=str(checkpoint / "usem_ocr"),

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
        "--output_dir", str(output_dir),
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
















 