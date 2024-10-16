# we will be executing our code using local environment
# Although this can work with Modal _Volumes_.

import modal
from modal import Image

volume = modal.Volume.from_name("vlm-training", create_if_missing=True)

cuda_version = "12.1.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

vlm = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install(  # required to build flash-attn
        "ninja",
        "packaging",
        "wheel",
        "torch",
        "datasets==3.0.1",
        "accelerate==0.34.2",
        "evaluate==0.4.3",
        "bitsandbytes==0.44.0",
        "trl==0.11.1",
        "peft==0.13.0",
        "qwen-vl-utils",
        "python-dotenv",
        "torch~=2.4.0",
        "torchvision",
        "wandb",
        "deepspeed",
        "einops",
        "ujson",
        "decord",
    )
)


app = modal.App("vlm-training", image=vlm)
volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
CHECKPOINTS_PATH = "/vol/experiment"



retries = modal.Retries(initial_delay=0.0, max_retries=10)
timeout = 7200  # in seconds this is 2 hrs

@app.function(
    volumes={CHECKPOINTS_PATH: volume},
    gpu=modal.gpu.H100(count=2),
    timeout=timeout, 
    retries=retries
)
def train():
    import os
    import subprocess
    from pathlib import Path
    import sys
    import wandb
    from VLM.utils.data import format2json
    from dotenv import load_dotenv
    from huggingface_hub import login


    # Set up training parameters
    MODEL_NAME = "allenai/Molmo-7B-D-0924"
    DEEPSPEEDPATH = "VLM/utils/zero3.json"






    print("⚡️ Starting training...")

    # Construct the DeepSpeed command
    deepspeed_command = [
        "deepspeed",
        "--num_gpus=2",
        "VLM/src/training/train.py",
        "--lora_enable", "True",
        "--vision_lora", "True",
        "--lora_rank", "64",
        "--lora_alpha", "128",
        "--lora_dropout", "0.05",
        "--num_lora_modules", "-1",
        "--deepspeed", DEEPSPEEDPATH,
        "--model_id", MODEL_NAME,
        "--data_path", JSON_FILE,
        "--image_folder", IMAGE_FOLDER,
        "--freeze_vision_tower", "False",
        "--freeze_llm", "False",
        "--tune_projector", "True",
        "--bf16", "True",
        "--fp16", "False",
        "--disable_flash_attn2", "False",
        "--output_dir", OUTPUT_DIR,
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--learning_rate", "2e-4",
        "--weight_decay", "0.",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--tf32", "True",
        "--gradient_checkpointing", "False",
        "--report_to", "wandb",
        "--lazy_preprocess", "True",
        "--save_strategy", "steps",
        "--save_steps", "200",
        "--save_total_limit", "10",
        "--dataloader_num_workers", "4"
    ]

    # Set the PYTHONPATH to include the necessary directories
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{os.getcwd()}/VLM/src:" + env.get('PYTHONPATH', '')

    # Run the DeepSpeed command
    try:
        result = subprocess.run(
            deepspeed_command,
            env=env,
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

    print("Training completed.") 



base_path = Path("/local_dir")

DATASET_NAME = "aidystark/fashion-tag"
OUTPUT_DIR = base_path / "molmo"
JSON_FILE = base_path / "output.json"
IMAGE_FOLDER = base_path / "IMG"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)

HF_TOKEN = "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"
DATASET_ID = "aidystark/fashion-tag"
JSON_FILE = "output.json"
WANDB_APIKEY = "0d505324ba165d96687f3624d4310bf171485b9d"

login(token=HF_TOKEN)
wandb.login(key=WANDB_APIKEY)


# Ensure output directory exists
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# Call the function from data.py to process and format the dataset
format2json(DATASET_NAME, JSON_FILE, IMAGE_FOLDER)