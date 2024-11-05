import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from modal import App, Volume, Image, gpu

load_dotenv()

# Model and storage setup
volume = Volume.from_name("ocrvlm-training", create_if_missing=True)
cuda_version = "12.1.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

ocr_vlm = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "ninja", "packaging", "wheel", "torch",
        "datasets==3.0.1", "accelerate==0.34.2", "evaluate==0.4.3",
        "bitsandbytes==0.44.0", "trl==0.11.1", "peft==0.13.0",
        "qwen-vl-utils", "python-dotenv", "torch~=2.4.0", 
        "torchvision", "wandb", "deepspeed", "einops", 
        "ujson", "decord",
    )
)

app = App("ocr-vlm", image=ocr_vlm)
volume = Volume.from_name("model-weights-vol", create_if_missing=True)
CHECKPOINTS_PATH = "/vol/experiment"

retries = modal.Retries(initial_delay=0.0, max_retries=10)
timeout = 7200  # 2 hrs

@app.function(
    volumes={CHECKPOINTS_PATH: volume},
    gpu=gpu.H100(count=2),
    timeout=timeout, 
    retries=retries
)
def train():
    # Load sensitive data from environment
    HF_TOKEN = os.getenv("HF_TOKEN")
    WANDB_APIKEY = os.getenv("WANDB_APIKEY")

    # Ensure sensitive data is set
    if not HF_TOKEN or not WANDB_APIKEY:
        raise ValueError("Please set Hugging Face and WandB tokens in your environment.")

    # Authenticate Hugging Face and Weights & Biases
    from huggingface_hub import login
    import wandb
    login(token=HF_TOKEN)
    wandb.login(key=WANDB_APIKEY)

    # Training parameters
    GPUS_PER_NODE = "8"
    NNODES = "1"
    NODE_RANK = "0"
    MASTER_ADDR = "localhost"
    MASTER_PORT = "6001"

    # Model and data paths
    MODEL = "openbmb/MiniCPM-V-2_6"
    DATA = "path/to/training_data"   # Replace with actual path
    LLM_TYPE = "qwen2"
    MODEL_MAX_LENGTH = "2048"
    OUTPUT_DIR = "minicpm/minicipm_lora",


    # Torchrun distributed arguments
    distributed_args = [
        "--nproc_per_node", GPUS_PER_NODE,
        "--nnodes", NNODES,
        "--node_rank", NODE_RANK,
        "--master_addr", MASTER_ADDR,
        "--master_port", MASTER_PORT,
    ]

    # Finetuning arguments
    finetune_args = [
        "torchrun",
        *distributed_args,
        "train.py",
        "--model_name_or_path", MODEL,
        "--llm_type", LLM_TYPE,
        "--data_path", DATA,
        "--remove_unused_columns", "false",
        "--label_names", "labels",
        "--prediction_loss_only", "false",
        "--bf16", "false",
        "--bf16_full_eval", "false",
        "--fp16", "true",
        "--fp16_full_eval", "true",
        "--do_train",
        "--do_eval",
        "--tune_vision", "true",
        "--tune_llm", "false",
        "--use_lora", "true",
        "--lora_target_modules", r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
        "--model_max_length", MODEL_MAX_LENGTH,
        "--max_slice_nums", "9",
        "--max_steps", "10000",
        "--eval_steps", "1000",
        "--output_dir", "output/output_lora",
        "--logging_dir", "output/output_lora",
        "--logging_strategy", "steps",
        "--per_device_train_batch_size", "1",
        "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--evaluation_strategy", "steps",
        "--save_strategy", "steps",
        "--save_steps", "1000",
        "--save_total_limit", "10",
        "--learning_rate", "1e-6",
        "--weight_decay", "0.1",
        "--adam_beta2", "0.95",
        "--warmup_ratio", "0.01",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--gradient_checkpointing", "true",
        "--deepspeed", "ds_config_zero2.json",
        "--report_to", "wandb"
    ]

    # Run the command in a subprocess
    try:
        result = subprocess.run(
            finetune_args,
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
