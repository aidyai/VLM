import os
from pathlib import Path
from uuid import uuid4
from modal import App, Volume, Image, gpu, Retries
import subprocess



# Model and storage setup
cuda_version = "12.1.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Create volume for training
volume = Volume.from_name("ocrvlm-training", create_if_missing=True)

# Container image setup
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

# Create Modal app
app = App("ocr-vlm", image=ocr_vlm)

# Retry configuration
retries = Retries(initial_delay=0.0, max_retries=1)

# Long timeout (2 hours)
timeout = 7200  # 2 hrs

@app.function(
    volumes={"/vol/experiment": volume},
    gpu=gpu.A100(count=2),
    timeout=timeout, 
    retries=retries
)

# def train(experiment=None):
    # # Generate a unique experiment name if not provided
    # if experiment is None:
    #     experiment = uuid4().hex[:8]
    
    # # Output directory for this experiment
    # # output_dir = Path("/vol/experiment/output") / experiment
    # # output_dir.mkdir(parents=True, exist_ok=True)


def train():

    HF_TOKEN = "hf_fwLedemoMdFfpurzXxaArMIOGlboMxGUup"
    WANDB_APIKEY = "0d505324ba165d96687f3624d4310bf171485b9d"
    WANDB_PROJECT = "usem_ocr"

    # Ensure sensitive data is set
    if not HF_TOKEN or not WANDB_APIKEY:
        raise ValueError("Please set Hugging Face and WandB tokens in your environment.")

    # Authenticate Hugging Face and Weights & Biases
    from huggingface_hub import login
    import wandb
    login(token=HF_TOKEN)
    wandb.login(key=WANDB_APIKEY)
    wandb.init(project=WANDB_PROJECT)

    # Training parameters
    GPUS_PER_NODE = "2"
    NNODES = "1"
    NODE_RANK = "0"
    MASTER_ADDR = "localhost"
    MASTER_PORT = "6001"

    # Model and data paths
    MODEL = "openbmb/MiniCPM-V-2_6"
    LLM_TYPE = "qwen2"
    MODEL_MAX_LENGTH = "2048"
    
    current_dir = Path(__file__).parent
    
    TRAIN_DATA = current_dir / "train_ocr.json"
    EVAL_DATA = current_dir / "eval_ocr.json"
    finetune_path = current_dir / 'src' / 'train.py'
    deeppseed_path = current_dir / 'utils' / 'ds_config_zero2.json'

    output_dir = current_dir / "usem_ocr"
    output_dir.mkdir(parents=True, exist_ok=True)


    # Full training arguments
    finetune_args = [
        "torchrun",
        "--nproc_per_node", GPUS_PER_NODE,
        "--nnodes", NNODES,
        "--node_rank", NODE_RANK,
        "--master_addr", MASTER_ADDR,
        "--master_port", MASTER_PORT,
        str(finetune_path),
        "--model_name_or_path", MODEL,
        "--llm_type", LLM_TYPE,
        "--data_path", str(TRAIN_DATA),
        "--eval_data_path", str(EVAL_DATA),
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
        "--output_dir", str(output_dir),
        "--logging_dir", str(output_dir),
        "--logging_strategy", "steps",
        "--per_device_train_batch_size", "1",
        "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--evaluation_strategy", "steps",
        "--save_strategy", "steps",
        "--save_steps", "100",
        "--save_total_limit", "10",
        "--learning_rate", "1e-6",
        "--weight_decay", "0.1",
        "--adam_beta2", "0.95",
        "--warmup_ratio", "0.01",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--gradient_checkpointing", "true",
        "--deepspeed", str(deeppseed_path),
        "--report_to", "wandb"
    ]

    # Set the PYTHONPATH to include the necessary directories
    # env = os.environ.copy()
    # env['PYTHONPATH'] = f"{os.getcwd()}/VLM/src:" + env.get('PYTHONPATH', '')


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

    print(f"Training completed for experiment")
