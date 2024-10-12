import subprocess

# Set environment variables
os.environ['PYTHONPATH'] = 'src:' + os.environ.get('PYTHONPATH', '')

# Define model name
MODEL_NAME= "allenai/Molmo-7B-D-0924"
DATA_PATH= ""
IMAGE_FOLDER= ""
OUTPUT_DIR= ""

# Prepare the command to run the training script
train_command = [
    "torchrun", 
    "--nproc_per_node=1",
    "src/training/train.py",
    "--lora_enable", "True",
    "--lora_rank", "64",
    "--lora_alpha", "128",
    "--lora_dropout", "0.05",
    "--num_lora_modules", "10",
    "--deepspeed", "scripts/zero3_offload.json",
    "--model_id", MODEL_NAME,
    "--data_path", DATA_PATH,
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
    "--learning_rate", "1e-4",
    "--projector_lr", "1e-5",
    "--vision_lr", "2e-6",
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