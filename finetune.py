import os
from pathlib import Path
from uuid import uuid4
from modal import App, Volume, Image, gpu, Retries


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


import wandb
from dotenv import load_dotenv
from trl import SFTConfig, SFTTrainer
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig
from huggingface_hub import login
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from datasets import load_dataset
from dataset import format_data




def train_model():
    HF_TOKEN = "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"
    WANDB_APIKEY = "0d505324ba165d96687f3624d4310bf171485b9d"
    WANDB_PROJECT = "usem_ocr"
    DATASET_ID = "aidystark/usem_ocr"
    
    # Ensure sensitive data is set
    if not HF_TOKEN or not WANDB_APIKEY:
        raise ValueError("Please set Hugging Face and WandB tokens in your environment.")

    # Authenticate Hugging Face and Weights & Biases
    from huggingface_hub import login
    import wandb
    login(token=HF_TOKEN)
    wandb.login(key=WANDB_APIKEY)

    dataset = load_dataset(DATASET_ID, split='train')
    formatted_dataset = [format_data(sample) for sample in dataset]

    current_dir = Path(__file__).parent
    output_dir = current_dir / "usem_ocr"
    output_dir.mkdir(parents=True, exist_ok=True)


    model_id = "Qwen/Qwen2-VL-7B-Instruct" 

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    # load the processor
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    # load the model
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        # attn_implementation="flash_attention_2", # not supported for training
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )


    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )

    args = SFTConfig(
        output_dir=output_dir,  # Now using the passed parameter for output directory
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        # gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=5,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=True,
        report_to="wandb",
        # gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True}
    )
    args.remove_unused_columns = False

    def collate_fn_molmo(examples):
        # Extract and process text and visual information from each example
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        image_inputs, _ = process_vision_info([example["messages"] for example in examples])
        
        # If image_inputs is None, you may want to handle it appropriately
        if image_inputs is None:
            image_inputs = []  # or handle as needed

        # Tokenize the texts and process the images using the processor
        batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

        # Create labels for the language model
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss calculation

        # Optionally, exclude image tokens from the loss calculation
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        labels[labels == image_token_id] = -100

        # Add the labels to the batch
        batch["labels"] = labels

        # Move tensors to the appropriate device
        batch = {k: v.to(torch.bfloat16 if bnb_config.bnb_4bit_compute_dtype == torch.bfloat16 else torch.float32) 
                for k, v in batch.items()}

        return batch

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=formatted_dataset,
        data_collator=collate_fn_molmo,
        dataset_text_field="",
        peft_config=peft_config,
        tokenizer=processor.tokenizer
    )

    trainer.train()
    trainer.save_model(output_dir)


