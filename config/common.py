import os
from pathlib import PurePosixPath
from typing import Union
import modal
from modal import App, Volume, Image, gpu, Retries
import string
import time
from pathlib import Path
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
        "ninja", "pillow", "packaging", "wheel", "torch", "accelerate",
        "datasets", "torchvision", "qwen_vl_utils", "bitsandbytes", "peft", "trl", 
        "huggingface_hub", "wandb", "deepspeed==0.15.4", 
        "transformers", "einops", "hf_transfer", "shortuuid",
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


MODEL_NAME = "Qwen/Qwen2-VL-2B" 
MODEL_REVISION = "d3a53f2484fce9d62fff115a5ddfc833f873bfde"

with ocr_vlm.imports():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForVision2Seq, AutoProcessor, Qwen2VLProcessor
    from qwen_vl_utils import process_vision_info



@app.function(
    image=ocr_vlm,
    volumes={
        MODEL_PATH: outputs,
    },
    timeout=20 * MINUTES,
)

def download_model(revision=MODEL_REVISION):
    # uses HF_HOME to point download to the model volume
    AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        revision=revision,
    )



# Data collator function

def qwen_collate_fn(examples, processor):
    # Process text inputs using the processor
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    # Process vision inputs from the messages
    image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

    # Create a batch of inputs using the processor
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    # Ensure input_ids are in the correct dtype (torch.long)
    batch["input_ids"] = batch["input_ids"].long()

    # Clone input_ids to prepare labels
    labels = batch["input_ids"].clone()

    # Mask padding tokens in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Ignore the image token index in the loss computation
    if isinstance(processor, Qwen2VLProcessor):
        image_tokens = [151652, 151653, 151655]
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    
    # Assign processed labels back to the batch
    batch["labels"] = labels

    return batch




# def qwen_collate_fn(examples, processor):
#     # Implement your collate function logic
#     texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
#     image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

#     batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

#     # The labels are the input_ids, and we mask the padding tokens in the loss computation
#     labels = batch["input_ids"].clone()
#     labels[labels == processor.tokenizer.pad_token_id] = -100  #
#     # Ignore the image token index in the loss computation (model specific)
#     if isinstance(processor, Qwen2VLProcessor):
#         image_tokens = [151652,151653,151655]
#     else: 
#         image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
#     for image_token_id in image_tokens:
#         labels[labels == image_token_id] = -100
#     batch["labels"] = labels
 
#     return batch







# # Ensure base model is downloaded
# try:
#     snapshot_download(
#         MODEL_NAME,
#         local_files=True,
#     )
#     print(f"Volume contains {MODEL_NAME}.")
# except FileNotFoundError:
#     print(f"Downloading {MODEL_NAME} ...")
#     snapshot_download(MODEL_NAME, local_dir=MODEL_PATH)
#     print("Committing /pretrained directory ...")
#   