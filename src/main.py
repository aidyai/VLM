import os
import torch
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor
from .dataset import format_data
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)



def main():
    # Parse arguments
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # Modify training arguments
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Model configuration
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Load processor and model
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path, 
        trust_remote_code=model_config.trust_remote_code
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_config.model_name_or_path, 
        trust_remote_code=model_config.trust_remote_code, 
        **model_kwargs
    )

    # Load and format dataset
    # Use the dataset name from script arguments or hardcode if needed
    dataset_name = script_args.dataset_name 
    
    # Load dataset/format dataset
    dataset = load_dataset(dataset_name, split='train')
    formatted_dataset = [format_data(sample) for sample in dataset]

    # Data collator function
    def qwen_collate_fn(examples):
        # Implement your collate function logic
        # This is a placeholder - modify according to your specific requirements
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        image_inputs = [process_vision_info(example["messages"])[0] for example in examples]
    
        batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        # Add logic for handling image tokens
        batch["labels"] = labels
    
        return batch



    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        data_collator=qwen_collate_fn,
        dataset_text_field="",
        tokenizer=processor.tokenizer,
        peft_config=get_peft_config(model_config),
    )


    # Train
    print("🚀 Starting OCR VLM training experiment")

    trainer.train()
    trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)
    #     if trainer.accelerator.is_main_process:
    #         processor.push_to_hub(training_args.hub_model_id)

