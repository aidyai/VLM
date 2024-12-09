import os
import torch
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor
from VLM.src.dataset import format_data
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
from functools import partial
from transformers import Qwen2VLProcessor
from VLM.config.common import (
    app,
    ocr_vlm,
    HOURS,
    MINUTES,
    model,
    outputs,
    MODEL_PATH,
    OUTPUTS_PATH,
    MODEL_NAME,
    qwen_collate_fn,
)



if __name__ == "__main__":

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

   
    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        data_collator=partial(qwen_collate_fn, processor=processor),
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

