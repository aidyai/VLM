from peft import PeftModel
from transformers import AutoProcessor, AutoModelForVision2Seq
 
adapter_path = "./qwen2-7b-instruct-amazon-description"
base_model_id = "Qwen/Qwen2-VL-7B-Instruct"
merged_path = "merged"
 
# Load Model base model
model = AutoModelForVision2Seq.from_pretrained(model_id, low_cpu_mem_usage=True)
 
# Path to save the merged model
 
# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, adapter_path)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(merged_path,safe_serialization=True, max_shard_size="2GB")
 
processor = AutoProcessor.from_pretrained(base_model_id)
processor.save_pretrained(merged_path)