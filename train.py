# from dataclasses import dataclass
# from pathlib import Path
# import os
# import secrets
# import subprocess
# from uuid import uuid4

# import modal
# import torch
# import wandb
# from huggingface_hub import login
# from VLM.config.common import (
#     app,
#     ocr_vlm,
#     HOURS,
#     MINUTES,
#     model,
#     outputs,
#     MODEL_PATH,
#     OUTPUTS_PATH,
#     MODEL_NAME,
# )



# # GPU Configuration
# GPU_CONFIG = os.environ.get("GPU_CONFIG", "a100:4")
# if len(GPU_CONFIG.split(":")) <= 1:
#     N_GPUS = int(os.environ.get("N_GPUS", 2))
#     GPU_CONFIG = f"{GPU_CONFIG}:{N_GPUS}"



# @dataclass
# class SharedConfig:
#     """Configuration information shared across project components."""
    
#     # Project-specific configuration
#     project_name: str = "usem_ocr"
#     dataset_name: str = "aidystark/usem_ocr"
#     model_name = MODEL_NAME

# @dataclass
# class TrainConfig(SharedConfig):
#     """Configuration for the training step."""
    
#     # Training hyperparameters
#     per_device_train_batch_size: int = 8
#     gradient_accumulation_steps: int = 8
#     learning_rate: float = 5e-5
#     max_train_steps: int = 1000
#     bf16: bool = False
#     fp16: bool = True
#     gradient_checkpointing: bool = True
    
#     # Experiment tracking
#     #report_to: str = "wandb"
#     wandb_project: str = "usem_ocr"

# os.environ['HF_TOKEN'] = "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"
# os.environ['WANDB_API_KEY'] = "0d505324ba165d96687f3624d4310bf171485b9d"
# os.environ['ALLOW_WANDB'] = 'true'  # Enable WandB logging




# def load_secrets():
#     """
#     Securely load secrets from environment or Modal secrets.
#     Raises an error if secrets are not properly configured.
#     """
#     try:
#         hf_token = os.environ.get("HF_TOKEN")
#         wandb_api_key = os.environ.get("WANDB_API_KEY")
        
#         if not hf_token or not wandb_api_key:
#             raise ValueError("Missing Hugging Face or WandB credentials")
        
#         return hf_token, wandb_api_key
#     except Exception as e:
#         raise RuntimeError(f"Secret loading failed: {e}")


# @app.function(
#     image=ocr_vlm,
#     gpu=GPU_CONFIG,
#     volumes={
#         OUTPUTS_PATH: model,  
#         MODEL_PATH: outputs,
#     },
#     timeout=4 * 60 * 60,  # 4 hours
#     # secrets=[huggingface_secret, wandb_secret]
# )

# def train(config: TrainConfig, experiment_id: str = None):
#     """
#     Perform model training with comprehensive configuration and logging.
    
#     Args:
#         config (TrainConfig): Configuration for the training run
#         experiment_id (str, optional): Unique identifier for the experiment
#     """
#     # Generate experiment ID if not provided
#     if experiment_id is None:
#         experiment_id = uuid4().hex[:8]
    
#     # Load secrets
#     hf_token, wandb_api_key = load_secrets()
    
#     # Login to Hugging Face and WandB
#     login(token=hf_token)
#     wandb.login(key=wandb_api_key)
    
#     # Initialize WandB tracking
#     wandb.init(
#         project=config.wandb_project,
#         name=f"ocr-training-{experiment_id}"
#     )

#     current_dir = Path(__file__).parent
#     config_file = current_dir/ 'VLM/config' / 'deepspeed_zero3.yaml'
#     train_path = current_dir/ 'VLM/src' / 'main.py'
#     output_dir=str(OUTPUTS_PATH / "usem_ocr")

    
#     # Construct training command
#     train_command = [
#         "accelerate", "launch",
#         "--config_file", str(config_file),
#         str(train_path),
#         "--dataset_name", config.dataset_name,
#         "--model_name_or_path", config.model_name,
#         "--per_device_train_batch_size", str(config.per_device_train_batch_size),
#         "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
#         "--gradient_checkpointing", str(config.gradient_checkpointing),
#         "--output_dir", output_dir,
#         "--bf16", str(config.bf16),
#         "--fp16", str(config.fp16),
#         # "report_to", config.report_to,
#     ]
    

#     # Execute training
#     try:
#         result = subprocess.run(
#             train_command,
#             check=True,
#             text=True,
#             capture_output=True
#         )
#         print(f"Training completed successfully for experiment: {experiment_id}")
#         print(result.stdout)
#     except subprocess.CalledProcessError as e:
#         print(f"Training failed for experiment: {experiment_id}")
#         print(f"Error output: {e.stderr}")
#         raise
#     finally:
#         # Commit volume changes and finish WandB run
#         model.commit()
#         wandb.finish()


# @app.local_entrypoint()
# def main(experiment_id: str = None):
#     """
#     Local entrypoint to initiate the training process.
    
#     Args:
#         experiment (str, optional): Custom experiment identifier
#     """
#     # Use provided experiment ID or generate a new one
#     experiment = experiment_id or uuid4().hex[:8]
    
#     print(f"🚀 Starting OCR VLM training experiment: {experiment_id}")
    
#     # Create configuration
#     config = TrainConfig()
    
#     # Launch remote training
#     train.remote(config, experiment)
    
#     print("✅ Training job queued successfully")


# if __name__ == "__main__":
#     main()
    



# # import os
# # from datetime import datetime
# # from pathlib import Path
# # import secrets
# # import yaml
# # from uuid import uuid4
# # from modal import App, Volume, Image, gpu, Retries
# # from VLM.config.common import (
# #     app,
# #     ocr_vlm,
# #     HOURS,
# #     MINUTES,
# #     model,
# #     outputs,
# #     MODEL_PATH,
# #     OUTPUTS_PATH,
# #     MODEL_NAME,
# # )


# # # GPU Configuration
# # GPU_CONFIG = os.environ.get("GPU_CONFIG", "a100:4")
# # if len(GPU_CONFIG.split(":")) <= 1:
# #     N_GPUS = int(os.environ.get("N_GPUS", 2))
# #     GPU_CONFIG = f"{GPU_CONFIG}:{N_GPUS}"


# # @app.function(
# #     image=ocr_vlm,
# #     gpu=GPU_CONFIG,
# #     volumes={
# #         OUTPUTS_PATH: model,  
# #         MODEL_PATH: outputs,
# #     },
# #     timeout=24 * HOURS,
# # )

# # def train(experiment=None):
# #     # Generate a unique experiment name if not provided
# #     if experiment is None:
# #         experiment = uuid4().hex[:8]
    
    
# #     import torch
# #     from huggingface_hub import login
# #     import wandb
# #     import subprocess
    
# #     HF_TOKEN = "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"
# #     WANDB_APIKEY = "0d505324ba165d96687f3624d4310bf171485b9d"
# #     WANDB_PROJECT = "usem_ocr"

# #     # Ensure sensitive data is set
# #     if not HF_TOKEN or not WANDB_APIKEY:
# #         raise ValueError("Please set Hugging Face and WandB tokens in your environment.")

# #     current_dir = Path(__file__).parent
# #     config_file = current_dir/ 'config' / 'deepspeed_zero3.yaml'
# #     train_path = current_dir/ 'src' / 'main.py'


# #     login(token=HF_TOKEN)
# #     wandb.login(key=WANDB_APIKEY)
# #     wandb.init(
# #         project=WANDB_PROJECT,
# #         )

# #     print(f"Using {torch.cuda.device_count()} {torch.cuda.get_device_name()} GPU(s).")

# #     # Construct the accelerate launch command with the specified arguments
# #     ALLOW_WANDB = os.environ.get("ALLOW_WANDB", "true").lower() == "true"
    

# #     # Parse configuration
# #     DATASET_NAME = "aidystark/usem_ocr"
# #     #####################################################
# #     output_dir=str(OUTPUTS_PATH / "usem_ocr")



# #     # Launch training
# #     print("Spawning container for training.")

# #     # Full training arguments
# #     train_command = [
# #         "accelerate", "launch",
# #         "--config_file", str(config_file),
# #         str(train_path),
# #         "--dataset_name", str(DATASET_NAME),
# #         "--model_name_or_path", str(MODEL_NAME),
# #         "--per_device_train_batch_size", "8",
# #         "--gradient_accumulation_steps", "8",
# #         "--output_dir", output_dir,
# #         "--bf16",
# #         "--torch_dtype", "bfloat16",
# #         "--gradient_checkpointing",
# #         f"{'--report_to wandb' if ALLOW_WANDB else ''}"
# #     ]


# #     model.commit()
# #     print("✅ done")

# #     # Run the command in a subprocess
# #     try:
# #         result = subprocess.run(
# #             train_command,
# #             check=True,
# #             text=True,
# #             stdout=subprocess.PIPE,
# #             stderr=subprocess.PIPE
# #         )
# #         print('Output:', result.stdout)
# #     except subprocess.CalledProcessError as e:
# #         print('Command failed. Return code:', e.returncode)
# #         print('Output:', e.stdout)
# #         print('Error:', e.stderr)

# #     print(f"Training completed for experiment: {experiment}")



# # # Local entrypoint to start the training job
# # @app.local_entrypoint()
# # def main(experiment: str = None):
# #     if experiment is None:
# #         experiment = uuid4().hex[:8]
    
# #     print(f"🚀 Starting OCR VLM training experiment: {experiment}")
# #     train.remote(experiment)





# from dataclasses import dataclass
# from pathlib import Path
# import os
# import secrets
# import subprocess
# from uuid import uuid4

# import modal
# import torch
# import wandb
# from huggingface_hub import login


# @dataclass
# class SharedConfig:
#     """Configuration information shared across project components."""
    
#     # Project-specific configuration
#     project_name: str = "usem_ocr"
#     dataset_name: str = "aidystark/usem_ocr"
#     model_name: str = "microsoft/layoutlmv3-base"  # example model, adjust as needed


# @dataclass
# class TrainConfig(SharedConfig):
#     """Configuration for the training step."""
    
#     # Training hyperparameters
#     per_device_train_batch_size: int = 8
#     gradient_accumulation_steps: int = 8
#     learning_rate: float = 5e-5
#     max_train_steps: int = 1000
#     bf16: bool = True
#     gradient_checkpointing: bool = True
    
#     # Paths and directories
#     output_dir: str = "/outputs/usem_ocr"
#     config_file: str = str(Path(__file__).parent / 'config' / 'deepspeed_zero3.yaml')
#     train_script: str = str(Path(__file__).parent / 'src' / 'main.py')
    
#     # Experiment tracking
#     use_wandb: bool = True
#     wandb_project: str = "usem_ocr"


# # Volume and image configuration
# volume = modal.Volume.from_name("ocr-training-volume", create_if_missing=True)

# # Create a Modal image with necessary dependencies
# image = modal.Image.debian_slim().pip_install(
#     "torch",
#     "transformers",
#     "datasets",
#     "accelerate",
#     "wandb",
#     "huggingface_hub"
# )

# # Secrets management
# huggingface_secret = modal.Secret.from_name(
#     "huggingface-secret", 
#     required_keys=["HF_TOKEN"]
# )

# wandb_secret = modal.Secret.from_name(
#     "wandb-secret", 
#     required_keys=["WANDB_API_KEY"]
# )


# def load_secrets():
#     """
#     Securely load secrets from environment or Modal secrets.
#     Raises an error if secrets are not properly configured.
#     """
#     try:
#         hf_token = os.environ.get("HF_TOKEN")
#         wandb_api_key = os.environ.get("WANDB_API_KEY")
        
#         if not hf_token or not wandb_api_key:
#             raise ValueError("Missing Hugging Face or WandB credentials")
        
#         return hf_token, wandb_api_key
#     except Exception as e:
#         raise RuntimeError(f"Secret loading failed: {e}")


# @modal.App.function(
#     image=image,
#     gpu=modal.gpu.A100(count=4),  # Configurable GPU setup
#     volumes={"/outputs": volume},
#     timeout=24 * 60 * 60,  # 24 hours
#     secrets=[huggingface_secret, wandb_secret]
# )
# def train(config: TrainConfig, experiment_id: str = None):
#     """
#     Perform model training with comprehensive configuration and logging.
    
#     Args:
#         config (TrainConfig): Configuration for the training run
#         experiment_id (str, optional): Unique identifier for the experiment
#     """
#     # Generate experiment ID if not provided
#     if experiment_id is None:
#         experiment_id = uuid4().hex[:8]
    
#     # Load secrets
#     hf_token, wandb_api_key = load_secrets()
    
#     # Login to Hugging Face and WandB
#     login(token=hf_token)
#     wandb.login(key=wandb_api_key)
    
#     # Initialize WandB tracking
#     wandb.init(
#         project=config.wandb_project,
#         name=f"ocr-training-{experiment_id}"
#     )
    
#     # Construct training command
#     train_command = [
#         "accelerate", "launch",
#         "--config_file", config.config_file,
#         config.train_script,
#         "--dataset_name", config.dataset_name,
#         "--model_name_or_path", config.model_name,
#         "--per_device_train_batch_size", str(config.per_device_train_batch_size),
#         "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
#         "--output_dir", config.output_dir,
#     ]
    
#     # Conditionally add optional flags
#     if config.bf16:
#         train_command.extend(["--bf16", "--torch_dtype", "bfloat16"])
    
#     if config.gradient_checkpointing:
#         train_command.append("--gradient_checkpointing")
    
#     if config.use_wandb:
#         train_command.append("--report_to wandb")
    
#     # Execute training
#     try:
#         result = subprocess.run(
#             train_command,
#             check=True,
#             text=True,
#             capture_output=True
#         )
#         print(f"Training completed successfully for experiment: {experiment_id}")
#         print(result.stdout)
#     except subprocess.CalledProcessError as e:
#         print(f"Training failed for experiment: {experiment_id}")
#         print(f"Error output: {e.stderr}")
#         raise
#     finally:
#         # Commit volume changes and finish WandB run
#         volume.commit()
#         wandb.finish()


# @modal.App.local_entrypoint()
# def main(experiment: str = None):
#     """
#     Local entrypoint to initiate the training process.
    
#     Args:
#         experiment (str, optional): Custom experiment identifier
#     """
#     # Use provided experiment ID or generate a new one
#     experiment_id = experiment or uuid4().hex[:8]
    
#     print(f"🚀 Starting OCR VLM training experiment: {experiment_id}")
    
#     # Create configuration
#     config = TrainConfig()
    
#     # Launch remote training
#     train.remote(config, experiment_id)
    
#     print("✅ Training job queued successfully")


# if __name__ == "__main__":
#     main()







# # from dataclasses import dataclass
# # from pathlib import Path
# # import os
# # import secrets
# # import subprocess
# # from uuid import uuid4

# # import modal
# # import torch
# # import wandb
# # from huggingface_hub import login
# # from VLM.config.common import (
# #     app,
# #     ocr_vlm,
# #     HOURS,
# #     MINUTES,
# #     model,
# #     outputs,
# #     MODEL_PATH,
# #     OUTPUTS_PATH,
# #     MODEL_NAME,
# # )



# # # GPU Configuration
# # GPU_CONFIG = os.environ.get("GPU_CONFIG", "a100:4")
# # if len(GPU_CONFIG.split(":")) <= 1:
# #     N_GPUS = int(os.environ.get("N_GPUS", 2))
# #     GPU_CONFIG = f"{GPU_CONFIG}:{N_GPUS}"



# # @dataclass
# # class SharedConfig:
# #     """Configuration information shared across project components."""
    
# #     # Project-specific configuration
# #     project_name: str = "usem_ocr"
# #     dataset_name: str = "aidystark/usem_ocr"
# #     model_name = MODEL_NAME

# # @dataclass
# # class TrainConfig(SharedConfig):
# #     """Configuration for the training step."""
    
# #     # Training hyperparameters
# #     per_device_train_batch_size: int = 8
# #     gradient_accumulation_steps: int = 8
# #     learning_rate: float = 5e-5
# #     max_train_steps: int = 1000
# #     bf16: bool = True
# #     gradient_checkpointing: bool = True
    
# #     # Paths and directories
# #     # output_dir: str = "/outputs/usem_ocr"
# #     # config_file: str = str(Path(__file__).parent / 'config' / 'deepspeed_zero3.yaml')
# #     # train_script: str = str(Path(__file__).parent / 'src' / 'main.py')
    
# #     # Experiment tracking
# #     use_wandb: bool = True
# #     wandb_project: str = "usem_ocr"

# # import os

# # # Set Hugging Face Token
# # os.environ['HF_TOKEN'] = "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"

# # # Set Weights & Biases API Key
# # os.environ['WANDB_API_KEY'] = "0d505324ba165d96687f3624d4310bf171485b9d"

# # # Optional: Add more configuration environment variables
# # os.environ['ALLOW_WANDB'] = 'true'  # Enable WandB logging


# # # huggingface_secret = modal.Secret.from_name(
# # #     "huggingface-secret", 
# # #     required_keys=["HF_TOKEN"]
# # # )

# # # wandb_secret = modal.Secret.from_name(
# # #     "wandb-secret", 
# # #     required_keys=["WANDB_API_KEY"]
# # # )

# # def load_secrets():
# #     """
# #     Securely load secrets from environment or Modal secrets.
# #     Raises an error if secrets are not properly configured.
# #     """
# #     try:
# #         hf_token = os.environ.get("HF_TOKEN")
# #         wandb_api_key = os.environ.get("WANDB_API_KEY")
        
# #         if not hf_token or not wandb_api_key:
# #             raise ValueError("Missing Hugging Face or WandB credentials")
        
# #         return hf_token, wandb_api_key
# #     except Exception as e:
# #         raise RuntimeError(f"Secret loading failed: {e}")


# # @app.function(
# #     image=ocr_vlm,
# #     gpu=GPU_CONFIG,
# #     volumes={
# #         OUTPUTS_PATH: model,  
# #         MODEL_PATH: outputs,
# #     },
# #     timeout=4 * 60 * 60,  # 24 hours
# #     # secrets=[huggingface_secret, wandb_secret]
# # )

# # def train(config: TrainConfig, experiment_id: str = None):
# #     """
# #     Perform model training with comprehensive configuration and logging.
    
# #     Args:
# #         config (TrainConfig): Configuration for the training run
# #         experiment_id (str, optional): Unique identifier for the experiment
# #     """
# #     # Generate experiment ID if not provided
# #     if experiment_id is None:
# #         experiment_id = uuid4().hex[:8]
    
# #     # Load secrets
# #     hf_token, wandb_api_key = load_secrets()
    
# #     # Login to Hugging Face and WandB
# #     login(token=hf_token)
# #     wandb.login(key=wandb_api_key)
    
# #     # Initialize WandB tracking
# #     wandb.init(
# #         project=config.wandb_project,
# #         name=f"ocr-training-{experiment_id}"
# #     )

# #     current_dir = Path(__file__).parent
# #     config_file = current_dir/ 'config' / 'deepspeed_zero3.yaml'
# #     train_path = current_dir/ 'src' / 'main.py'
# #     output_dir=str(OUTPUTS_PATH / "usem_ocr")

    
# #     # Construct training command
# #     train_command = [
# #         "accelerate", "launch",
# #         "--config_file", str(config_file),
# #         str(train_path),
# #         "--dataset_name", config.dataset_name,
# #         "--model_name_or_path", config.model_name,
# #         "--per_device_train_batch_size", str(config.per_device_train_batch_size),
# #         "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
# #         "--output_dir", output_dir,
# #     ]
    
# #     # Conditionally add optional flags
# #     if config.bf16:
# #         train_command.extend(["--bf16", "--torch_dtype", "bfloat16"])
    
# #     if config.gradient_checkpointing:
# #         train_command.append("--gradient_checkpointing")
    
# #     if config.use_wandb:
# #         train_command.append("--report_to wandb")
    
# #     # Execute training
# #     try:
# #         result = subprocess.run(
# #             train_command,
# #             check=True,
# #             text=True,
# #             capture_output=True
# #         )
# #         print(f"Training completed successfully for experiment: {experiment_id}")
# #         print(result.stdout)
# #     except subprocess.CalledProcessError as e:
# #         print(f"Training failed for experiment: {experiment_id}")
# #         print(f"Error output: {e.stderr}")
# #         raise
# #     finally:
# #         # Commit volume changes and finish WandB run
# #         model.commit()
# #         wandb.finish()


# # @app.local_entrypoint()
# # def main(experiment_id: str = None):
# #     """
# #     Local entrypoint to initiate the training process.
    
# #     Args:
# #         experiment (str, optional): Custom experiment identifier
# #     """
# #     # Use provided experiment ID or generate a new one
# #     experiment = experiment_id or uuid4().hex[:8]
    
# #     print(f"🚀 Starting OCR VLM training experiment: {experiment_id}")
    
# #     # Create configuration
# #     config = TrainConfig()
    
# #     # Launch remote training
# #     train.remote(config, experiment)
    
# #     print("✅ Training job queued successfully")


# # if __name__ == "__main__":
# #     main()
    



from dataclasses import dataclass
from pathlib import Path
import os
import secrets
import subprocess
from uuid import uuid4

import modal
import torch
import wandb
from huggingface_hub import login
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
)



# GPU Configuration
GPU_CONFIG = os.environ.get("GPU_CONFIG", "H100:8")
if len(GPU_CONFIG.split(":")) <= 1:
    N_GPUS = int(os.environ.get("N_GPUS", 2))
    GPU_CONFIG = f"{GPU_CONFIG}:{N_GPUS}"



@dataclass
class SharedConfig:
    """Configuration information shared across project components."""
    
    # Project-specific configuration
    project_name: str = "usem_ocr"
    dataset_name: str = "aidystark/usem_ocr"
    model_name = MODEL_NAME

@dataclass
class TrainConfig(SharedConfig):
    """Configuration for the training step."""
    
    # Training hyperparameters
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    max_train_steps: int = 1000
    bf16: bool = True
    tf32: bool = True
    torch_dtype: str = "bfloat16"
    fp16: bool =False
    gradient_checkpointing: bool = True
    
    # Experiment tracking
    #report_to: str = "wandb"
    wandb_project: str = "usem_ocr"

os.environ['HF_TOKEN'] = "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"
os.environ['WANDB_API_KEY'] = "0d505324ba165d96687f3624d4310bf171485b9d"
os.environ['ALLOW_WANDB'] = 'true'  # Enable WandB logging




def load_secrets():
    """
    Securely load secrets from environment or Modal secrets.
    Raises an error if secrets are not properly configured.
    """
    try:
        hf_token = os.environ.get("HF_TOKEN")
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        
        if not hf_token or not wandb_api_key:
            raise ValueError("Missing Hugging Face or WandB credentials")
        
        return hf_token, wandb_api_key
    except Exception as e:
        raise RuntimeError(f"Secret loading failed: {e}")


@app.function(
    image=ocr_vlm,
    gpu=GPU_CONFIG,
    volumes={
        OUTPUTS_PATH: model,  
        MODEL_PATH: outputs,
    },
    timeout=4 * 60 * 60,  # 4 hours
    # secrets=[huggingface_secret, wandb_secret]
)

def train(config: TrainConfig, experiment_id: str = None):
    """
    Perform model training with comprehensive configuration and logging.
    
    Args:
        config (TrainConfig): Configuration for the training run
        experiment_id (str, optional): Unique identifier for the experiment
    """
    # Generate experiment ID if not provided
    if experiment_id is None:
        experiment_id = uuid4().hex[:8]
    
    # Load secrets
    hf_token, wandb_api_key = load_secrets()
    
    # Login to Hugging Face and WandB
    login(token=hf_token)
    wandb.login(key=wandb_api_key)
    
    # Initialize WandB tracking
    wandb.init(
        project=config.wandb_project,
        name=f"ocr-training-{experiment_id}"
    )

    current_dir = Path(__file__).parent
    config_file = current_dir/ 'VLM/config' / 'deepspeed_zero3.yaml'
    train_path = current_dir/ 'VLM/src' / 'main.py'
    output_dir=str(OUTPUTS_PATH / "usem_ocr")


    print(f"Using {torch.cuda.device_count()} {torch.cuda.get_device_name()} GPU(s).")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


    # Construct training command
    train_command = [
        "accelerate", "launch",
        "--config_file", str(config_file),
        str(train_path),
        "--dataset_name", config.dataset_name,
        "--model_name_or_path", config.model_name,
        "--per_device_train_batch_size", str(config.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
        "--gradient_checkpointing", str(config.gradient_checkpointing),
        "--output_dir", output_dir,
        "--bf16", str(config.bf16),
        "--tf32", str(config.tf32),
        "--torch_dtype", str(config.torch_dtype),
        # "report_to", config.report_to,
    ]
    

    # Execute training
    try:
        result = subprocess.run(
            train_command,
            check=True,
            text=True,
            capture_output=True
        )
        print(f"Training completed successfully for experiment: {experiment_id}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Training failed for experiment: {experiment_id}")
        print(f"Error output: {e.stderr}")
        raise
    finally:
        # Commit volume changes and finish WandB run
        model.commit()
        wandb.finish()


@app.local_entrypoint()
def main(experiment_id: str = None):
    """
    Local entrypoint to initiate the training process.
    
    Args:
        experiment (str, optional): Custom experiment identifier
    """
    # Use provided experiment ID or generate a new one
    experiment = experiment_id or uuid4().hex[:8]
    
    print(f"🚀 Starting OCR VLM training experiment: {experiment_id}")
    
    # Create configuration
    config = TrainConfig()


    
    # Launch remote training
    train.remote(config, experiment)
    
    print("✅ Training job queued successfully")


if __name__ == "__main__":
    main()
    