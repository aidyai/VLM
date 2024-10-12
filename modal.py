# we will be executing our code using local environment
# Although this can work with Modal _Volumes_.
import modal
from modal import Image


volume = modal.Volume.from_name("vlm-training", create_if_missing=True)
vlm = (
        Image.debian_slim(python_version="3.12")
            .pip_install(
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
                "einops",
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

    from VLM.utils.data import dataset2json

    load_dotenv()
    HF_TOKEN = os.getenv("hf_token")
    DATASET_ID = os.getenv("dataset_id")
    WANDB_APIKEY = os.getenv("wandb_api_key")

    login(token=HF_TOKEN)
    wandb.login(key=WANDB_APIKEY)

    dataset2json(dataset_name, json_file)
    
    experiment_dir = CHECKPOINTS_PATH 

    print("⚡️ starting training.........")  # we can still create options for resuming from checkpoint

    import subprocess
    import sys
    subprocess.run(
        ["python", "./utils/run.py"],
        stdout=sys.stdout, stderr=sys.stderr,
        check=True,
    )





# def train_interruptible(*args, **kwargs):
#     train(*args, **kwargs)

# def main(experiment: str = None):
#     if experiment is None:
#         from uuid import uuid4

#         experiment = uuid4().hex[:8]
#     print(f"⚡️ starting interruptible training experiment {experiment}")
#     train_interruptible.remote(experiment)


# You can run this with
# ```bash
# modal run --detach 06_gpu_and_ml/long-training.py