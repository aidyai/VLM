# we will be executing our code using local environment
# Although this can work with Modal _Volumes_.

from pathlib import Path
import modal
from model import finetune_vlm



volume = modal.Volume.from_name("example-long-training", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "lightning~=2.4.0", "torch~=2.4.0", "torchvision==0.19.0"
)
app = modal.App("example-long-training-lightning", image=image)
volume_path = Path("/experiments")
DATA_PATH = volume_path / "data"
CHECKPOINTS_PATH = volume_path / "checkpoints"


volumes = {volume_path: volume}


def train(experiment):
    experiment_dir = CHECKPOINTS_PATH / experiment

    print("⚡️ starting training from scratch")  # we can still create options for resuming from checkpoint
    train_model(DATA_PATH, experiment_dir)

retries = modal.Retries(initial_delay=0.0, max_retries=10)
timeout = 30  # seconds

@app.function(volumes=volumes, gpu="a10g", timeout=timeout, retries=retries)
def train_interruptible(*args, **kwargs):
    train(*args, **kwargs)

@app.local_entrypoint()
def main(experiment: str = None):
    if experiment is None:
        from uuid import uuid4

        experiment = uuid4().hex[:8]
    print(f"⚡️ starting interruptible training experiment {experiment}")
    train_interruptible.remote(experiment)


# You can run this with
# ```bash
# modal run --detach 06_gpu_and_ml/long-training.py