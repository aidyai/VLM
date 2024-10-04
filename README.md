# Vision-Language-Model Finetuning: Molmo-7B-D-0924

This repository contains code to finetune the **Molmo-7B-D-0924** Vision-Language Model from [Allen.ai](https://github.com/allenai) for two specific tasks:

1. Optical Character Recognition (OCR) for a Low Resource Language that is Diacritics Sensitive
2. Fashion Tag Generator

## Installation

Install the required packages using either `requirements.txt` or `environment.yml`.

### Using requirements.txt

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Using environment.yaml

```bash
conda env create -f environment.yaml
conda activate molmo-env
pip install flash-attn --no-build-isolation
```

**Note:** Install flash-attn after running other libraries from requirements.txt or environment.yaml.

## Dataset Preparation

### OCR Dataset

For the OCR task, the dataset should consist of images containing text in the target low-resource language and their corresponding transcriptions. Ensure that the dataset captures the diacritical marks accurately.

### Fashion Dataset

For the Fashion Tag Generator task, the dataset should include fashion images and their corresponding tags or descriptions.

Both datasets should follow the LLaVA JSON format. Each entry contains image paths and conversations describing the content.

### Example Dataset Format

```json
[
  {
    "id": "ocr_001",
    "image": "text_image_001.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat text is shown in this image?"
      },
      {
        "from": "gpt",
        "value": "The text in the image reads: 'Héllò Wórld' in the target language with diacritics."
      }
    ]
  },
  {
    "id": "fashion_001",
    "image": "fashion_item_001.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nDescribe this fashion item and provide relevant tags."
      },
      {
        "from": "gpt",
        "value": "This image shows a red floral summer dress. Tags: #summerdress #floralprint #redDress #casualWear"
      }
    ]
  }
]
```

Ensure that the image paths in the dataset match the provided `--image_folder`.

## Training

To run the training script, use the following commands:

### Full Finetuning

```bash
bash scripts/finetune.sh
```

### Finetune with LoRA

To train the language model with LoRA and perform full training on the vision model:

```bash
bash scripts/finetune_lora.sh
```

To train both the language model and the vision model with LoRA:

```bash
bash scripts/finetune_lora_vision.sh
```

## Configuration

The `config.json` file is where you define your training settings. Here's an example configuration:

```json
{
  "model": "allenai/molmo-7b-d-0924",
  "dataset_path": "./dataset/",
  "batch_size": 32,
  "learning_rate": 5e-5,
  "epochs": 3,
  "task": ["ocr", "fashion_tagging"]
}
```

Adjust the parameters as needed to fit your use case.

## Customizing the Model

If you want to modify the model architecture or integrate custom parameter-efficient fine-tuning techniques, check the `model_utils.py` file for instructions on adding custom layers or parameters.

## Citation

This project is based on the work from the following repository:fine-tuning approach:
```
@misc{philschmid2023,
  author = {Philipp Schmid},
  title = {Deep Learning PyTorch Hugging Face},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/fine-tune-multimodal-llms-with-trl.ipynb}}
}
```
