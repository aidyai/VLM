# Vision-Language-Model Finetuning: MiniCPM-V 2.6	

This repository contains code to finetune the **MiniCPM-V 2.6	** Vision-Language Model from [OpenBMB](https://huggingface.co/openbmb/MiniCPM-V-2_6) for two specific tasks:

1. Optical Character Recognition (OCR) for a Low Resource Language that is Diacritics Sensitive

## Installation

Install the required packages either `requirements.txt` 

### Using requirements.txt

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```



## Dataset Preparation

### OCR Dataset

For the OCR task, the dataset should consist of images containing text in the target low-resource language and their corresponding transcriptions. Ensure that the dataset captures the diacritical marks accurately.

### Example Dataset Format

```json
[
    {
        "id": "663A5fFR4JBaSbchfQxsMU",
        "image": "663A5fFR4JBaSbchfQxsMU.jpg",
        "conversations": [
            {
                "role": "user",
                "content": "<image>\nPerform precise OCR on the image to extract Ibibio text, sentences, and words, while maintaining the integrity of all diacritical markers and orthographic nuances."
            },
            {
                "role": "assistant",
                "content": "ii. Utem Ayop\nKe esioho utom utọ inwañ efep.\n"
            }
        ]
    },
    {
        "id": "DXsPpRUhGshC53szkie8zr",
        "image": "DXsPpRUhGshC53szkie8zr.jpg",
        "conversations": [
            {
                "role": "user",
                "content": "<image>\nExtract all the written texts from the image from beginning to end, taking note of the Ibibio alphabet (Aa, Ʌʌ, Bb, Dd, Ee, Ff, Gg, Gh gh, Hh, Ii, Ịị, Kk, Kp kp, Mm, Nn, Ññ, Ñw ñw, Ọọ, Pp, Rr, Ss, Tt, Uu, Ww, Yy, Əə) from the images, ensuring full retention of diacritical markers and orthographic integrity."
            },
            {
                "role": "assistant",
                "content": "etƏkayịn amanake. Utọ etƏkayịn idọ ana .\n"
            }
        ]
    },
]
```

Ensure that the image paths in the dataset match the provided `--image_folder`.

## Training

To run the training script, use the following commands:


### Finetune with LoRA

To train the language model with LoRA and perform full training on the vision model:

```python
finetune.py
```

To train both the language model and the vision model with LoRA:


## Citation

This project is based on the work from the following repository:fine-tuning approach:
```
@misc{OpenBMB,
  author = {OpenBMB},
  title = {Finetuning MiniCPM-V-2_6 for Custom VLM Task},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/OpenBMB/MiniCPM-V/tree/main/finetune}}
}
```
