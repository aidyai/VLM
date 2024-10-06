from datasets import load_dataset
from PIL import Image
import random

description_list = [
  "Create fashion tags based on this image.",
  "Identify fashion-related tags from the image.",
  "Tag the fashion items visible in this image.",
  "Generate clothing and accessory tags from this picture.",
  "Provide fashion tags for the items in the image.",
  "Label the fashion elements seen in this image.",
  "Extract fashion tags based on the image content.",
]

# Function to format the data, since the image is already a PIL.Image object
def format_data(sample):

    # Use the image directly from the dataset
    pil_image = sample['image_path']

    # Create 'messages' structure
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an helpful assistant who describes fashion images accurately."}]},
        {"role": "user", "content": [
            {"type": "text", "text": random.choice(description_list)},
            {"type": "image", "text": pil_image}  # Use the existing PIL.Image object
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": sample['caption']}]}
    ]

    return {
        "messages": messages
    }