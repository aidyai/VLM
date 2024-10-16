import json
import os
import random
from datasets import load_dataset
from PIL import Image

# 10 random descriptors
descriptors = [
    "Identify the key fashion elements in this image.",
    "What are the primary clothing items shown in the image?",
    "Describe the outfit and accessories present in the image.",
    "List the fashion items and styles visible in the image.",
    "What clothing pieces are featured in this image?",
    "Describe the textures and fabrics visible in the outfit.",
    "What types of garments can you see in the image?",
    "List the different fashion elements in this outfit.",
    "What are the main fashion trends visible in the image?",
    "Identify the different clothing and accessory items shown."
]

# Function to save images from the dataset to a folder
def save_image(image, image_name, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Create the folder if it doesn't exist
    image_path = os.path.join(folder_path, image_name)
    image.save(image_path)
    return image_path

# Function to format dataset into JSON with random descriptors
def format2json(dataset_name, output_json_file, image_folder):
    # Load the Hugging Face dataset
    dataset = load_dataset(dataset_name)

    # Create the list that will store the formatted data
    formatted_data = []

    # Iterate over the dataset rows and convert to the desired format
    for idx, row in enumerate(dataset['train']):  # Adjust the split if necessary ('train', 'test', etc.)
        image = row['image_path']  # Image from the dataset
        image_name = f"image_{idx}.jpg"  # You can adjust the naming convention
        image_path = save_image(image, image_name, image_folder)  # Save image to the folder

        tags = row['caption']  # List of tags (e.g., ["T-shirt", "Jeans", "Sneakers"])

        # Randomly select one descriptor from the list
        random_descriptor = random.choice(descriptors)

        entry = {
            "image": image_path,  # Image path
            "conversations": [
                {
                    "from": "human",
                    "value": random_descriptor  # Use a random descriptor
                },
                {
                    "from": "gpt",
                    "value": tags  # Keep tags as a list of strings
                }
            ]
        }

        # Add the formatted entry to the list
        formatted_data.append(entry)

    # Save the formatted data to a JSON file
    with open(output_json_file, "w") as f:
        json.dump(formatted_data, f, indent=4)

    print(f"Formatted dataset saved to {output_json_file}")