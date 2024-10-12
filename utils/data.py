from datasets import load_dataset
import json
import os

# Load your Hugging Face dataset (adjust the dataset name and column names)
dataset = load_dataset("your_dataset_name")  # Replace with your dataset

# Define the output path where the formatted JSON will be saved
output_json_file = "formatted_dataset.json"

# Create the list that will store the formatted data
formatted_data = []

# Iterate over the dataset rows and convert to the desired format
for row in dataset['train']:  # Adjust the split if necessary ('train', 'test', etc.)
    image_path = row['image_path']  # Adjust the column name for your image paths
    tags = row['caption']  # Adjust the column name for your tags (["Fringe", "Dress", "Crochet"])

    entry = {
        "image": image_path,  # Image path or image file name
        "conversations": [
            {
                "from": "human",
                "value": "Describe the fashion items in the image."
            },
            {
                "from": "gpt",
                "value": json.dumps(tags)  # Convert the list of tags to a string
            }
        ]
    }

    # Add the formatted entry to the list
    formatted_data.append(entry)

# Save the formatted data to a JSON file
with open(output_json_file, "w") as f:
    json.dump(formatted_data, f, indent=4)

print(f"Formatted dataset saved to {output_json_file}")
