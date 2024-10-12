import json
from datasets import load_dataset

def dataset2json(dataset_name, json_file):
    """
    Loads a Hugging Face dataset, format it, and save as a JSON file.

    Parameters:
    - dataset_name (str): The name of the dataset to load.
    - output_json_file (str): The path to save the formatted JSON file.
    
    Returns:
    - None: The function writes the formatted data to the specified JSON file.
    """

    # Load the Hugging Face dataset
    dataset = load_dataset(dataset_name)

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
    with open(json_file, "w") as f:
        json.dump(formatted_data, f, indent=4)

    print(f"Formatted dataset saved to {json_file}")