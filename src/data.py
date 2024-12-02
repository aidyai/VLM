import shortuuid
from datasets import load_dataset
from PIL import Image
import random
import json
import tqdm
import os

# Load the dataset from Huggingface
ds = load_dataset("aidystark/usem_ocr")

# Paths for saving processed data
auto_data_image_path = './image'  # Path where images will be saved
auto_data_data_path = './ocr.json'  # Path to save the JSON data

# Ensure the directory for saving images exists
os.makedirs(auto_data_image_path, exist_ok=True)

# Description list for OCR tasks
description_list = [
    "Extract all the written texts from the image from beginning to end, taking note of the Ibibio alphabet (Aa, Ʌʌ, Bb, Dd, Ee, Ff, Gg, Gh gh, Hh, Ii, Ịị, Kk, Kp kp, Mm, Nn, Ññ, Ñw ñw, Ọọ, Pp, Rr, Ss, Tt, Uu, Ww, Yy, Əə) from the images, ensuring full retention of diacritical markers and orthographic integrity.",
    "Undertake precise text extraction from images written in the Ibibio script, ensuring every alphabet character (Aa, Ʌʌ, Bb, Dd, Ee, Ff, Gg, Gh gh, Hh, Ii, Ịị, Kk, Kp kp, Mm, Nn, Ññ, Ñw ñw, Ọọ, Pp, Rr, Ss, Tt, Uu, Ww, Yy, Əə) and diacritical detail is preserved.",
    "Extract all text, sentences, and words from the image, ensuring accurate recognition of Ibibio characters (Aa, Ʌʌ, Bb, Dd, Ee, Ff, Gg, Gh gh, Hh, Ii, Ịị, Kk, Kp kp, Mm, Nn, Ññ, Ñw ñw, Ọọ, Pp, Rr, Ss, Tt, Uu, Ww, Yy, Əə) and preserving diacritical details.",
    "Perform precise OCR on the image to extract Ibibio text, sentences, and words, while maintaining the integrity of all diacritical markers and orthographic nuances.",
    "Extract sentences, words, and phrases written in the Ibibio language from the image, ensuring accurate recognition of its unique alphabet and diacritical features.",
    "Retrieve all text written in the Ibibio language from the image, capturing each character (Aa, Ʌʌ, Bb, Dd, Ee, Ff, Gg, Gh gh, Hh, Ii, Ịị, Kk, Kp kp, Mm, Nn, Ññ, Ñw ñw, Ọọ, Pp, Rr, Ss, Tt, Uu, Ww, Yy, Əə) and preserving orthographic details.",
    "Perform detailed text extraction on the image, focusing on sentences and words written in the Ibibio alphabet, with full attention to diacritical accuracy and linguistic fidelity.",
    "Extract all textual elements from the image which are written in the Ibibio language, ensuring precise recognition of the alphabet and correct diacritical representation."
]

# Process the dataset
ocr_data = []

for sample in tqdm.tqdm(ds['train'], desc="Processing Dataset"):
    uuid = shortuuid.uuid()
    sample_dict = {
        'id': uuid,
        'image': f"./image/{uuid}.jpg"
    }

    # Save the image
    image_path = os.path.join(auto_data_image_path, f"{uuid}.jpg")
    
    # If it's already a PIL Image, save directly
    # If it's a path or bytes, open it first
    if isinstance(sample['image'], Image.Image):
        sample['image'].save(image_path)
    else:
        Image.open(sample['image']).save(image_path)

    # Create the conversation with a random description and the provided text
    conversations = [
        {"role": "user", "content": f"<image>\n{random.choice(description_list)}"},
        {"role": "assistant", "content": sample['text']}
    ]
    sample_dict['conversations'] = conversations

    # Append the processed data
    ocr_data.append(sample_dict)

# Save the processed data to a JSON file
with open(auto_data_data_path, 'w', encoding='utf-8') as f:
    json.dump(ocr_data, f, indent=4, ensure_ascii=False)

print(f"Processing complete. Data saved to {auto_data_data_path}")
print(f"Total samples processed: {len(ocr_data)}")