import random

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

# Function to format the data, since the image is already a PIL.Image object
def format_data(sample):

    # Create 'messages' structure
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an helpful assistant who does OCR by extracting texts from Nigeria."}]},
        {"role": "user", "content": [
            {"type": "text", "text": random.choice(description_list)},
            {"type": "image", "image": sample["image"], },
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": sample['text']}]}
    ]

    return {
        "messages": messages
    }


