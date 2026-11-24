import base64
import io
import requests
import streamlit as st

from PIL import Image


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

OPENROUTER_MODEL = "deepseek/deepseek-v4.1-flash"


def analyze_with_openrouter(image, detected_objects):

    # Convert image to JPEG
    image = image.convert("RGB")
    image.thumbnail((1024, 1024))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)

    # Convert image into Base64
    base64_image = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    # Prepare detector results
    if detected_objects:
        objects_text = "\n".join(
            f"- {item['class_name']} "
            f"(confidence: {item['confidence']})"
            for item in detected_objects
        )
    else:
        objects_text = (
            "No objects were detected above the confidence threshold."
        )

    # Environmental analysis prompt
    prompt = f"""
You are an assistant supporting underwater marine debris assessment.

The Faster R-CNN detector produced these predictions:

{objects_text}

Study the attached underwater image and provide a concise environmental analysis.

Use these headings:
1. Visible objects
2. Environmental observations
3. Potential environmental impacts
4. Possible marine risks
5. Suggested cleanup or monitoring actions
6. Overall assessment

Only describe what is visually supported.
Separate visible evidence from possibilities.
Do not claim chemical contamination is confirmed.
Do not treat detector confidence as proof of certainty.
"""

    # API request
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": 900,
        "reasoning": {
            "enabled": False
        }
    }

    headers = {
        "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        OPENROUTER_API_URL,
        headers=headers,
        json=payload,
        timeout=180
    )

    response.raise_for_status()

    response_data = response.json()

    return response_data["choices"][0]["message"]["content"]