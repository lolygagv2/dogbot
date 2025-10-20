#!/usr/bin/env python3
"""
Connect DogBot to Claude/GPT for intelligent narration
"""

import base64
import requests
from anthropic import Anthropic  # pip install anthropic

class DogBotNarrator:
    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)
        
    def describe_dog_behavior(self, image_path, detected_behavior):
        """Send image to Claude for rich description"""
        
        # Encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": f"I detected my dog '{detected_behavior}'. Can you describe what you see and suggest a training response?"
                    }
                ]
            }]
        )
        
        return response.content[0].text

# Usage
narrator = DogBotNarrator(api_key="your-api-key")
description = narrator.describe_dog_behavior(
    "elsa_sitting.jpg",
    "sitting"
)
print(description)
# "I see a small white dog sitting attentively. The posture is excellent 
#  with straight back. This is a perfect moment to reward with a treat
#  and verbal praise to reinforce this behavior."