import os
from openai import AzureOpenAI

endpoint = ""
model_name = "model-router"
deployment = "model-router"
subscription_key = ""
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Write a comprehensive account of solo backpacking adventure through South America, covering countries like Colombia, Peru, Bolivia, and Argentina. Detail the challenges and triumphs of traveling solo, your interactions with locals, and the cultural diversity you encountered along the way. Include must-visit attractions, unique experiences like hiking the Inca Trail or exploring Patagonia, and recommendations for budget travelers. Provide a structured itinerary, a cost breakdown, and safety tips for solo adventurers in the region.",
        }
    ],
    max_tokens=8192,
    temperature=0.7,
    top_p=0.95,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    model=deployment
)

print("Model chosen by the router: ", response.model)
print(response.choices[0].message.content)