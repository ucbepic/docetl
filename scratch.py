import random
from typing import List, Tuple, Any
from motion.dataset import Dataset
from motion.operators import (
    LLMMapper,
    LLMReducer,
    LLMListKeyResolver,
    LLMFlatMapper,
)

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

MODEL = "gpt-4o-mini"

# Generate synthetic data
items = [
    "apple",
    "banana",
    "cherry",
    "date",
    "elderberry",
    "fig",
    "grape",
    "honeydew",
    "kiwi",
    "lemon",
    "dog",
    "cat",
    "horse",
    "cow",
    "pig",
    "sheep",
    "goat",
    "chicken",
    "duck",
    "rabbit",
    "car",
    "bicycle",
    "train",
    "airplane",
    "boat",
    "bus",
    "motorcycle",
    "truck",
    "scooter",
    "helicopter",
    "chair",
    "table",
    "bed",
    "sofa",
    "dresser",
    "desk",
    "bookshelf",
    "lamp",
    "mirror",
    "rug",
    "phone",
    "computer",
    "tablet",
    "camera",
    "watch",
    "television",
    "printer",
    "headphones",
    "speaker",
    "keyboard",
]

# Create initial dataset with importance scores
data = [(item, random.randint(1, 5)) for item in items]
dataset = Dataset(data)


# LLMMapper to categorize items
class CategoryMapper(LLMMapper):
    def generate_prompt(self, key: str, value: int) -> list:
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant that categorizes items.",
            },
            {
                "role": "user",
                "content": f"Please categorize this item into a higher-level category. Item: {key}\nYour answer should be a single word.",
            },
        ]

    def process_response(
        self, response: Any, **prompt_kwargs
    ) -> Tuple[str, Tuple[str, int]]:
        category = response.choices[0].message.content.strip()
        return (category, (prompt_kwargs["key"], prompt_kwargs["value"]))


class CategoryResolver(LLMListKeyResolver):
    def generate_prompt(self, key: str, label_keys: List[str]) -> list:
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant that determines the most appropriate category for an item.",
            },
            {
                "role": "user",
                "content": f"Given the category '{key}' and the existing categories {label_keys}, which category should it be assigned to? If it doesn't match any existing categories, respond with 'NEW'. Provide your answer as a single word or 'NEW'.",
            },
        ]

    def process_response(self, response: Any, **prompt_kwargs) -> str:
        content = response.choices[0].message.content.strip()
        return content if content != "NEW" else prompt_kwargs["key"]

    def get_label_key(self, keys: set) -> str:
        return min(keys)


# LLMReducer to generate a story for each category
class StoryGenerator(LLMReducer):
    def generate_prompt(self, key: str, values: List[Tuple[str, int]]) -> list:
        items = ", ".join(
            [f"{item} (importance: {importance}/5)" for item, importance in values]
        )
        return [
            {
                "role": "system",
                "content": "You are a creative storyteller that creates cute, short stories.",
            },
            {
                "role": "user",
                "content": f"Create a cute, short story about these items in the category '{key}'. Focus more on items with higher importance: {items}",
            },
        ]

    def process_response(self, response: Any, **prompt_kwargs) -> str:
        return response.choices[0].message.content.strip()


# Apply operations
result = (
    dataset.map(CategoryMapper(model=MODEL))
    .resolve_keys(CategoryResolver(model=MODEL))
    .reduce(StoryGenerator(model=MODEL))
    .execute()
)

# Print results


for category, story in result:
    print(f"\nCategory: {category}")
    print(f"Story: {story}")
    print("-" * 50)

# Print the number of categories
print(f"Number of categories: {len(result)}")
