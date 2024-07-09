import re
import sqlite3
import os
from datetime import datetime, timedelta
import modal
import itertools
import logging
import hashlib
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the Modal app
image = modal.Image.debian_slim().pip_install("openai")
app = modal.App(name="chronicling-america-summaries", image=image)

# Volume for persistent storage
volume = modal.Volume.from_name("chronicling-america-vol", create_if_missing=True)

# SQLite database file path
db_file_path = "/my_vol/chronicling_america.db"


@app.function(volumes={"/my_vol": volume}, secrets=[modal.Secret.from_dotenv()])
def fetch_texts():
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    cursor.execute("SELECT url, summary FROM summaries LIMIT 5")
    sample_summaries = cursor.fetchall()
    for url, summary in sample_summaries:
        logger.info(f"URL: {url}, Summary: {summary}")

    cursor.execute("SELECT COUNT(*) FROM summaries")
    count_summaries = cursor.fetchone()[0]
    logger.info(f"Total number of summaries in the database: {count_summaries}")

    # Select URLs that are not in the summaries table
    cursor.execute(
        """
        SELECT url, ocr_text 
        FROM issues 
        WHERE status='success' 
        AND url NOT IN (SELECT url FROM summaries)
        ORDER BY SUBSTR(url, INSTR(url, '/') + 1, 10) ASC 
    """
    )

    texts = cursor.fetchall()
    logger.info(f"Fetched {len(texts)} URLs and OCR texts from the database.")
    conn.close()

    urls = [text[0] for text in texts]
    texts = [text[1] for text in texts]

    return urls, texts


@app.function(volumes={"/my_vol": volume}, secrets=[modal.Secret.from_dotenv()])
def summarize_text(url, ocr_text):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    date_match = re.search(r"/(\d{4}-\d{2}-\d{2})/", url)
    date_str = date_match.group(1) if date_match else "unknown date"
    if date_str == "unknown date":
        return None

    user_prompt = f"""Please extract short (one or two sentence) summaries for each significant world event, crime, or gossip reported on the front page of the Chicago Tribune dated {date_str}. Your response should be formatted as a JSON dictionary where the keys are lists of entities involved (up to three), and the values are the summaries. The summary should be self-contained and not rely on the context of the entire issue. Here is an example of the format:

{{
    "['Entity1', 'Entity2']": "Summary of the event involving Entity1 and Entity2.",
    "['Entity3']": "Summary of the gossip involving Entity3."
}}

The OCR text is as follows:

{ocr_text}"""

    # Strip the text of any non-ASCII characters, newlines, and extra spaces
    user_prompt = re.sub(r"[^\x00-\x7F]+", " ", user_prompt)
    user_prompt = re.sub(r"[\s\n]+", " ", user_prompt)

    summary = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant, helping me find interesting and useful information in an old newspaper.",
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    token_counts = {
        "prompt": summary.usage.prompt_tokens,
        "completion": summary.usage.completion_tokens,
    }

    logger.info(f"Processed {date_str} and got {summary.choices[0].message.content}")

    return url, date_str, summary.choices[0].message.content, token_counts


@app.function(volumes={"/my_vol": volume})
def store_summaries(summaries, token_counts):
    volume.reload()
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    # Create the new table if it doesn't exist
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS summaries (
            url TEXT,
            date TEXT,
            summary TEXT
        )
    """
    )

    for url, date_str, summary in summaries:
        cursor.execute(
            "INSERT INTO summaries (url, date, summary) VALUES (?, ?, ?)",
            (url, date_str, summary),
        )

    conn.commit()

    # Print how many summaries were stored
    logger.info(f"Stored {len(summaries)} summaries in the database.")

    conn.close()
    volume.commit()

    return compute_cost(token_counts)


def compute_cost(token_counts_list):
    prompt_cost_per_million = 5.00
    completion_cost_per_million = 15.00
    total_cost = 0

    for token_counts in token_counts_list:
        prompt_tokens = token_counts.get("prompt", 0)
        completion_tokens = token_counts.get("completion", 0)

        total_cost += (prompt_tokens / 1_000_000) * prompt_cost_per_million + (
            completion_tokens / 1_000_000
        ) * completion_cost_per_million

    return total_cost


@app.local_entrypoint()
def main():
    urls, texts = fetch_texts.remote()
    all_pairs = list(zip(urls, texts))
    results = []

    # Process in batches of 25
    for i in range(0, len(all_pairs), 25):
        batch = all_pairs[i : i + 25]
        results.extend(summarize_text.starmap(batch))

    summaries = []
    token_counts = []
    for result in results:
        if result:
            url, date_str, summary, token_count = result
            summaries.append((url, date_str, summary))
            token_counts.append(token_count)

    total_cost = store_summaries.remote(summaries, token_counts)

    print(f"Processed {len(summaries)} summaries.")
    print(f"Total cost: ${total_cost:.2f}")
