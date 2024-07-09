import pickle
import re
import sqlite3
import os
import modal
import logging
import json
from openai import OpenAI
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the Modal app
image = modal.Image.debian_slim().pip_install("openai", "scikit-learn")
app = modal.App(name="chronicling-america-reduce", image=image)

# Volume for persistent storage
volume = modal.Volume.from_name("chronicling-america-vol", create_if_missing=True)

# SQLite database file path
db_file_path = "/my_vol/chronicling_america.db"

# folder for intermediates
intermediate_folder = "/my_vol/intermediates"


@app.function(volumes={"/my_vol": volume}, secrets=[modal.Secret.from_dotenv()])
def fetch_summaries():
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    cursor.execute("SELECT date, summary FROM summaries WHERE summary != ''")
    summaries = cursor.fetchall()
    logger.info(f"Fetched {len(summaries)} non-empty summaries from the database.")
    conn.close()
    return summaries


@app.function(volumes={"/my_vol": volume}, secrets=[modal.Secret.from_dotenv()])
def process_summaries(summaries):
    # Check if the result is in the intermediate_folder
    if os.path.exists(intermediate_folder) and os.path.isfile(
        os.path.join(intermediate_folder, "entity_events.pkl")
    ):
        with open(os.path.join(intermediate_folder, "entity_events.pkl"), "rb") as f:
            entity_events = pickle.load(f)
            return entity_events

    entity_events = defaultdict(list)

    for date, summary in summaries:
        summary_dict = json.loads(summary)
        for entities, event_summary in summary_dict.items():
            try:
                if "[" not in entities:
                    entities = "[" + entities
                if "]" not in entities:
                    entities = entities + "]"

                if (
                    entities.startswith("'")
                    or entities.startswith("‘")
                    or entities.startswith('"')
                ):
                    entities = entities[1:]
                if (
                    entities.endswith("'")
                    or entities.endswith("’")
                    or entities.endswith('"')
                ):
                    entities = entities[:-1]

                entity_string = (
                    entities.replace("['", '["')
                    .replace("[‘", '["')
                    .replace("’]", '"]')
                    .replace("']", '"]')
                    .replace("', '", '", "')
                    .replace("’, ‘", '", "')
                )

                entity_list = json.loads(entity_string)
                for entity in entity_list:
                    standardized_entity = entity.lower().strip()
                    entity_events[standardized_entity].append((date, event_summary))
            except json.JSONDecodeError:
                logger.error(f"Failed to decode entities: {entities}")
            except Exception as e:
                logger.error(f"Exception while processing entities {entities}: {e}")

    # Log statistics about the entities
    entity_frequency = defaultdict(int)
    for entity, events in entity_events.items():
        entity_frequency[entity] = len(events)

    # Sort entities by frequency in descending order
    sorted_entities = sorted(
        entity_frequency.items(), key=lambda item: item[1], reverse=True
    )

    # Log the top 10 most frequent entities
    logger.info("Top 10 most frequent entities:")
    for entity, frequency in sorted_entities[:10]:
        logger.info(f"Entity: {entity}, Frequency: {frequency}")

    num_distinct_entities = len(entity_events)
    logger.info(f"Number of distinct entities: {num_distinct_entities}")

    # Write to intermediate_folder
    # Create intermediate_folder if it doesn't exist
    if not os.path.exists(intermediate_folder):
        os.makedirs(intermediate_folder)
    with open(os.path.join(intermediate_folder, "entity_events.pkl"), "wb") as f:
        pickle.dump(entity_events, f)

    return entity_events


@app.function(
    volumes={"/my_vol": volume}, secrets=[modal.Secret.from_dotenv()], timeout=10000
)
def create_embeddings(entity_events):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = {}
    token_count = 0

    # Look in cache
    if os.path.exists(intermediate_folder) and os.path.isfile(
        os.path.join(intermediate_folder, "embeddings.pkl")
    ):
        with open(os.path.join(intermediate_folder, "embeddings.pkl"), "rb") as f:
            embeddings = pickle.load(f)
            return embeddings, token_count

    def fetch_embedding(entity):
        response = client.embeddings.create(
            model="text-embedding-3-large", input=entity, encoding_format="float"
        )
        return (
            entity,
            response.data[0].embedding,
            response.usage.total_tokens,
        )

    entities = list(entity_events.keys())
    with ThreadPoolExecutor(max_workers=1000) as executor:
        futures = []
        for i in range(0, len(entities), 1000):
            batch = entities[i : i + 1000]
            for entity in batch:
                futures.append(executor.submit(fetch_embedding, entity))
            for future in as_completed(futures):
                entity, embedding, tokens = future.result()
                embeddings[entity] = embedding
                token_count += tokens
            futures.clear()  # Clear the futures list
            logger.info(
                f"Processed batch {i // 1000 + 1}/{(len(entities) + 999) // 1000}"
            )
            time.sleep(10)  # Wait for 10 seconds after processing each batch

    # Write to intermediate_folder
    # Create intermediate_folder if it doesn't exist
    logger.info(f"Writing embeddings to {intermediate_folder}")
    if not os.path.exists(intermediate_folder):
        os.makedirs(intermediate_folder)
    with open(os.path.join(intermediate_folder, "embeddings.pkl"), "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings, token_count


def cluster_entities_tfidf(entity_events):

    entities = list(entity_events.keys())  # Convert dict_keys to list
    vectorizer = TfidfVectorizer().fit_transform(entities)
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)

    # Calculate the 90th percentile threshold for cosine similarities
    similarity_values = cosine_sim[np.triu_indices_from(cosine_sim, k=1)]
    threshold = np.percentile(similarity_values, 99.9)

    logger.info(f"The threshold for cosine similarities is: {threshold}")

    clusters = defaultdict(list)

    for i, entity in enumerate(entities):
        for j, similarity in enumerate(cosine_sim[i]):
            if i != j and similarity > threshold:
                clusters[i].append(j)

    clustered_entity_events = defaultdict(list)
    for cluster_id, indices in clusters.items():
        primary_entity = entities[cluster_id]
        clustered_entity_events[primary_entity].extend(entity_events[primary_entity])
        for index in indices:
            entity = entities[index]
            clustered_entity_events[primary_entity].extend(entity_events[entity])

    logger.info(f"Total number of clusters formed: {len(clusters)}")
    top_clusters = sorted(
        clusters.items(), key=lambda item: len(item[1]), reverse=True
    )[:5]
    for cluster_id, members in top_clusters:
        logger.info(f"Top Cluster {cluster_id} contains {len(members)} entities.")

    return clustered_entity_events


@app.function(
    volumes={"/my_vol": volume}, secrets=[modal.Secret.from_dotenv()], timeout=10000
)
def cluster_entities(embeddings, entity_events):
    # Check if the result is in the intermediate_folder
    if os.path.exists(intermediate_folder) and os.path.isfile(
        os.path.join(intermediate_folder, "clusters.pkl")
    ):
        with open(os.path.join(intermediate_folder, "clusters.pkl"), "rb") as f:
            clusters = pickle.load(f)
            return clusters

    entity_list = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[entity] for entity in entity_list])
    cosine_sim = cosine_similarity(embedding_matrix)

    similarity_values = cosine_sim[np.triu_indices_from(cosine_sim, k=1)]
    threshold = np.percentile(similarity_values, 99.9)

    logger.info(f"The threshold for cosine similarities is: {threshold}")

    clusters = defaultdict(list)

    for i, entity in enumerate(entity_list):
        for j, similarity in enumerate(cosine_sim[i]):
            if i != j and similarity > threshold:
                clusters[i].append(j)

    clustered_entity_events = defaultdict(list)
    for cluster_id, indices in clusters.items():
        primary_entity = entity_list[cluster_id]
        clustered_entity_events[primary_entity].extend(entity_events[primary_entity])
        for index in indices:
            entity = entity_list[index]
            clustered_entity_events[primary_entity].extend(entity_events[entity])

    logger.info(
        f"Total number of clusters formed: {len(clusters)} for {len(entity_list)} entities."
    )
    top_clusters = sorted(
        clusters.items(), key=lambda item: len(item[1]), reverse=True
    )[:5]
    for cluster_id, members in top_clusters:
        logger.info(
            f"Top Cluster {cluster_id} contains {len(members)} entities: {members}"
        )

    # Write to intermediate_folder
    # Create intermediate_folder if it doesn't exist
    if not os.path.exists(intermediate_folder):
        os.makedirs(intermediate_folder)
    with open(os.path.join(intermediate_folder, "clusters.pkl"), "wb") as f:
        pickle.dump(clustered_entity_events, f)

    return clustered_entity_events


import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


@app.function(
    volumes={"/my_vol": volume}, secrets=[modal.Secret.from_dotenv()], timeout=86400
)
def generate_timelines(clusters, entity_events):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    timelines = {}
    prompt_tokens = 0
    completion_tokens = 0

    # Check if the result is in the intermediate_folder
    if os.path.exists(intermediate_folder) and os.path.isfile(
        os.path.join(intermediate_folder, "timelines.pkl")
    ):
        with open(os.path.join(intermediate_folder, "timelines.pkl"), "rb") as f:
            timelines = pickle.load(f)
            return timelines, prompt_tokens, completion_tokens

    def fetch_timeline(entity, combined_events):
        timeline_prompt = f"Generate a report of events for the following entity: {entity}. The events are:\n"
        for date, event_summary in combined_events:
            timeline_prompt += f"- {date}: {event_summary}\n"

        timeline = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates reports of historical events for given entities. The events are taken from historical newspapers. The report should be in prose form, not a list of events.",
                },
                {
                    "role": "user",
                    "content": timeline_prompt,
                },
            ],
        )
        return (
            entity,
            timeline.choices[0].message.content,
            timeline.usage.prompt_tokens,
            timeline.usage.completion_tokens,
        )

    entities = list(clusters.keys())
    with ThreadPoolExecutor(max_workers=1000) as executor:
        futures = []
        for i in range(0, len(entities), 1000):
            batch = entities[i : i + 1000]
            for entity in batch:
                combined_events = sorted(
                    clusters[entity], key=lambda x: x[0]
                )  # Sort by date
                futures.append(executor.submit(fetch_timeline, entity, combined_events))

            for future in as_completed(futures):
                entity, timeline_content, curr_prompt_tokens, curr_completion_tokens = (
                    future.result()
                )
                timelines[entity] = timeline_content
                prompt_tokens += curr_prompt_tokens
                completion_tokens += curr_completion_tokens

            # Print cost so far
            total_gpt4o_cost = compute_gpt4o_cost(prompt_tokens, completion_tokens)
            logger.info(f"Total cost for GPT-4o so far: ${total_gpt4o_cost:.2f}")

            futures.clear()  # Clear the futures list
            logger.info(
                f"Processed batch {i // 1000 + 1}/{(len(entities) + 999) // 1000}"
            )
            time.sleep(10)  # Wait for 10 seconds after processing each batch

    # Write to intermediate_folder
    # Create intermediate_folder if it doesn't exist
    if not os.path.exists(intermediate_folder):
        os.makedirs(intermediate_folder)
    with open(os.path.join(intermediate_folder, "timelines.pkl"), "wb") as f:
        pickle.dump(timelines, f)

    return timelines, prompt_tokens, completion_tokens


@app.function(volumes={"/my_vol": volume})
def store_timelines(timelines, incremental=False):
    volume.reload()
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    table_name = "incremental_entity_timelines" if incremental else "entity_timelines"

    # Create the new table if it doesn't exist
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            entity TEXT,
            timeline TEXT
        )
    """
    )

    for entity, timeline in timelines.items():
        cursor.execute(
            f"INSERT INTO {table_name} (entity, timeline) VALUES (?, ?)",
            (entity, timeline),
        )

    conn.commit()

    # Print how many timelines were stored
    logger.info(f"Stored {len(timelines)} timelines in the database to {table_name}.")

    conn.close()
    volume.commit()


def compute_cost(embedding_tokens):
    logger.info(f"Total number of tokens used for embeddings: {embedding_tokens}")
    embedding_cost_per_million = 0.13
    total_cost = (embedding_tokens / 1_000_000) * embedding_cost_per_million
    return total_cost


def compute_gpt4o_cost(prompt_tokens, completion_tokens):
    prompt_cost_per_million = 5.00
    completion_cost_per_million = 15.00

    total_cost = (prompt_tokens / 1_000_000) * prompt_cost_per_million + (
        completion_tokens / 1_000_000
    ) * completion_cost_per_million
    return total_cost


@app.function(volumes={"/my_vol": volume}, secrets=[modal.Secret.from_dotenv()])
def sample_entity_timelines(sample_size=10):
    """
    Query the entity_timelines table and print a sample of timelines.

    Args:
    sample_size (int): The number of samples to retrieve and print.
    """
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    # Fetch a sample of timelines
    cursor.execute(
        """
        SELECT entity, timeline FROM entity_timelines
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (sample_size,),
    )
    samples = cursor.fetchall()

    # Print the samples
    for entity, timeline in samples:
        logger.info(f"Entity: {entity}\nTimeline:\n\t{timeline}")

    conn.close()


def calculate_k(clusters):
    total_events = sum(len(events) for events in clusters.values())
    num_entities = len(clusters)
    logger.info(f"Total number of events: {total_events}")
    logger.info(f"Total number of entities: {num_entities}")
    avg_events_per_entity = total_events / num_entities
    k = max(1, int(avg_events_per_entity / 2))
    logger.info(f"Calculated k value: {k}")
    return k


@app.function(
    volumes={"/my_vol": volume}, secrets=[modal.Secret.from_dotenv()], timeout=86400
)
def generate_incremental_timelines(clusters, entity_events):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    timelines = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Check if the result is in the intermediate_folder
    if os.path.exists(intermediate_folder) and os.path.isfile(
        os.path.join(intermediate_folder, "incremental_timelines.pkl")
    ):
        with open(
            os.path.join(intermediate_folder, "incremental_timelines.pkl"), "rb"
        ) as f:
            timelines = pickle.load(f)
            return timelines, total_prompt_tokens, total_completion_tokens

    k = calculate_k(clusters)

    logger.info(f"Ideal k value for generation: {k}")

    def fetch_incremental_timeline(entity, combined_events, previous_report=""):
        local_prompt_tokens = 0
        local_completion_tokens = 0
        for i in range(0, len(combined_events), k):
            event_chunk = combined_events[i : i + k]
            timeline_prompt = (
                f"Here is the previous report of events for the entity: {entity}\n{previous_report}\n\nAdd the following events to the report for the entity: {entity}.\n"
                if previous_report
                else f"Generate a report of events for the following entity: {entity}. The events are:\n"
            )
            for date, event_summary in event_chunk:
                timeline_prompt += f"- {date}: {event_summary}\n"

            timeline = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates reports of historical events for given entities. The events are taken from historical newspapers. The report should be in prose form, not a list of events.",
                    },
                    {
                        "role": "user",
                        "content": timeline_prompt,
                    },
                ],
            )
            previous_report = timeline.choices[0].message.content
            local_prompt_tokens += timeline.usage.prompt_tokens
            local_completion_tokens += timeline.usage.completion_tokens

        return entity, previous_report, local_prompt_tokens, local_completion_tokens

    entities = list(clusters.keys())
    with ThreadPoolExecutor(max_workers=1000) as executor:
        futures = []
        for i in range(0, len(entities), 1000):
            batch = entities[i : i + 1000]
            for entity in batch:
                combined_events = sorted(
                    clusters[entity], key=lambda x: x[0]
                )  # Sort by date
                futures.append(
                    executor.submit(fetch_incremental_timeline, entity, combined_events)
                )

            for future in as_completed(futures):
                entity, timeline_content, prompt_tokens, completion_tokens = (
                    future.result()
                )
                timelines[entity] = timeline_content
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

            # Print cost so far
            total_gpt4o_cost = compute_gpt4o_cost(
                total_prompt_tokens, total_completion_tokens
            )
            logger.info(f"Total cost for GPT-4o so far: ${total_gpt4o_cost:.2f}")

            futures.clear()  # Clear the futures list
            logger.info(
                f"Processed batch {i // 1000 + 1}/{(len(entities) + 999) // 1000}"
            )
            time.sleep(10)  # Wait for 10 seconds after processing each batch

            # Write to intermediate_folder
            if not os.path.exists(intermediate_folder):
                os.makedirs(intermediate_folder)
            with open(
                os.path.join(intermediate_folder, "incremental_timelines.pkl"), "wb"
            ) as f:
                pickle.dump(timelines, f)

    return timelines, total_prompt_tokens, total_completion_tokens


def generate_and_store_incremental():
    summaries = fetch_summaries.remote()
    entity_events = process_summaries.remote(summaries)
    print(f"Loading clusters for {len(entity_events)} entities...")
    clusters = cluster_entities.remote([], entity_events)

    print(f"Generating incremental timelines for {len(clusters)} entities...")
    timelines, prompt_tokens, completion_tokens = generate_incremental_timelines.remote(
        clusters, entity_events
    )

    # Print costs
    total_gpt4o_cost = compute_gpt4o_cost(prompt_tokens, completion_tokens)
    print(f"Total cost for GPT-4o: ${total_gpt4o_cost:.2f}")

    print(f"Storing incremental timelines for {len(timelines)} entities...")
    store_timelines.remote(timelines, incremental=True)


@app.local_entrypoint()
def main():
    # summaries = fetch_summaries.remote()
    # entity_events = process_summaries.remote(summaries)

    # print(f"Loading embeddings for {len(entity_events)} entities...")
    # embeddings, embedding_tokens = create_embeddings.remote(entity_events)

    # print("Clustering entities...")
    # clusters = cluster_entities.remote(embeddings, entity_events)

    # print("Loading clusters...")
    # clusters = cluster_entities.remote([], entity_events)

    # print("Generating timelines...")
    # timelines, prompt_tokens, completion_tokens = generate_timelines.remote(
    #     clusters, entity_events
    # )

    # sample_entity_timelines.remote(10)

    # store_timelines.remote(timelines)

    generate_and_store_incremental()
