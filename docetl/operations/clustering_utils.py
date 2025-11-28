"""
This module contains utilities for clustering based on different methods.

We use these in map and reduce operations.
"""

import json

from docetl.operations.utils import APIWrapper
from docetl.utils import completion_cost


def get_embeddings_for_clustering(
    items: list[dict], sampling_config: dict, api_wrapper: APIWrapper
) -> tuple[list[list[float]], float]:
    embedding_model = sampling_config.get("embedding_model", "text-embedding-3-small")
    embedding_keys = sampling_config.get("embedding_keys")
    if not embedding_keys:
        embedding_keys = list(items[0].keys())

    if embedding_model == "sentence-transformer":
        return get_embeddings_for_clustering_with_st(items, embedding_keys)

    embeddings = []
    cost = 0
    batch_size = 1000

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        texts = [
            " ".join(str(item[key]) for key in embedding_keys if key in item)[:1000]
            for item in batch
        ]
        response = api_wrapper.gen_embedding(embedding_model, json.dumps(texts))
        embeddings.extend([data["embedding"] for data in response["data"]])
        cost += completion_cost(response)

    return embeddings, cost


def get_embeddings_for_clustering_with_st(
    items: list[dict], embedding_keys: list[str]
) -> tuple[list[list[float]], float]:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embeddings = model.encode(
        [
            " ".join(str(item[key]) for key in embedding_keys if key in item)[:10000]
            for item in items
        ]
    )
    return embeddings, 0


def cluster_documents(
    documents: list[dict],
    sampling_config: dict,
    sample_size: int,
    api_wrapper: APIWrapper,
) -> tuple[dict[int, list[dict]], float]:
    """
    Cluster documents using KMeans clustering algorithm.

    Args:
        documents (list[dict]): The list of documents to cluster.
        sampling_config (dict): The sampling configuration. Must contain embedding_model. If embedding_keys is not specified, it will use all keys in the document. If embedding_model is not specified, it will use text-embedding-3-small. If embedding_model is sentence-transformer, it will use all-MiniLM-L6-v2.
        sample_size (int): The number of clusters to create.
        api_wrapper (APIWrapper): The API wrapper to use for embedding.
    Returns:
        dict[int, list[dict]]: A dictionary of clusters, where each cluster is a list of documents.
    """
    embeddings, cost = get_embeddings_for_clustering(
        documents, sampling_config, api_wrapper
    )

    from sklearn.cluster import KMeans

    num_clusters = min(sample_size, len(documents))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    clusters = {i: [] for i in range(num_clusters)}
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(documents[idx])

    return clusters, cost
