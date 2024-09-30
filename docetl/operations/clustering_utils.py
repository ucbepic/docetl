"""
This module contains utilities for clustering based on different methods.

We use these in map and reduce operations.
"""

import random
from typing import Dict, List, Literal, Tuple

from docetl.operations.utils import gen_embedding
from docetl.utils import completion_cost


def get_embeddings_for_clustering(
    items: List[Dict], sampling_config: Dict
) -> Tuple[List[List[float]], float]:
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
            " ".join(str(item[key]) for key in embedding_keys if key in item)[:10000]
            for item in batch
        ]
        response = gen_embedding(embedding_model, texts)
        embeddings.extend([data["embedding"] for data in response["data"]])
        cost += completion_cost(response)

    return embeddings, cost


def get_embeddings_for_clustering_with_st(
    items: List[Dict], embedding_keys: List[str]
) -> Tuple[List[List[float]], float]:
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
    documents: List[Dict], sampling_config: Dict, sample_size: int
) -> Tuple[Dict[int, List[Dict]], float]:
    """
    Cluster documents using KMeans clustering algorithm.

    Args:
        documents (List[Dict]): The list of documents to cluster.
        sampling_config (Dict): The sampling configuration. Must contain embedding_model. If embedding_keys is not specified, it will use all keys in the document. If embedding_model is not specified, it will use text-embedding-3-small. If embedding_model is sentence-transformer, it will use all-MiniLM-L6-v2.
        sample_size (int): The number of clusters to create.

    Returns:
        Dict[int, List[Dict]]: A dictionary of clusters, where each cluster is a list of documents.
    """
    embeddings, cost = get_embeddings_for_clustering(documents, sampling_config)

    from sklearn.cluster import KMeans

    num_clusters = min(sample_size, len(documents))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    clusters = {i: [] for i in range(num_clusters)}
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(documents[idx])

    return clusters, cost


def cluster_documents_for_map(
    documents: List[Dict],
    max_batch_size: int,
    clustering_method: Literal["random", "sem_cluster"],
) -> Tuple[List[List[Dict]], float]:
    """
    Clusters documents based on the configured clustering method.

    Args:
        documents (List[Dict]): The list of documents to cluster.
        batch_size (int): The number of documents to cluster.
        clustering_method (str): The clustering method to use.

    Returns:
        List[List[Dict]]: A list of clusters, where each cluster is a list of documents.
        float: The cost of the clustering.
    """

    # If there are no documents, return an empty list
    if not documents:
        return [], 0

    # If the batch size is infinite, return the documents as a single cluster
    if max_batch_size == float("inf"):
        return [documents], 0

    # If the batch size is greater than 1, cluster the documents
    if max_batch_size > 1:
        # If the clustering method is random, shuffle the documents and return batches
        if clustering_method == "random":
            random.seed(42)
            random.shuffle(documents)
            return [
                documents[i : i + max_batch_size]
                for i in range(0, len(documents), max_batch_size)
            ], 0

        # If the clustering method is sem_cluster, use kmeans clustering
        elif clustering_method == "sem_cluster":
            clusters, cost = cluster_documents(
                documents, len(documents) // max_batch_size
            )

            # For each cluster, create batches and add to the list
            batches = []
            for cluster in clusters:
                batches.extend(
                    [
                        cluster[i : i + max_batch_size]
                        for i in range(0, len(cluster), max_batch_size)
                    ]
                )
            return batches, cost

    else:
        raise ValueError("max_batch_size must be greater than 0.")
