from typing import Any, Dict, List, Literal, Tuple, Union
import os
from pathlib import Path

import numpy as np
from pydantic import Field, field_validator, model_validator

from docetl.operations.base import BaseOperation
from docetl.operations.clustering_utils import get_embeddings_for_clustering


class SampleOperation(BaseOperation):
    """
    A sampling operation that can select a subset of items from input data.
    
    This operation supports multiple sampling methods:
    - uniform: Random uniform sampling
    - stratify: Stratified sampling based on a key
    - outliers: Sample based on embedding distance from center
    - custom: Sample specific items by matching keys
    - first: Take the first N items
    - retrieve_vector: Vector-based similarity search using LanceDB
    - retrieve_fts: Full-text search with optional semantic reranking using LanceDB
    """

    class schema(BaseOperation.schema):
        type: str = "sample"
        method: Literal["uniform", "stratify", "outliers", "custom", "first", "retrieve_vector", "retrieve_fts"]
        samples: Union[int, float, list] | None = None
        method_kwargs: dict[str, Any] | None = Field(default_factory=dict)
        random_state: int | None = Field(None, ge=0)

        @field_validator("samples")
        def validate_samples(cls, v, info):
            if v is not None:
                # For custom method, samples must be a list
                if hasattr(info.data, "method") and info.data.get("method") == "custom":
                    if not isinstance(v, list):
                        raise TypeError("'samples' must be a list for custom sampling")
                elif isinstance(v, (int, float)):
                    if v <= 0:
                        raise ValueError("'samples' must be a positive number")
            return v

        @field_validator("method_kwargs")
        def validate_method_kwargs(cls, v):
            if v is not None:
                if not isinstance(v, dict):
                    raise TypeError("'method_kwargs' must be a dictionary")

                # Validate specific keys in method_kwargs
                if "stratify_key" in v:
                    stratify_key = v["stratify_key"]
                    if isinstance(stratify_key, str):
                        pass  # Single key is fine
                    elif isinstance(stratify_key, list):
                        if not all(isinstance(key, str) for key in stratify_key):
                            raise TypeError("All items in 'stratify_key' list must be strings")
                        if len(stratify_key) == 0:
                            raise ValueError("'stratify_key' list cannot be empty")
                    else:
                        raise TypeError("'stratify_key' must be a string or list of strings")

                if "center" in v and not isinstance(v["center"], dict):
                    raise TypeError("'center' must be a dictionary")

                if "embedding_keys" in v:
                    if not isinstance(v["embedding_keys"], list) or not all(
                        isinstance(key, str) for key in v["embedding_keys"]
                    ):
                        raise TypeError("'embedding_keys' must be a list of strings")

                if "std" in v:
                    if not isinstance(v["std"], (int, float)) or v["std"] <= 0:
                        raise TypeError("'std' must be a positive number")

                if "samples" in v:
                    if not isinstance(v["samples"], (int, float)) or v["samples"] <= 0:
                        raise TypeError(
                            "'samples' in method_kwargs must be a positive number"
                        )

                if "samples_per_group" in v:
                    if not isinstance(v["samples_per_group"], bool):
                        raise TypeError("'samples_per_group' must be a boolean")

            return v

        @model_validator(mode="after")
        def validate_method_specific_requirements(self):
            method = self.method

            if method in ["uniform", "stratify"] and self.samples is None:
                raise ValueError(f"Must specify 'samples' for {method} sampling")

            if method == "stratify":
                method_kwargs = self.method_kwargs or {}
                if "stratify_key" not in method_kwargs:
                    raise ValueError(
                        "Must specify 'stratify_key' for stratify sampling"
                    )

            if method == "outliers":
                method_kwargs = self.method_kwargs or {}
                if "std" not in method_kwargs and "samples" not in method_kwargs:
                    raise ValueError(
                        "Must specify either 'std' or 'samples' in outliers configuration"
                    )

                if "embedding_keys" not in method_kwargs:
                    raise ValueError(
                        "'embedding_keys' must be specified in outliers configuration"
                    )

            if method == "custom" and self.samples is None:
                raise ValueError("Must specify 'samples' for custom sampling")

            if method in ["retrieve_vector", "retrieve_fts"]:
                method_kwargs = self.method_kwargs or {}
                if "query" not in method_kwargs:
                    raise ValueError(f"Must specify 'query' in method_kwargs for {method}")
                if "embedding_keys" not in method_kwargs:
                    raise ValueError(f"Must specify 'embedding_keys' in method_kwargs for {method}")

            return self

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._db = None
        self._table = None

    def execute(
        self, input_data: List[Dict], is_build: bool = False
    ) -> Tuple[List[Dict], float]:
        """
        Executes the sample operation on the input data.

        Args:
            input_data (List[Dict]): A list of dictionaries to process.
            is_build (bool): Whether the operation is being executed
              in the build phase. Defaults to False.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the filtered
              list of dictionaries and the total cost of the operation.
        """
        cost = 0
        if not input_data:
            return [], cost

        method = self.config["method"]
        
        if method == "first":
            return self._sample_first(input_data), cost
        elif method == "outliers":
            return self._sample_outliers(input_data)
        elif method == "custom":
            return self._sample_custom(input_data), cost
        elif method == "uniform":
            return self._sample_uniform(input_data), cost
        elif method == "stratify":
            return self._sample_stratified(input_data), cost
        elif method == "retrieve_vector":
            return self._retrieve_vector(input_data)
        elif method == "retrieve_fts":
            return self._retrieve_fts(input_data)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def _sample_first(self, input_data: List[Dict]) -> List[Dict]:
        """Take the first N items from the input data."""
        return input_data[: self.config["samples"]]

    def _sample_uniform(self, input_data: List[Dict]) -> List[Dict]:
        """Perform uniform random sampling."""
        import sklearn.model_selection
        
        output_data, _ = sklearn.model_selection.train_test_split(
            input_data,
            train_size=self.config["samples"],
            random_state=self.config.get("random_state", None),
            stratify=None,
        )
        return output_data

    def _sample_stratified(self, input_data: List[Dict]) -> List[Dict]:
        """Perform stratified sampling based on a key."""
        method_kwargs = self.config.get("method_kwargs", {})
        stratify_key = method_kwargs["stratify_key"]
        samples = self.config["samples"]
        samples_per_group = method_kwargs.get("samples_per_group", False)

        # Helper function to get stratify value(s) for an item
        def get_stratify_value(item):
            if isinstance(stratify_key, str):
                return item[stratify_key]
            else:  # list of keys
                return tuple(item[key] for key in stratify_key)

        if samples_per_group:
            # Sample N items from each group
            import random
            from collections import defaultdict

            # Group data by stratify key(s)
            groups = defaultdict(list)
            for item in input_data:
                groups[get_stratify_value(item)].append(item)

            # Sample from each group
            output_data = []
            random_state = self.config.get("random_state", None)
            if random_state is not None:
                random.seed(random_state)

            for group_items in groups.values():
                if not isinstance(samples, int):
                    group_samples = int(samples * len(group_items))
                else:
                    group_samples = min(samples, len(group_items))

                sampled_items = random.sample(group_items, group_samples)
                output_data.extend(sampled_items)
        else:
            # Original stratified sampling behavior
            stratify = [get_stratify_value(data) for data in input_data]

            import sklearn.model_selection

            output_data, _ = sklearn.model_selection.train_test_split(
                input_data,
                train_size=samples,
                random_state=self.config.get("random_state", None),
                stratify=stratify,
            )
        
        return output_data

    def _sample_custom(self, input_data: List[Dict]) -> List[Dict]:
        """Sample specific items based on matching keys."""
        samples = self.config["samples"]
        keys = list(samples[0].keys())
        
        # Create a mapping from key tuples to documents
        key_to_doc = {
            tuple([doc[key] for key in keys]): doc for doc in input_data
        }
        
        # Find matching documents
        output_data = []
        for sample in samples:
            key_tuple = tuple([sample[key] for key in keys])
            if key_tuple in key_to_doc:
                output_data.append(key_to_doc[key_tuple])
        
        return output_data

    def _sample_outliers(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        """Sample based on embedding distance from a center point."""
        outliers_config = self.config.get("method_kwargs", {})
        
        # Get embeddings for all input data
        embeddings, embedding_cost = get_embeddings_for_clustering(
            input_data, outliers_config, self.runner.api
        )
        embeddings = np.array(embeddings)
        
        # Determine the center point
        if "center" in outliers_config:
            center_embeddings, center_cost = get_embeddings_for_clustering(
                [outliers_config["center"]], outliers_config, self.runner.api
            )
            center = np.array(center_embeddings[0])
            total_cost = embedding_cost + center_cost
        else:
            center = embeddings.mean(axis=0)
            total_cost = embedding_cost
        
        # Calculate distances from center
        distances = np.sqrt(((embeddings - center) ** 2).sum(axis=1))
        
        # Determine cutoff threshold
        if "std" in outliers_config:
            # Use standard deviation based cutoff
            cutoff = (
                np.sqrt((embeddings.std(axis=0) ** 2).sum())
                * outliers_config["std"]
            )
        else:  # "samples" in outliers_config
            # Use percentile based cutoff
            distance_distribution = np.sort(distances)
            samples = outliers_config["samples"]
            if isinstance(samples, float):
                samples = int(samples * (len(distance_distribution) - 1))
            cutoff = distance_distribution[samples]
        
        # Determine which items to include
        keep = outliers_config.get("keep", False)
        include = distances > cutoff if keep else distances <= cutoff
        
        output_data = [item for idx, item in enumerate(input_data) if include[idx]]
        
        return output_data, total_cost

    def _initialize_db(self):
        """Initialize LanceDB connection."""
        try:
            import lancedb
        except ImportError:
            raise ImportError(
                "LanceDB is required for retrieve_vector and retrieve_fts methods. "
                "Install it with: pip install lancedb"
            )
        
        # Import DOCETL_HOME_DIR from utils
        from docetl.operations.utils import DOCETL_HOME_DIR
        
        method_kwargs = self.config.get("method_kwargs", {})
        
        # Default path: DOCETL_HOME_DIR/lancedb/{operation_name}
        default_db_path = os.path.join(DOCETL_HOME_DIR, "lancedb", self.config["name"])
        db_path = Path(method_kwargs.get("db_path", default_db_path))
        db_path.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        self._db = lancedb.connect(str(db_path))

    def _prepare_text_for_embedding(self, doc: dict, embedding_keys: List[str]) -> str:
        """Prepare document text for embedding by concatenating specified keys."""
        texts = []
        for key in embedding_keys:
            if key in doc:
                value = doc[key]
                if isinstance(value, str):
                    texts.append(value)
                elif value is not None:
                    texts.append(str(value))
        return " ".join(texts)

    def _retrieve_vector(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        """Perform vector-based similarity search using LanceDB."""
        try:
            import lancedb
            from lancedb.pydantic import LanceModel, Vector
        except ImportError:
            raise ImportError(
                "LanceDB is required for retrieve_vector method. "
                "Install it with: pip install lancedb"
            )
        
        self._initialize_db()
        
        method_kwargs = self.config.get("method_kwargs", {})
        query = method_kwargs["query"]
        embedding_keys = method_kwargs["embedding_keys"]
        num_chunks = method_kwargs.get("num_chunks", self.config.get("samples", 10))
        embedding_model = method_kwargs.get("embedding_model", "text-embedding-3-small")
        table_name = method_kwargs.get("table_name", "docetl_vectors")
        persist = method_kwargs.get("persist", False)
        output_key = method_kwargs.get("output_key", "_retrieved")
        stratify_key = method_kwargs.get("stratify_key", None)
        
        total_cost = 0.0
        
        # Helper function to get stratify value(s) for an item
        def get_stratify_value(item):
            if stratify_key is None:
                return None
            elif isinstance(stratify_key, str):
                return item.get(stratify_key)
            else:  # list of keys
                return tuple(item.get(key) for key in stratify_key)
        
        # Group data by stratify key if specified
        if stratify_key:
            from collections import defaultdict
            strata_groups = defaultdict(list)
            for i, doc in enumerate(input_data):
                strata_groups[get_stratify_value(doc)].append((i, doc))
        else:
            # No stratification - treat all data as one group
            strata_groups = {None: [(i, doc) for i, doc in enumerate(input_data)]}
        
        # Process each stratum separately
        all_results = [None] * len(input_data)
        
        for stratum_key, stratum_docs in strata_groups.items():
            # Prepare documents for embedding within this stratum
            docs_to_embed = []
            for idx, doc in stratum_docs:
                text = self._prepare_text_for_embedding(doc, embedding_keys)
                if text:
                    docs_to_embed.append({
                        "_original_index": idx,
                        "_stratum_index": len(docs_to_embed),
                        "text": text,
                        "document": doc
                    })
            
            if not docs_to_embed:
                # No valid documents in this stratum
                for idx, doc in stratum_docs:
                    all_results[idx] = {**doc, output_key: []}
                continue
            
            # Get embeddings for documents in this stratum
            texts = [d["text"] for d in docs_to_embed]
            embeddings, embedding_cost = get_embeddings_for_clustering(
                [{"text": t} for t in texts],
                {"embedding_keys": ["text"], "embedding_model": embedding_model},
                self.runner.api
            )
            total_cost += embedding_cost
            
            # Prepare data for LanceDB
            vector_dim = len(embeddings[0]) if embeddings else 0
            
            # Create dynamic model for LanceDB
            class DocumentVector(LanceModel):
                text: str
                vector: Vector(vector_dim)
                _stratum_index: int
                _original_index: int
                
            # Add document data to records
            records = []
            for i, (doc_info, embedding) in enumerate(zip(docs_to_embed, embeddings)):
                record = {
                    "text": doc_info["text"],
                    "vector": embedding,
                    "_stratum_index": doc_info["_stratum_index"],
                    "_original_index": doc_info["_original_index"]
                }
                records.append(record)
            
            # Create table name specific to this stratum
            if stratify_key:
                stratum_suffix = str(hash(str(stratum_key)))[-8:]  # Use last 8 chars of hash
                stratum_table_name = f"{table_name}_stratum_{stratum_suffix}"
            else:
                stratum_table_name = table_name
            
            # Create or overwrite table for this stratum
            if stratum_table_name in self._db.table_names():
                if not persist:
                    self._db.drop_table(stratum_table_name)
                    stratum_table = self._db.create_table(
                        stratum_table_name, 
                        data=records,
                        schema=DocumentVector
                    )
                else:
                    stratum_table = self._db.open_table(stratum_table_name)
                    stratum_table.add(records)
            else:
                stratum_table = self._db.create_table(
                    stratum_table_name,
                    data=records,
                    schema=DocumentVector
                )
            
            # Get embedding for the query
            query_embeddings, query_cost = get_embeddings_for_clustering(
                [{"text": query}],
                {"embedding_keys": ["text"], "embedding_model": embedding_model},
                self.runner.api
            )
            total_cost += query_cost
            
            if not query_embeddings:
                # If query embedding fails, return empty results for this stratum
                for idx, doc in stratum_docs:
                    all_results[idx] = {**doc, output_key: []}
                continue
                
            query_vector = query_embeddings[0]
            
            # Search for similar documents within this stratum
            search_results = (
                stratum_table.search(query_vector)
                .limit(num_chunks)
                .to_list()
            )
            
            # Prepare retrieved documents for this stratum
            retrieved_docs = []
            for result in search_results:
                # Get the original document from the stratum
                stratum_idx = result["_stratum_index"]
                if stratum_idx < len(docs_to_embed):
                    original_idx = docs_to_embed[stratum_idx]["_original_index"]
                    retrieved_doc = input_data[original_idx]
                    retrieved_docs.append({
                        **retrieved_doc,
                        "_distance": result.get("_distance", None)
                    })
            
            # Assign retrieved docs to all documents in this stratum
            for idx, doc in stratum_docs:
                all_results[idx] = {
                    **doc,
                    output_key: retrieved_docs
                }
        
        return all_results, total_cost

    def _retrieve_fts(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        """Perform full-text search with optional semantic reranking using LanceDB."""
        try:
            import lancedb
            from lancedb.pydantic import LanceModel, Vector
        except ImportError:
            raise ImportError(
                "LanceDB is required for retrieve_fts method. "
                "Install it with: pip install lancedb"
            )
        
        self._initialize_db()
        
        method_kwargs = self.config.get("method_kwargs", {})
        query = method_kwargs["query"]
        embedding_keys = method_kwargs["embedding_keys"]
        num_chunks = method_kwargs.get("num_chunks", self.config.get("samples", 10))
        embedding_model = method_kwargs.get("embedding_model", "text-embedding-3-small")
        table_name = method_kwargs.get("table_name", "docetl_fts")
        persist = method_kwargs.get("persist", False)
        rerank = method_kwargs.get("rerank", True)
        output_key = method_kwargs.get("output_key", "_retrieved")
        
        total_cost = 0.0
        
        # Prepare documents for indexing
        docs_to_index = []
        for i, doc in enumerate(input_data):
            text = self._prepare_text_for_embedding(doc, embedding_keys)
            if text:
                docs_to_index.append({
                    "_index": i,
                    "text": text,
                    "document": doc
                })
        
        if not docs_to_index:
            return [], 0.0
        
        # Get embeddings if reranking is enabled
        embeddings = None
        if rerank:
            texts = [d["text"] for d in docs_to_index]
            embeddings, embedding_cost = get_embeddings_for_clustering(
                [{"text": t} for t in texts],
                {"embedding_keys": ["text"], "embedding_model": embedding_model},
                self.runner.api
            )
            total_cost += embedding_cost
        
        # Prepare data for LanceDB
        if embeddings:
            vector_dim = len(embeddings[0])
            
            class DocumentWithVector(LanceModel):
                text: str
                vector: Vector(vector_dim)
                _index: int
                
            records = []
            for i, (doc_info, embedding) in enumerate(zip(docs_to_index, embeddings)):
                record = {
                    "text": doc_info["text"],
                    "vector": embedding,
                    "_index": doc_info["_index"]
                }
                records.append(record)
            
            schema = DocumentWithVector
        else:
            class DocumentText(LanceModel):
                text: str
                _index: int
                
            records = []
            for doc_info in docs_to_index:
                record = {
                    "text": doc_info["text"],
                    "_index": doc_info["_index"]
                }
                records.append(record)
            
            schema = DocumentText
        
        # Create or overwrite table
        if table_name in self._db.table_names():
            if not persist:
                self._db.drop_table(table_name)
                self._table = self._db.create_table(
                    table_name, 
                    data=records,
                    schema=schema
                )
            else:
                self._table = self._db.open_table(table_name)
                self._table.add(records)
        else:
            self._table = self._db.create_table(
                table_name,
                data=records,
                schema=schema
            )
        
        # Perform search
        if rerank and embeddings:
            # Get embedding for the query
            query_embeddings, query_cost = get_embeddings_for_clustering(
                [{"text": query}],
                {"embedding_keys": ["text"], "embedding_model": embedding_model},
                self.runner.api
            )
            total_cost += query_cost
            
            if query_embeddings:
                query_vector = query_embeddings[0]
                search_results = (
                    self._table.search(query_vector)
                    .limit(num_chunks)
                    .to_list()
                )
            else:
                search_results = self._perform_text_search(query, num_chunks)
        else:
            # Pure text search without embeddings
            search_results = self._perform_text_search(query, num_chunks)
        
        # Prepare retrieved documents
        retrieved_docs = []
        for result in search_results:
            original_idx = result["_index"]
            if original_idx < len(input_data):
                retrieved_doc = input_data[original_idx]
                score_field = "_distance" if "_distance" in result else "_score"
                retrieved_docs.append({
                    **retrieved_doc,
                    score_field: result.get(score_field, None)
                })
        
        # Create output
        results = []
        for doc in input_data:
            results.append({
                **doc,
                output_key: retrieved_docs
            })
        
        return results, total_cost
    
    def _perform_text_search(self, query: str, limit: int) -> List[dict]:
        """Perform text-based search on the table."""
        # LanceDB supports SQL-like queries for text search
        where_clause = " OR ".join([
            f"text LIKE '%{term}%'" 
            for term in query.split() 
            if term.strip()
        ])
        
        if where_clause:
            results = (
                self._table.search()
                .where(where_clause)
                .limit(limit)
                .to_list()
            )
        else:
            results = []
        
        return results
