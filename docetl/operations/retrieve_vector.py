from typing import Any, List, Optional, Union
import os
from pathlib import Path

from pydantic import Field, field_validator

from docetl.operations.base import BaseOperation
from docetl.operations.clustering_utils import get_embeddings_for_clustering


class RetrieveVectorOperation(BaseOperation):
    """
    A vector retrieval operation that uses LanceDB to find similar documents.
    
    This operation embeds documents and queries them against a vector database
    to retrieve the most similar items based on embedding distance.
    """

    class schema(BaseOperation.schema):
        type: str = "retrieve_vector"
        query: str = Field(
            ...,
            description="The query text to search for similar documents"
        )
        embedding_model: str = Field(
            default="text-embedding-3-small",
            description="The embedding model to use"
        )
        embedding_keys: List[str] = Field(
            ...,
            description="Keys from documents to embed and search"
        )
        num_chunks: int = Field(
            default=10,
            ge=1,
            description="Number of top similar chunks to retrieve (like top-k)"
        )
        table_name: str = Field(
            default="docetl_vectors",
            description="Name of the LanceDB table to use"
        )
        db_path: str = Field(
            default="./.lancedb",
            description="Path to the LanceDB database"
        )
        persist: bool = Field(
            default=False,
            description="Whether to persist the vector database between runs"
        )
        distance_metric: str = Field(
            default="cosine",
            description="Distance metric to use for similarity search"
        )
        output_key: str = Field(
            default="_retrieved",
            description="Key to store retrieved documents in the output"
        )
        
        @field_validator("embedding_keys")
        def validate_embedding_keys(cls, v):
            if not v:
                raise ValueError("embedding_keys must contain at least one key")
            return v

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._db = None
        self._table = None
        
    def _initialize_db(self):
        """Initialize LanceDB connection and table."""
        try:
            import lancedb
        except ImportError:
            raise ImportError(
                "LanceDB is required for retrieve_vector operation. "
                "Install it with: pip install lancedb"
            )
        
        # Create database directory if it doesn't exist
        db_path = Path(self.config["db_path"])
        db_path.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        self._db = lancedb.connect(str(db_path))
        
    def _prepare_text_for_embedding(self, doc: dict) -> str:
        """Prepare document text for embedding by concatenating specified keys."""
        texts = []
        for key in self.config["embedding_keys"]:
            if key in doc:
                value = doc[key]
                if isinstance(value, str):
                    texts.append(value)
                elif value is not None:
                    texts.append(str(value))
        return " ".join(texts)

    def execute(
        self, input_data: List[dict], is_build: bool = False
    ) -> tuple[List[dict], float]:
        """
        Execute the vector retrieval operation.
        
        Args:
            input_data: List of documents to index and search
            is_build: Whether this is a build phase execution
            
        Returns:
            Tuple of (results, cost) where results contain retrieved documents
        """
        if not input_data:
            return [], 0.0
            
        try:
            import lancedb
            from lancedb.pydantic import LanceModel, Vector
        except ImportError:
            raise ImportError(
                "LanceDB is required for retrieve_vector operation. "
                "Install it with: pip install lancedb"
            )
        
        self._initialize_db()
        
        total_cost = 0.0
        
        # Prepare documents for embedding
        docs_to_embed = []
        for i, doc in enumerate(input_data):
            text = self._prepare_text_for_embedding(doc)
            if text:  # Only include documents with content
                docs_to_embed.append({
                    "_index": i,
                    "text": text,
                    "document": doc
                })
        
        if not docs_to_embed:
            return [], 0.0
        
        # Get embeddings for all documents
        texts = [d["text"] for d in docs_to_embed]
        embeddings, embedding_cost = get_embeddings_for_clustering(
            [{"text": t} for t in texts],
            {"embedding_keys": ["text"], "embedding_model": self.config.get("embedding_model")},
            self.runner.api
        )
        total_cost += embedding_cost
        
        # Prepare data for LanceDB
        vector_dim = len(embeddings[0]) if embeddings else 0
        
        # Create dynamic model for LanceDB
        class DocumentVector(LanceModel):
            text: str
            vector: Vector(vector_dim)
            _index: int
            
        # Add document data to records
        records = []
        for i, (doc_info, embedding) in enumerate(zip(docs_to_embed, embeddings)):
            record = {
                "text": doc_info["text"],
                "vector": embedding,
                "_index": doc_info["_index"]
            }
            records.append(record)
        
        # Create or overwrite table
        table_name = self.config["table_name"]
        if table_name in self._db.table_names():
            if not self.config["persist"]:
                # Drop existing table if not persisting
                self._db.drop_table(table_name)
                self._table = self._db.create_table(
                    table_name, 
                    data=records,
                    schema=DocumentVector
                )
            else:
                # Add to existing table
                self._table = self._db.open_table(table_name)
                self._table.add(records)
        else:
            # Create new table
            self._table = self._db.create_table(
                table_name,
                data=records,
                schema=DocumentVector
            )
        
        # Get embedding for the query
        query_text = self.config["query"]
        query_embeddings, query_cost = get_embeddings_for_clustering(
            [{"text": query_text}],
            {"embedding_keys": ["text"], "embedding_model": self.config.get("embedding_model")},
            self.runner.api
        )
        total_cost += query_cost
        
        if not query_embeddings:
            # Return empty results if query embedding fails
            return [{**doc, self.config["output_key"]: []} for doc in input_data], total_cost
            
        query_vector = query_embeddings[0]
        
        # Search for similar documents
        num_chunks = self.config["num_chunks"]
        search_results = (
            self._table.search(query_vector)
            .limit(num_chunks)
            .to_list()
        )
        
        # Prepare retrieved documents
        retrieved_docs = []
        for result in search_results:
            # Get the original document
            original_idx = result["_index"]
            if original_idx < len(input_data):
                retrieved_doc = input_data[original_idx]
                retrieved_docs.append({
                    **retrieved_doc,
                    "_distance": result.get("_distance", None)
                })
        
        # Create output - each input document gets the same retrieved results
        output_key = self.config["output_key"]
        results = []
        for doc in input_data:
            results.append({
                **doc,
                output_key: retrieved_docs
            })
        
        return results, total_cost