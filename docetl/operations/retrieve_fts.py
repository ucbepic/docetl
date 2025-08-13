from typing import Any, List, Optional, Union
import os
from pathlib import Path

from pydantic import Field, field_validator

from docetl.operations.base import BaseOperation
from docetl.operations.clustering_utils import get_embeddings_for_clustering


class RetrieveFTSOperation(BaseOperation):
    """
    A full-text search retrieval operation that uses LanceDB with embeddings for semantic search.
    
    This operation creates a searchable index of documents and retrieves the most relevant
    items based on semantic similarity to a query using full-text search capabilities.
    """

    class schema(BaseOperation.schema):
        type: str = "retrieve_fts"
        query: str = Field(
            ...,
            description="The search query text"
        )
        embedding_model: str = Field(
            default="text-embedding-3-small",
            description="The embedding model to use for semantic search"
        )
        embedding_keys: List[str] = Field(
            ...,
            description="Keys from documents to embed and search"
        )
        num_chunks: int = Field(
            default=10,
            ge=1,
            description="Number of top results to retrieve (like top-k)"
        )
        table_name: str = Field(
            default="docetl_fts",
            description="Name of the LanceDB table to use"
        )
        db_path: str = Field(
            default="./.lancedb",
            description="Path to the LanceDB database"
        )
        persist: bool = Field(
            default=False,
            description="Whether to persist the database between runs"
        )
        rerank: bool = Field(
            default=True,
            description="Whether to rerank results using embeddings for better relevance"
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
        """Initialize LanceDB connection."""
        try:
            import lancedb
        except ImportError:
            raise ImportError(
                "LanceDB is required for retrieve_fts operation. "
                "Install it with: pip install lancedb"
            )
        
        # Create database directory if it doesn't exist
        db_path = Path(self.config["db_path"])
        db_path.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        self._db = lancedb.connect(str(db_path))
        
    def _prepare_text_for_search(self, doc: dict) -> str:
        """Prepare document text for indexing by concatenating specified keys."""
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
        Execute the full-text search retrieval operation.
        
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
            from lancedb.embeddings import get_registry
        except ImportError:
            raise ImportError(
                "LanceDB is required for retrieve_fts operation. "
                "Install it with: pip install lancedb"
            )
        
        self._initialize_db()
        
        total_cost = 0.0
        
        # Prepare documents for indexing
        docs_to_index = []
        for i, doc in enumerate(input_data):
            text = self._prepare_text_for_search(doc)
            if text:  # Only include documents with content
                docs_to_index.append({
                    "_index": i,
                    "text": text,
                    "document": doc
                })
        
        if not docs_to_index:
            return [], 0.0
        
        # Get embeddings for all documents if reranking is enabled
        embeddings = None
        if self.config["rerank"]:
            texts = [d["text"] for d in docs_to_index]
            embeddings, embedding_cost = get_embeddings_for_clustering(
                [{"text": t} for t in texts],
                {"embedding_keys": ["text"], "embedding_model": self.config.get("embedding_model")},
                self.runner.api
            )
            total_cost += embedding_cost
        
        # Prepare data for LanceDB
        if embeddings:
            vector_dim = len(embeddings[0])
            
            # Create dynamic model for LanceDB with embeddings
            class DocumentWithVector(LanceModel):
                text: str
                vector: Vector(vector_dim)
                _index: int
                
            # Add document data to records
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
            # Create model without embeddings for pure text search
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
        table_name = self.config["table_name"]
        if table_name in self._db.table_names():
            if not self.config["persist"]:
                # Drop existing table if not persisting
                self._db.drop_table(table_name)
                self._table = self._db.create_table(
                    table_name, 
                    data=records,
                    schema=schema
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
                schema=schema
            )
        
        # Perform search
        query_text = self.config["query"]
        num_chunks = self.config["num_chunks"]
        
        if self.config["rerank"] and embeddings:
            # Get embedding for the query
            query_embeddings, query_cost = get_embeddings_for_clustering(
                [{"text": query_text}],
                {"embedding_keys": ["text"], "embedding_model": self.config.get("embedding_model")},
                self.runner.api
            )
            total_cost += query_cost
            
            if query_embeddings:
                query_vector = query_embeddings[0]
                
                # Perform vector search with the query
                search_results = (
                    self._table.search(query_vector)
                    .limit(num_chunks)
                    .to_list()
                )
            else:
                # Fallback to text search if embedding fails
                search_results = self._perform_text_search(query_text, num_chunks)
        else:
            # Pure text search without embeddings
            search_results = self._perform_text_search(query_text, num_chunks)
        
        # Prepare retrieved documents
        retrieved_docs = []
        for result in search_results:
            # Get the original document
            original_idx = result["_index"]
            if original_idx < len(input_data):
                retrieved_doc = input_data[original_idx]
                score_field = "_distance" if "_distance" in result else "_score"
                retrieved_docs.append({
                    **retrieved_doc,
                    score_field: result.get(score_field, None)
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
    
    def _perform_text_search(self, query: str, limit: int) -> List[dict]:
        """
        Perform text-based search on the table.
        
        This is a fallback method when embeddings are not available or disabled.
        It uses LanceDB's built-in search capabilities.
        """
        # LanceDB supports SQL-like queries for text search
        # We'll search for documents where the text contains the query terms
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
            # If no valid search terms, return empty results
            results = []
        
        return results