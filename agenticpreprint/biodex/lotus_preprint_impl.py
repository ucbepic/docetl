from typing import Any
import pandas as pd
import faulthandler
import os
import pickle

import lotus
from dotenv import load_dotenv

import numpy as np
import time

start = time.time()

@pd.api.extensions.register_dataframe_accessor("sem_index_ss")
class SemIndexSimpleDataframe:
    """DataFrame accessor for creating simple semantic index."""
    
    def __init__(self, pandas_obj: Any):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def __call__(self, column: str, index_dir: str) -> pd.DataFrame:
        """
        Create embeddings for a column and save them to a pickle file.
        If embeddings already exist in the pickle file, load them instead of recomputing.
        
        Args:
            column (str): Column to create embeddings for
            index_dir (str): Directory to save embeddings pickle file
        """
        rm = lotus.settings.rm
        if not isinstance(rm, lotus.models.RM):
            raise ValueError("The retrieval model must be an instance of RM")

        os.makedirs(index_dir, exist_ok=True)
        pickle_path = os.path.join(index_dir, "embeddings.pkl")
        
        # Try to load existing embeddings
        if os.path.exists(pickle_path):
            print(f"Loading cached embeddings for {column}")
            with open(pickle_path, "rb") as f:
                embeddings = pickle.load(f)
        else:
            print(f"Computing new embeddings for {column}")
            texts = self._obj[column].tolist()
            embeddings = rm._embed(texts)  # Already returns numpy array
            # Save embeddings to pickle file
            with open(pickle_path, "wb") as f:
                pickle.dump(embeddings, f)
            
        # Store the index directory in DataFrame attributes
        if "index_dirs_ss" not in self._obj.attrs:
            self._obj.attrs["index_dirs_ss"] = {}
        self._obj.attrs["index_dirs_ss"][column] = index_dir
        
        return self._obj

@pd.api.extensions.register_dataframe_accessor("sem_sim_join_ss")
class SemSimJoinSimpleDataframe:
    """DataFrame accessor for semantic similarity join using simple numpy operations."""

    def __init__(self, pandas_obj: Any):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def __call__(
        self,
        other: pd.DataFrame,
        left_on: str,
        right_on: str,
        K: int,
        lsuffix: str = "",
        rsuffix: str = "",
        score_suffix: str = "",
    ) -> pd.DataFrame:
        """
        Perform semantic similarity join using numpy operations.
        """
        if isinstance(other, pd.Series):
            if other.name is None:
                raise ValueError("Other Series must have a name")
            other = pd.DataFrame({other.name: other})

        rm = lotus.settings.rm
        if not isinstance(rm, lotus.models.RM):
            raise ValueError(
                "The retrieval model must be an instance of RM"
            )

        # Try to load pre-computed embeddings
        left_embeddings = None
        right_embeddings = None
        
        # Load left embeddings from pickle if available
        if "index_dirs_ss" in self._obj.attrs and left_on in self._obj.attrs["index_dirs_ss"]:
            left_pickle = os.path.join(self._obj.attrs["index_dirs_ss"][left_on], "embeddings.pkl")
            if os.path.exists(left_pickle):
                print(f"Loading cached embeddings for {left_on}")
                with open(left_pickle, "rb") as f:
                    left_embeddings = pickle.load(f)
                    
        # Load right embeddings from pickle if available
        if "index_dirs_ss" in other.attrs and right_on in other.attrs["index_dirs_ss"]:
            right_pickle = os.path.join(other.attrs["index_dirs_ss"][right_on], "embeddings.pkl")
            if os.path.exists(right_pickle):
                print(f"Loading cached embeddings for {right_on}")
                with open(right_pickle, "rb") as f:
                    right_embeddings = pickle.load(f)

        # Only compute embeddings if not found in pickle files
        if left_embeddings is None:
            print(f"Computing new embeddings for {left_on}")
            left_texts = self._obj[left_on].tolist()
            left_embeddings = rm._embed(left_texts)  # Already returns numpy array
            
        if right_embeddings is None:
            print(f"Computing new embeddings for {right_on}")
            right_texts = other[right_on].tolist()
            right_embeddings = rm._embed(right_texts)  # Already returns numpy array

        # Reshape embeddings if needed
        left_embeddings = left_embeddings.reshape(-1, 1) if len(left_embeddings.shape) == 1 else left_embeddings
        right_embeddings = right_embeddings.reshape(-1, 1) if len(right_embeddings.shape) == 1 else right_embeddings

        # Compute cosine similarity matrix
        similarity_matrix = np.dot(left_embeddings, right_embeddings.T)
        
        # Get top K indices and scores for each query
        K = min(K, len(right_embeddings))  # Ensure K isn't larger than number of candidates
        top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :K]
        top_k_scores = np.take_along_axis(similarity_matrix, top_k_indices, axis=1)

        # Create join results
        join_results = []
        for q_idx, (res_ids, scores) in enumerate(zip(top_k_indices, top_k_scores)):
            for res_id, score in zip(res_ids, scores):
                join_results.append((self._obj.index[q_idx], other.index[res_id], score))

        # Create joined dataframe
        df1 = self._obj.copy()
        df2 = other.copy()
        df1["_left_id"] = df1.index
        df2["_right_id"] = df2.index
        temp_df = pd.DataFrame(join_results, columns=["_left_id", "_right_id", "_scores" + score_suffix])
        
        joined_df = (
            df1.join(
                temp_df.set_index("_left_id"),
                how="right",
                on="_left_id",
            )
            .join(
                df2.set_index("_right_id"),
                how="left",
                on="_right_id",
                lsuffix=lsuffix,
                rsuffix=rsuffix,
            )
            .drop(columns=["_left_id", "_right_id"])
        )

        return joined_df

load_dotenv()

faulthandler.enable()

LM_CALL_BUDGET = 12750
original_LM_CALL_BUDGET = LM_CALL_BUDGET

lm = lotus.models.LM(
    model="gpt-4o-mini",
    max_batch_size=250
)
lotus.settings.configure(
    lm=lm,
    rm = lotus.models.LiteLLMRM(model="text-embedding-3-small", max_batch_size=1000)
)

articles = pd.read_json("agenticpreprint/biodex/biodex_sample.json")
def truncate_to_n_words(text: str, n: int) -> str:
    words = text.replace("\n", " ").split()
    return " ".join(words[:n])

# If we don't do this, we will get an error about the text exceeding context length for the sem_index_ss step
articles["fulltext_processed"] = articles["fulltext_processed"].apply(lambda x: truncate_to_n_words(x, 2700))

reaction_labels_df = pd.read_json("agenticpreprint/biodex/biodex_terms.json")


articles = articles.sem_index_ss("fulltext_processed", "agenticpreprint/biodex/fulltext_index_dir")
reaction_labels_df = reaction_labels_df.sem_index_ss("reaction", "agenticpreprint/biodex/reaction_index_dir")

print(f"There are {len(articles)} articles and {len(reaction_labels_df)} reaction labels")


# Map-search-filter
articles = articles.sem_map("The article {fulltext_processed} indicates that the patient is experiencing the reaction:", suffix="article_reaction")

LM_CALL_BUDGET -= (2 * len(articles))

k_for_join = LM_CALL_BUDGET // len(articles)
print(f"Doing a sem sim join with K={k_for_join}")
sem_joined_df = articles.sem_sim_join_ss(reaction_labels_df, left_on="article_reaction", right_on="reaction", K=k_for_join)

# Apply filter to each group to throw out article_reaction, reaction pairs that are not equivalent
print(f"Applying filter to dataframe of size {len(sem_joined_df)}")
sem_joined_df = sem_joined_df.sem_filter("The article {article_reaction} indicates that the patient is experiencing the {reaction}")

# Save to csv
print(sem_joined_df.columns)
print(sem_joined_df.head())
print(f"Found {len(sem_joined_df)} results for {len(articles)} articles and {len(reaction_labels_df)} reaction labels")
sem_joined_df.to_csv("agenticpreprint/biodex/results/lotus_output.csv", index=False)


cost = lm.stats.total_usage.total_cost
print(f"Total cost: {cost}")

end = time.time()
print(f"Total time: {end - start}")