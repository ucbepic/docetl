"""
This script is used to try out a block nested loop join for semantic joins.

It does the following:
1. Gets a train set of 10 documents & their matches
2. Tries to find the "correct" block size for the right dataset, by varying the block size to see what is the biggest block size with 90% accuracy
3. Tries to find the "correct" block size for the left dataset, by increasing the block size until accuracy suffers
4. Runs the join with the best block sizes for the entire dataset
"""
import json
from typing import List, Dict, Any
import pandas as pd
from litellm import completion
import litellm
from litellm.caching import Cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np
import pickle
load_dotenv()

litellm.cache = Cache(type="disk")

def load_dataset():
    labels_path = "preprint_workloads/biodex/biodex_ground_truth.json"
    right_dataset_path = "preprint_workloads/biodex/biodex_terms.json"
    left_dataset_path = "preprint_workloads/biodex/biodex_sample.json"
    
    labels = pd.read_json(labels_path)
    right_dataset = pd.read_json(right_dataset_path)
    left_dataset = pd.read_json(left_dataset_path)
    
    # Load embeddings
    article_embeddings, reaction_embeddings = load_embeddings()
    
    # Set the embeddings as columns in the datasets
    left_dataset["embedding"] = [embedding for embedding in article_embeddings]
    right_dataset["embedding"] = [embedding for embedding in reaction_embeddings]
    
    ground_truth = left_dataset.merge(labels, on="id")
    
    # Sample 10 documents from the ground truth
    train_left = ground_truth.sample(n=10, random_state=42)
    
    # Get the rest of the documents as test set
    test_left = ground_truth[~ground_truth.index.isin(train_left.index)]
    
    # Print dataset sizes
    print(f"Train left dataset size: {train_left.shape}")
    print(f"Test left dataset size: {test_left.shape}")
    
    return right_dataset, train_left, test_left


def find_embedding_similarity_threshold(left_dataset, right_dataset, threshold=0.9):
    # Find the embedding similarity threshold that will permit 90% recall
    # First get the embeddings for the ground_truth
    document_embeddings = left_dataset["embedding"].tolist()
    ground_truth_reactions = left_dataset["ground_truth_reactions"].tolist()
    
    right_dataset_embeddings = {}
    for _, row in right_dataset.iterrows():
        right_dataset_embeddings[row["reaction"]] = row["embedding"]
    
    
    # Get the embeddings for each ground truth reaction
    pairs = []
    for document_embedding, ground_truth_reaction in zip(document_embeddings, ground_truth_reactions):
        for reaction in ground_truth_reaction:
            pairs.append((document_embedding, right_dataset_embeddings[reaction]))
    
    # Calculate the cosine similarity between the pairs
    # Convert pairs list of tuples into numpy arrays
    doc_embeddings = np.array([pair[0] for pair in pairs])
    reaction_embeddings = np.array([pair[1] for pair in pairs])
    similarities = []
    for doc_embedding, reaction_embedding in zip(doc_embeddings, reaction_embeddings):
        similarity = np.dot(doc_embedding, reaction_embedding) / (np.linalg.norm(doc_embedding) * np.linalg.norm(reaction_embedding))
        similarities.append(similarity)
    
    # Find the value of the threshold that will permit 90% recall
    # First sort the similarities
    similarities.sort()
    # Find the index of the 90th percentile
    threshold_index = int(0.9 * len(similarities))
    threshold = similarities[threshold_index]
    return threshold

def load_embeddings():
    article_embeddings_path = "paper_workloads/biodex/fulltext_index_dir/embeddings.pkl"
    reaction_embeddings_path = "paper_workloads/biodex/reaction_index_dir/embeddings.pkl"
    
    # Load embeddings
    with open(article_embeddings_path, "rb") as f:
        article_embeddings = pickle.load(f)
    with open(reaction_embeddings_path, "rb") as f:
        reaction_embeddings = pickle.load(f)
    
    return article_embeddings, reaction_embeddings
   
def sem_join_llm_call(left_id_key: str, left_docs: List[Dict[str, Any]], 
                     right_id_key: str, right_docs: List[Dict[str, Any]]) -> List[tuple[str, str]]:
    """
    Uses LLM to find matching pairs between medical articles and reaction labels.
    
    Args:
        left_id_key: ID key for the left documents (medical articles)
        left_docs: List of medical article documents
        right_id_key: ID key for the right documents (reaction labels)
        right_docs: List of reaction label documents
        
    Returns:
        List of tuples containing matching (left_id, right_id) pairs
    """
    
    # Get rid of the embedding key from the docs
    left_docs_copy = [doc.copy() for doc in left_docs]
    right_docs_copy = [doc.copy() for doc in right_docs]
    for doc in left_docs_copy:
        doc.pop("embedding", None)
    for doc in right_docs_copy:
        doc.pop("embedding", None)
    
    system_prompt = """You are an expert in medical literature and pharmacology. Your task is to analyze medical articles and match them with the most relevant medical reaction labels.

A medical reaction label describes a specific medical condition, side effect, or outcome. Your goal is to identify which medical articles discuss or are most relevant to each reaction label.

Only return matches when there is a clear and strong connection between the article's content and the reaction label. Consider:
- Direct mentions or discussions of the reaction
- Related symptoms or conditions
- Clinical relevance and context

Output the matches as a list of ID pairs."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""Here are the medical articles and reaction labels to match:

Medical Articles (ID: {left_id_key}):
{json.dumps(left_docs_copy, indent=2)}

Reaction Labels (ID: {right_id_key}):
{json.dumps([item[right_id_key] for item in right_docs_copy], indent=2)}

For each medical article, identify which reaction label(s) are most relevant. Return the matches as a list of ID pairs.

Output the matches in this exact JSON format:
{{"matches": [
    {{"left_id": "article_id", "right_id": "label_id"}},
    ...
]}}"""}
    ]

    try:
        response = completion(
            model="azure/gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
            caching=True,
            num_retries=3
        )

        result = json.loads(response.choices[0].message.content)
        matches = [(match["left_id"], match["right_id"]) for match in result["matches"]]
        return matches
    except Exception as e:
        print(f"Error in LLM call or parsing response: {e}")
        return []

def rerank_matches_llm_call(matches: List[str], left_doc: str) -> List[str]:
    # Deduplicate matches
    deduped_matches = list(set(matches))
    
    """Reranks the matches using an LLM."""
    system_prompt = """You are an expert in medical literature and pharmacology. Your task is to analyze a medical article and rerank a list of medical reaction labels based on how likely they are to be present in the article.

For each reaction label, carefully consider:
- Direct mentions or discussions of the reaction
- Related symptoms or conditions described
- Clinical relevance and context
- Strength of evidence in the text

Rank the reactions from most likely to least likely to be present in the article."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""Here is the medical article:
{left_doc}

Here are the reaction labels to rerank:
{json.dumps(deduped_matches, indent=2)}

Return the reranked list of reactions in order from most likely to least likely to be present in the article.

Output the reranked list in this exact JSON format:
{{"reranked_matches": [
    "reaction1",
    "reaction2",
    ...
]}}"""}
    ]

    try:
        response = completion(
            model="azure/gpt-4o-mini", 
            messages=messages,
            response_format={"type": "json_object"},
            caching=True,
            num_retries=3
        )
        
        result = json.loads(response.choices[0].message.content)
        return result["reranked_matches"]
    except Exception as e:
        print(f"Error in reranking LLM call: {e}")
        return matches


def process_single_block(left_doc: Dict[str, Any], right_block: List[Dict[str, Any]], left_id_key: str, right_id_key: str) -> List[str]:
    """Helper function to process a single block of right documents for a left document."""
    matches = sem_join_llm_call(left_id_key, [left_doc], right_id_key, right_block)
    return [reaction for _, reaction in matches]

def find_right_block_size(left_dataset, right_dataset, embedding_similarity_threshold):
    # Find the block size that gives 90% accuracy
    left_id_key = "id"
    right_id_key = "reaction"
    left_docs = left_dataset[["id", "fulltext_processed", "embedding"]].to_dict(orient="records")
    right_docs = right_dataset[["reaction", "embedding"]].to_dict(orient="records")
    
    best_rp = 0
    best_block_size = float('-inf')
    block_results = {}
    block_rp_at_50 = {}
    block_rp_at_25 = {}
    
    # Have a loop for 8, 16, 32, 64 llm calls for a left doc
    for target_num_calls in [8, 16, 32, 64]:
        # Create a dictionary to store matches
        matches = {}    
        
        # Create right block size based on target_num_calls
        right_block_size = len(right_docs) // target_num_calls
        
        block_results[right_block_size] = {}
        block_rp_at_50[right_block_size] = {}
        block_rp_at_25[right_block_size] = {}
        
        total_recall = 0
        num_docs = len(left_docs)
        
        # For each left doc
        for left_doc in left_docs:
            matches[left_doc["id"]] = []
            
            # Create blocks of right docs
            # Subset right_docs to be whichever reactions are in the left doc as well as the ones that are similar to the left doc
            # First compute the embedding similarity
            left_embedding = left_doc["embedding"]
            right_embeddings = [reaction["embedding"] for reaction in right_docs]
            similarities = [np.dot(left_embedding, right_embedding) / (np.linalg.norm(left_embedding) * np.linalg.norm(right_embedding)) for right_embedding in right_embeddings]
            # Then subset the right docs based on the embedding similarity
            right_docs_subset = [reaction for reaction, similarity in zip(right_docs, similarities) if similarity > embedding_similarity_threshold]
            
            right_docs_subset += [reaction for reaction in right_docs if reaction not in right_docs_subset and any([word in left_doc["fulltext_processed"].lower() for word in reaction[right_id_key].lower().split()])]
            
            if len(right_docs_subset) == 0:
                right_docs_subset = right_docs
            else:
                print(f"Subsetted right docs for left doc {left_doc['id']} to {len(right_docs_subset)} reactions")
            
            right_blocks = [right_docs_subset[i:i+right_block_size] for i in range(0, len(right_docs_subset), right_block_size)]
            
            # Create a ThreadPool to process blocks in parallel
            with ThreadPoolExecutor(max_workers=len(right_blocks)) as executor:
                # Submit all blocks for processing
                future_to_block = {
                    executor.submit(process_single_block, left_doc, right_block, left_id_key, right_id_key): right_block 
                    for right_block in right_blocks
                }
                
                # Collect results as they complete
                for future in tqdm(as_completed(future_to_block), total=len(future_to_block), desc=f"Processing right blocks for left doc {left_doc['id']}"):
                    block_matches = future.result()
                    matches[left_doc["id"]].extend(block_matches)
                    
            # Do an LLM call to rerank the matches
            reranked_matches = rerank_matches_llm_call(matches[left_doc["id"]], left_doc["fulltext_processed"])
            matches[left_doc["id"]] = reranked_matches
            
            # Check accuracy 
            # Find ground truth in left_dataset for this id
            ground_truth = left_dataset[left_dataset["id"] == left_doc["id"]]["ground_truth_reactions"].values[0]
            num_ground_truth = len(ground_truth)
            ground_truth = [reaction.lower() for reaction in ground_truth]
            # Deduplicate ground truth
            ground_truth = list(set(ground_truth))
            # Deduplicate matches
            matches[left_doc["id"]] = list(set(matches[left_doc["id"]]))
            
            
            # Check if the matches are in the ground truth
            recall = sum(1 for match in matches[left_doc["id"]] if match.lower() in ground_truth) / num_ground_truth
            total_recall += recall
            
            # Print predictions and ground truth for comparison
            print("\nPredictions vs Ground Truth:")
            print("-" * 50)
            print("Predictions:")
            print(", ".join(matches[left_doc["id"]]))
            print("\nGround Truth:")
            print(ground_truth)
            print(f"Recall: {recall}")
            print("-" * 50)
            
            block_rp_at_50[right_block_size][left_doc["id"]] = calculate_rp_at_k(matches[left_doc["id"]], ground_truth, 50)
            block_rp_at_25[right_block_size][left_doc["id"]] = calculate_rp_at_k(matches[left_doc["id"]], ground_truth, 25)
        
        avg_recall = total_recall / num_docs
        print(f"\nAverage recall for {target_num_calls} calls: {avg_recall}")
        
        if np.mean(list(block_rp_at_50[right_block_size].values())) > best_rp:
            best_rp = np.mean(list(block_rp_at_50[right_block_size].values()))
            best_block_size = right_block_size
        
        print(f"Best RP@50 found so far: {best_rp} for block size: {best_block_size}")
        
    # Print average RP@50 for each block size
    print("\nAverage RP for each block size:")
    for block_size in block_rp_at_50:
        avg_rp_at_50 = sum(block_rp_at_50[block_size].values()) / len(block_rp_at_50[block_size])
        avg_rp_at_25 = sum(block_rp_at_25[block_size].values()) / len(block_rp_at_25[block_size])
        print(f"Block size: {block_size}, Average RP@50: {avg_rp_at_50}, Average RP@25: {avg_rp_at_25}")
    
    return best_block_size

def calculate_rp_at_k(extracted: List[str], ground_truth: List[str], k: int) -> float:
    ground_truth = [gt.lower() for gt in ground_truth]
    relevant = sum(1 for item in extracted[:k] if item.lower() in ground_truth)
    return relevant / min(k, len(ground_truth))

def calculate_recall(extracted: List[str], ground_truth: List[str]) -> float:
    ground_truth = [gt.lower() for gt in ground_truth]
    relevant = sum(1 for item in extracted if item.lower() in ground_truth)
    return relevant / len(ground_truth)

def perform_join(left_dataset, right_dataset, right_block_size, left_id_key, right_id_key, embedding_similarity_threshold):
    """
    Performs a block nested loop join between left and right datasets using parallel processing.
    
    Args:
        left_dataset: DataFrame containing the left dataset
        right_dataset: DataFrame containing the right dataset
        left_block_size: Size of blocks for left dataset
        right_block_size: Size of blocks for right dataset
        left_id_key: Key for left dataset IDs
        right_id_key: Key for right dataset IDs
        
    Returns:
        List of dictionaries containing matches in format [{left_id_key: ..., right_id_key: ...}, ...]
    """
    # Convert datasets to records for easier processing
    left_docs = left_dataset[[left_id_key, "fulltext_processed", "embedding"]].to_dict(orient="records")
    right_docs = right_dataset[[right_id_key, "embedding"]].to_dict(orient="records")
    
    all_matches = []
    
    def process_left_block(left_doc):
        # Subset right docs based on text similarity and embedding similarity
        # First compute the embedding similarity
        left_embedding = left_doc["embedding"]
        right_embeddings = [reaction["embedding"] for reaction in right_docs]
        similarities = [np.dot(left_embedding, right_embedding) / (np.linalg.norm(left_embedding) * np.linalg.norm(right_embedding)) for right_embedding in right_embeddings]
        # Then subset the right docs based on the embedding similarity
        right_docs_subset = [reaction for reaction, similarity in zip(right_docs, similarities) if similarity > embedding_similarity_threshold]
        
        right_docs_subset += [
            reaction for reaction in right_docs 
            if reaction not in right_docs_subset and any([word in left_doc["fulltext_processed"].lower() 
                    for word in reaction[right_id_key].lower().split()])
        ]
        
        if len(right_docs_subset) == 0:
            right_docs_subset = right_docs
        
        # Create right blocks
        right_blocks = [right_docs_subset[i:i+right_block_size] 
                        for i in range(0, len(right_docs_subset), right_block_size)]
        
        doc_matches = []
        # Process each right block
        with ThreadPoolExecutor(max_workers=len(right_blocks)) as executor:
            future_to_block = {
                executor.submit(process_single_block, left_doc, right_block, left_id_key, right_id_key): right_block 
                for right_block in right_blocks
            }
            
            for future in as_completed(future_to_block):
                block_matches = future.result()
                doc_matches.extend(block_matches)
        
        # Rerank matches for this document
        reranked_matches = rerank_matches_llm_call(doc_matches, left_doc["fulltext_processed"])
        
        return [
            {left_id_key: left_doc[left_id_key], right_id_key: match}
            for match in reranked_matches
        ]
    
    # Process left blocks in parallel
    with ThreadPoolExecutor(max_workers=min(8, len(left_docs))) as executor:
        future_to_left_block = {
            executor.submit(process_left_block, left_doc): left_doc 
            for left_doc in left_docs
        }
        
        # Show progress with tqdm
        for future in tqdm(as_completed(future_to_left_block), 
                         total=len(future_to_left_block),
                         desc="Processing left blocks"):
            block_results = future.result()
            all_matches.extend(block_results)
    
    return all_matches

if __name__ == "__main__":
    # Load dataset
    right_dataset, train_dataset, test_dataset = load_dataset()
    
    # Find embedding similarity threshold that will permit 90% recall
    embedding_similarity_threshold = find_embedding_similarity_threshold(train_dataset, right_dataset)
    print(f"Embedding similarity threshold: {embedding_similarity_threshold}")
    
    # Find right block size
    # best_block_size = find_right_block_size(train_dataset, right_dataset, embedding_similarity_threshold)
    
    # print(f"Best block size: {best_block_size}")
    
    # Perform join
    left_id_key = "id"
    right_id_key = "reaction"
    matches = perform_join(test_dataset.head(50), right_dataset, 1519, left_id_key, right_id_key, embedding_similarity_threshold)

    # Compute RP@25 and RP@50
    ks = [25, 50]
    for k in ks:
        rps = []
        recalls = []
        # For each left_doc in test_dataset, find the ground truth in test_dataset
        for _, left_doc in test_dataset.head(50).iterrows():
            ground_truth = left_doc["ground_truth_reactions"]
            # Find the matches for this left_doc in matches
            matches_for_doc = [match[right_id_key] for match in matches if match[left_id_key] == left_doc[left_id_key]]
            
            # Compute RP@k for this left_doc
            rp_at_k = calculate_rp_at_k(matches_for_doc, ground_truth, k)
            rps.append(rp_at_k)
            
            # Compute recall for this left_doc
            recall = calculate_recall(matches_for_doc, ground_truth)
            recalls.append(recall)
        
        print(f"Average RP@{k}: {sum(rps) / len(rps)}")
        print(f"Average Recall@{k}: {sum(recalls) / len(recalls)}")