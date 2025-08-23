import argparse
import json
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial

import chromadb
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction

import palimpzest as pz
from palimpzest.constants import Model
from palimpzest.policy import MaxQuality, MaxQualityAtFixedCost

from dotenv import load_dotenv
load_dotenv()

# Add the evaluation utils to the path
from experiments.reasoning.evaluation.utils import get_evaluate_func

# Budget fractions to test
FRACS = [0.75, 0.5, 0.25, 0.1]

# BioDEX column definitions
biodex_entry_cols = [
    {"name": "pmid", "type": str, "desc": "The PubMed ID of the medical paper"},
    {"name": "title", "type": str, "desc": "The title of the medical paper"},
    {"name": "abstract", "type": str, "desc": "The abstract of the medical paper"},
    {"name": "fulltext", "type": str, "desc": "The full text of the medical paper, which contains information relevant for creating a drug safety report"},
]

biodex_eligible_reactions_cols = [
    {"name": "eligible_reactions", "type": str, "desc": "The list of all medical conditions that we are interested in. Only a subset of these will actually be in the report."},
]

# Load BioDEX reactions string (this will be loaded from file)
biodex_reactions_str = ""


def load_biodex_data(split="train"):
    """Load BioDEX data from our existing JSON files."""
    file_path = f"experiments/reasoning/data/{split}/biodex.json"
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def load_biodex_labels():
    """Load BioDEX reaction labels from text file."""
    labels_path = "experiments/reasoning/data/biodex_labels.txt"
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return labels


class BioDEXDataReader(pz.DataReader):
    def __init__(
        self,
        rp_at_k: int = 5,
        num_samples: int = 250,
        split: str = "test",
        shuffle: bool = False,
        seed: int = 42,
    ):
        super().__init__(biodex_entry_cols + biodex_eligible_reactions_cols)

        # Load from our JSON files instead of HuggingFace datasets
        data = load_biodex_data(split)
        
        # Sample texts if needed
        if len(data) > num_samples:
            np.random.seed(seed)
            if shuffle:
                indices = np.random.choice(len(data), num_samples, replace=False)
                self.dataset = [data[i] for i in indices]
            else:
                self.dataset = data[:num_samples]
        else:
            self.dataset = data

        self.rp_at_k = rp_at_k
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed
        self.split = split
        
        # Load eligible reactions and set global variable
        global biodex_reactions_str
        self.eligible_reactions_list = load_biodex_labels()
        biodex_reactions_str = "\n".join(self.eligible_reactions_list)

    def compute_label(self, entry: dict) -> dict:
        """Compute the label for a BioDEX report given its entry in the dataset."""
        reactions_lst = entry.get("ground_truth_reactions", [])
        if isinstance(reactions_lst, str):
            # If it's a comma-separated string, split it
            reactions_lst = [
                reaction.strip().lower().replace("'", "").replace("^", "")
                for reaction in reactions_lst.split(",")
            ]
        else:
            # If it's already a list, normalize it
            reactions_lst = [
                reaction.strip().lower().replace("'", "").replace("^", "")
                for reaction in reactions_lst
            ]
        
        label_dict = {
            "reactions": reactions_lst,
            "reaction_labels": reactions_lst,
            "ranked_reaction_labels": reactions_lst,
        }
        return label_dict

    @staticmethod
    def rank_precision_at_k(preds: list | None, targets: list, k: int):
        print(f"preds: {preds}")
        print(f"targets: {targets}")
        print(f"k: {k}")
        
        if preds is None:
            return 0.0

        try:
            # lower-case each list
            preds = [pred.strip().lower().replace("'", "").replace("^", "") for pred in preds]
            targets = set([target.strip().lower().replace("'", "").replace("^", "") for target in targets])

            # compute rank-precision at k
            rn = len(targets)
            denom = min(k, rn)
            total = 0.0
            for i in range(k):
                total += preds[i] in targets if i < len(preds) else 0.0

            return total / denom

        except Exception:
            os.makedirs("rp@k-errors", exist_ok=True)
            ts = time.time()
            with open(f"rp@k-errors/error-{ts}.txt", "w") as f:
                f.write(str(preds))
            return 0.0

    @staticmethod
    def term_recall(preds: list | None, targets: list):
        if preds is None:
            return 0.0

        try:
            # normalize terms in each list
            pred_terms = set([
                term.strip()
                for pred in preds
                for term in pred.lower().replace("'", "").replace("^", "").split(" ")
            ])
            target_terms = set([
                term.strip()
                for target in targets
                for term in target.lower().replace("'", "").replace("^", "").split(" ")
            ])

            # compute term recall and return
            intersect = pred_terms.intersection(target_terms)
            term_recall = len(intersect) / len(target_terms) if target_terms else 0.0

            return term_recall

        except Exception:
            os.makedirs("term-recall-eval-errors", exist_ok=True)
            ts = time.time()
            with open(f"term-recall-eval-errors/error-{ts}.txt", "w") as f:
                f.write(str(preds))
            return 0.0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # get entry
        entry = self.dataset[idx]

        # get input fields
        pmid = str(entry.get("id", entry.get("pmid", f"unknown_{idx}")))
        title = entry.get("title", "")
        abstract = entry.get("abstract", "")
        fulltext = entry.get("fulltext_processed", "")
        
        # Add the reactions that we are interested in
        eligible_reactions = biodex_reactions_str

        # create item with fields
        item = {"fields": {}, "labels": {}, "score_fn": {}}
        item["fields"]["pmid"] = pmid
        item["fields"]["title"] = title
        item["fields"]["abstract"] = abstract
        item["fields"]["fulltext"] = fulltext
        item["fields"]["eligible_reactions"] = eligible_reactions

        if self.split == "train":
            # add label info
            item["labels"] = self.compute_label(entry)

            # add scoring functions for list fields
            rank_precision_at_k = partial(BioDEXDataReader.rank_precision_at_k, k=self.rp_at_k)
            item["score_fn"]["reactions"] = BioDEXDataReader.term_recall
            item["score_fn"]["reaction_labels"] = BioDEXDataReader.term_recall
            item["score_fn"]["ranked_reaction_labels"] = rank_precision_at_k

        return item
    
    def get_label_df(self):
        """Get a dataframe with labels for evaluation."""
        final_label_dataset = []
        for entry in self.dataset:
            labels = self.compute_label(entry)
            row = {
                "pmid": str(entry.get("id", entry.get("pmid", "unknown"))),
                "reactions": labels["reactions"],
                "reaction_labels": labels["reaction_labels"],
                "ranked_reaction_labels": labels["ranked_reaction_labels"],
            }
            final_label_dataset.append(row)
        
        return pd.DataFrame(final_label_dataset)


def build_biodex_direct_query(dataset):
    """Build the direct BioDEX query (Shreya's plan)."""
    ds = pz.Dataset(dataset)
    
    # Define the direct reaction extraction schema
    cols = [
        {
            "name": "ranked_reaction_labels", 
            "type": list[str], 
            "desc": "The ranked list of medical conditions experienced by the patient. The most relevant label occurs first in the list."
        }
    ]
    
    
    ds = ds.sem_add_columns(cols, depends_on=["fulltext", "eligible_reactions"])
    
    return ds


def build_biodex_retrieval_query(dataset, index, search_func):
    """Build the retrieval-based BioDEX query (original template plan)."""
    ds = pz.Dataset(dataset)
    
    # Step 1: Extract reactions
    reactions_cols = [
        {
            "name": "reactions", 
            "type": list[str], 
            "desc": "The list of all medical conditions experienced by the patient as discussed in the report. Try to provide as many relevant medical conditions as possible."
        }
    ]
    
    reactions_desc = """You are an expert medical reviewer. Read the medical paper and identify every adverse drug reaction (ADR) or medical condition reported by patients. Extract all relevant medical conditions mentioned in the text, focusing on adverse reactions and patient symptoms."""
    
    ds = ds.sem_add_columns(reactions_cols, desc=reactions_desc, depends_on=["fulltext"])
    
    # Step 2: Use retrieval to find official labels
    reaction_labels_cols = [
        {
            "name": "reaction_labels", 
            "type": list[str], 
            "desc": "Official terms for medical conditions listed in `reactions`"
        }
    ]
    
    ds = ds.retrieve(
        index=index,
        search_func=search_func,
        search_attr="reactions",
        output_attrs=reaction_labels_cols,
    )
    
    # Step 3: Rank the labels
    ranked_labels_cols = [
        {
            "name": "ranked_reaction_labels", 
            "type": list[str], 
            "desc": "The ranked list of medical conditions experienced by the patient. The most relevant label occurs first in the list. Be sure to rank ALL of the inputs."
        }
    ]
    
    ranking_desc = """Given the official reaction labels, rank them in order of relevance and confidence based on the original medical text. The most relevant and well-supported reactions should appear first in the list."""
    
    ds = ds.sem_add_columns(ranked_labels_cols, desc=ranking_desc, depends_on=["title", "abstract", "fulltext", "reaction_labels"])
    
    return ds


def convert_predictions_to_biodex_format(pred_df, data_reader):
    """Convert Palimpzest predictions to the format expected by BioDEX evaluation."""
    results = []
    
    # Create a mapping from pmid to original data for ground truth
    original_data_map = {str(item.get("id", item.get("pmid", f"unknown_{i}"))): item 
                        for i, item in enumerate(data_reader.dataset)}
    
    for _, row in pred_df.iterrows():
        # Handle prediction values
        ranked_reaction_labels = row.get("ranked_reaction_labels", [])
        
        # Clean up the values
        if ranked_reaction_labels is None or (hasattr(ranked_reaction_labels, '__len__') and len(ranked_reaction_labels) == 0):
            ranked_reaction_labels = []
        elif pd.isna(ranked_reaction_labels).all() if hasattr(ranked_reaction_labels, '__len__') else pd.isna(ranked_reaction_labels):
            ranked_reaction_labels = []
        elif isinstance(ranked_reaction_labels, str):
            # Try to parse if it's a string representation of a list
            try:
                import ast
                ranked_reaction_labels = ast.literal_eval(ranked_reaction_labels)
            except:
                ranked_reaction_labels = []
        
        # Ensure it's a list
        if not isinstance(ranked_reaction_labels, list):
            ranked_reaction_labels = []
        
        # Get original data for ground truth
        pmid = str(row["pmid"])
        original_data = original_data_map.get(pmid, {})
        
        # Create the record in the format matching the evaluation
        record = {
            "id": pmid,
            "fulltext_processed": row["fulltext"],
            "ground_truth_reactions": original_data.get("ground_truth_reactions", []),
            "ranked_reaction_labels": ranked_reaction_labels,
            "possible_labels": data_reader.eligible_reactions_list
        }
        results.append(record)
    
    return results


def run_experiment(data_reader, val_data_reader, policy, models, 
                   pipeline_type="direct", sentinel_strategy="mab", k=10, j=3, 
                   sample_budget=100, seed=42, exp_name=None, index=None, search_func=None):
    """Run a single experiment with given policy and return results."""
    print(f"\nRunning {pipeline_type} pipeline experiment with policy: {policy}")
    
    # Build query based on pipeline type
    if pipeline_type == "direct":
        query = build_biodex_direct_query(data_reader)
    else:  # retrieval
        if index is None or search_func is None:
            raise ValueError("Retrieval pipeline requires index and search_func parameters")
        query = build_biodex_retrieval_query(data_reader, index, search_func)
    
    # Configure query processor
    config = pz.QueryProcessorConfig(
        policy=policy,
        verbose=False,
        val_datasource=val_data_reader,
        processing_strategy="sentinel",
        optimizer_strategy="pareto",
        sentinel_execution_strategy=sentinel_strategy,
        execution_strategy="parallel",
        max_workers=64,
        available_models=models,
        allow_bonded_query=True,
        allow_code_synth=False,
        allow_critic=True,
        allow_mixtures=True,
        allow_rag_reduction=True,
        progress=True,
    )
    
    # Execute the query
    data_record_collection = query.run(
        config=config,
        k=k,
        j=j,
        sample_budget=sample_budget,
        seed=seed,
        exp_name=exp_name if exp_name else f"biodex-pz-{pipeline_type}-{policy.__class__.__name__}",
        priors=None,
    )
    
    pred_df = data_record_collection.to_df()
    
    # Convert to BioDEX format
    results_list = convert_predictions_to_biodex_format(pred_df, data_reader)
    
    # Save results as JSON
    output_file = f"experiments/reasoning/othersystems/biodex/{exp_name}.json" if exp_name else f"experiments/reasoning/othersystems/biodex/pz_{pipeline_type}_temp.json"
    with open(output_file, 'w') as f:
        json.dump(results_list, f, indent=2)
    
    # Evaluate using our existing evaluation framework
    evaluate_func = get_evaluate_func("biodex")
    metrics = evaluate_func("palimpzest", output_file)
    
    # Get execution statistics for cost
    exec_stats = data_record_collection.execution_stats
    
    return {
        "avg_rp_at_5": metrics["avg_rp_at_5"],
        "avg_rp_at_10": metrics["avg_rp_at_10"],
        "avg_term_recall": metrics["avg_term_recall"],
        "total_documents": metrics["total_documents"],
        "optimization_time": exec_stats.optimization_time if exec_stats else 0,
        "optimization_cost": exec_stats.optimization_cost if exec_stats else 0,
        "plan_execution_time": exec_stats.plan_execution_time if exec_stats else 0,
        "plan_execution_cost": exec_stats.plan_execution_cost if exec_stats else 0,
        "total_execution_time": exec_stats.total_execution_time if exec_stats else 0,
        "total_execution_cost": exec_stats.total_execution_cost if exec_stats else 0,
        "output_file": output_file,
        "pipeline_type": pipeline_type,
        "sentinel_strategy": sentinel_strategy,
        "k": k,
        "j": j,
        "sample_budget": sample_budget,
    }


def main():
    parser = argparse.ArgumentParser(description="Run BioDEX experiments with budget analysis using Palimpzest")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-texts", type=int, default=250, help="Number of medical texts to process")
    parser.add_argument(
        "--sentinel-execution-strategy",
        default="mab",
        type=str,
        help="The engine to use. One of mab or random",
    )
    parser.add_argument(
        "--k",
        default=6,
        type=int,
        help="Number of columns to sample in Random Sampling or MAB sentinel execution",
    )
    parser.add_argument(
        "--j",
        default=4,
        type=int,
        help="Number of rows to sample in Random Sampling or MAB sentinel execution",
    )
    parser.add_argument(
        "--sample-budget",
        default=150,
        type=int,
        help="Total sample budget in Random Sampling or MAB sentinel execution",
    )
    parser.add_argument(
        "--exp-name",
        default=None,
        type=str,
        help="The experiment name prefix.",
    )
    
    args = parser.parse_args()
    
    if os.getenv("OPENAI_API_KEY") is None:
        print("ERROR: OPENAI_API_KEY is not set")
        return
    
    # Set models - use all
    models = [
        Model.GPT_4o,
        Model.GPT_4o_MINI,
        Model.GPT_41_MINI,
        Model.GPT_41,
        Model.GPT_41_NANO,
        Model.GPT_5_MINI,
        Model.GPT_5,
        Model.GPT_5_NANO,
        Model.GEMINI_25_FLASH,
        Model.GEMINI_25_FLASH_LITE,
        Model.GEMINI_25_PRO,
    ]
    
    print(f"Loading BioDEX dataset...")
    
    # Create data readers: test set for main evaluation, train set for validation
    data_reader = BioDEXDataReader(split="test", num_samples=args.num_texts, seed=args.seed)
    val_data_reader = BioDEXDataReader(split="train", num_samples=50, seed=args.seed)
    
    print(f"Processing {len(data_reader)} test medical texts with Palimpzest...")
    print(f"Using {len(val_data_reader)} train medical texts for validation...")
    
    # Load ChromaDB index for retrieval pipeline
    print("Loading ChromaDB index for retrieval pipeline...")
    chroma_client = chromadb.PersistentClient("experiments/reasoning/othersystems/biodex/.chroma-biodex")
    openai_ef = OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small",
    )
    index = chroma_client.get_collection("biodex-reaction-terms", embedding_function=openai_ef)

    def search_func(index: chromadb.Collection, query: list[list[float]], k: int) -> list[str]:
        # execute query with embeddings
        results = index.query(query, n_results=5)

        # get list of result terms with their cosine similarity scores
        final_results = []
        for query_docs, query_distances in zip(results["documents"], results["distances"]):
            for doc, dist in zip(query_docs, query_distances):
                cosine_similarity = 1 - dist
                final_results.append({"content": doc, "similarity": cosine_similarity})

        # sort the results by similarity score
        sorted_results = sorted(final_results, key=lambda result: result["similarity"], reverse=True)

        # remove duplicates
        sorted_results_set = set()
        final_sorted_results = []
        for result in sorted_results:
            if result["content"] not in sorted_results_set:
                sorted_results_set.add(result["content"])
                final_sorted_results.append(result["content"])

        # return the top-k similar results and generation stats
        return {"reaction_labels": final_sorted_results[:k]}
    
    # Run experiments for both pipeline types
    pipeline_types = ["retrieval", "direct"]
    all_results = {}
    
    for pipeline_type in pipeline_types:
        print(f"\n{'='*60}")
        print(f"Running {pipeline_type.upper()} Pipeline Experiments")
        print(f"{'='*60}")
        
        results = {}
        
        # Step 1: Run unconstrained max quality
        print(f"\n=== Step 1: Running unconstrained max quality ({pipeline_type}) ===")
        policy = MaxQuality()
        exp_name_unconstrained = f"{args.exp_name}-{pipeline_type}-unconstrained" if args.exp_name else f"pz-{pipeline_type}-unconstrained"
        unconstrained_result = run_experiment(
            data_reader, val_data_reader, policy, models,
            pipeline_type=pipeline_type,
            sentinel_strategy=args.sentinel_execution_strategy,
            k=args.k, j=args.j, sample_budget=args.sample_budget,
            seed=args.seed, exp_name=exp_name_unconstrained,
            index=index if pipeline_type == "retrieval" else None,
            search_func=search_func if pipeline_type == "retrieval" else None
        )
        results["unconstrained_max_quality"] = unconstrained_result
        unconstrained_cost = unconstrained_result["plan_execution_cost"]
        print(f"Unconstrained cost: ${unconstrained_cost:.4f}")
        print(f"Unconstrained RP@5: {unconstrained_result['avg_rp_at_5']:.4f}")
        
        # Step 2: Run at each budget fraction
        budget_targets = {}
        for i, frac in enumerate(FRACS, 2):
            budget = unconstrained_cost * frac
            budget_targets[f"budget_{int(frac*100)}_percent"] = budget
            
            print(f"\n=== Step {i}: Running max quality at {int(frac*100)}% budget (${budget:.4f}) ({pipeline_type}) ===")
            policy = MaxQualityAtFixedCost(max_cost=budget)
            exp_name_budget = f"{args.exp_name}-{pipeline_type}-{int(frac*100)}pct" if args.exp_name else f"pz-{pipeline_type}-{int(frac*100)}pct"
            
            budget_result = run_experiment(
                data_reader, val_data_reader, policy, models,
                pipeline_type=pipeline_type,
                sentinel_strategy=args.sentinel_execution_strategy,
                k=args.k, j=args.j, sample_budget=args.sample_budget,
                seed=args.seed, exp_name=exp_name_budget,
                index=index if pipeline_type == "retrieval" else None,
                search_func=search_func if pipeline_type == "retrieval" else None
            )
            results[f"budget_{int(frac*100)}_percent"] = budget_result
            print(f"{int(frac*100)}% budget cost: ${budget_result['plan_execution_cost']:.4f}")
            print(f"{int(frac*100)}% budget RP@5: {budget_result['avg_rp_at_5']:.4f}")
        
        # Add metadata for this pipeline
        results["metadata"] = {
            "seed": args.seed,
            "num_texts": args.num_texts,
            "unconstrained_cost": unconstrained_cost,
            "budget_fractions": FRACS,
            "budget_targets": budget_targets,
            "sentinel_execution_strategy": args.sentinel_execution_strategy,
            "k": args.k,
            "j": args.j,
            "sample_budget": args.sample_budget,
            "exp_name": args.exp_name,
            "system": "palimpzest",
            "pipeline_type": pipeline_type,
        }
        
        all_results[pipeline_type] = results
        
        # Save results for this pipeline
        output_path = f"experiments/reasoning/othersystems/biodex/pz_{pipeline_type}_evaluation.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📈 {pipeline_type.capitalize()} pipeline results saved to: {output_path}")
        
        # Print summary for this pipeline
        print(f"\n=== {pipeline_type.capitalize()} Pipeline Summary ===")
        print(f"{'Configuration':<30} {'Cost ($)':<12} {'RP@5':<12} {'RP@10':<12} {'Term Recall':<12}")
        print("-" * 78)
        
        # Print unconstrained result
        r = results["unconstrained_max_quality"]
        print(f"{'Unconstrained':<30} {r['total_execution_cost']:<12.4f} {r['avg_rp_at_5']:<12.4f} {r['avg_rp_at_10']:<12.4f} {r['avg_term_recall']:<12.4f}")
        
        # Print budget results
        for frac in FRACS:
            key = f"budget_{int(frac*100)}_percent"
            label = f"{int(frac*100)}% Budget"
            r = results[key]
            print(f"{label:<30} {r['total_execution_cost']:<12.4f} {r['avg_rp_at_5']:<12.4f} {r['avg_rp_at_10']:<12.4f} {r['avg_term_recall']:<12.4f}")
    
    # Save combined results
    combined_output_path = "experiments/reasoning/othersystems/biodex/pz_evaluation.json"
    with open(combined_output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n📈 Combined results for both pipelines saved to: {combined_output_path}")


if __name__ == "__main__":
    main()
