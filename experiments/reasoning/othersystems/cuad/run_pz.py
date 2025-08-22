import argparse
import json
import os
import string
from functools import partial
import numpy as np
import pandas as pd
from pathlib import Path

import palimpzest as pz
from palimpzest.constants import Model
from palimpzest.core.lib.fields import ListField, StringField
from palimpzest.policy import MaxQuality, MaxQualityAtFixedCost

from dotenv import load_dotenv
load_dotenv()

# Add the evaluation utils to the path
from experiments.reasoning.evaluation.utils import get_evaluate_func
from functools import partial
import string

# Budget fractions to test
FRACS = [0.75, 0.5, 0.25, 0.1]

# Evaluation functions for score_fn
IOU_THRESH = 0.15

def get_jaccard(label, pred):
    """Calculate Jaccard similarity between two strings."""
    remove_tokens = [c for c in string.punctuation if c != "/"]
    for token in remove_tokens:
        label = label.replace(token, "")
        pred = pred.replace(token, "")
    label = label.lower()
    pred = pred.lower()
    label = label.replace("/", " ")
    pred = pred.replace("/", " ")

    label_words = set(label.split(" "))
    pred_words = set(pred.split(" "))

    intersection = label_words.intersection(pred_words)
    union = label_words.union(pred_words)
    jaccard = len(intersection) / len(union) if union else 0
    return jaccard

def evaluate_entry(labels, preds, substr_ok):
    """Find the number of true positives, false positives, and false negatives."""
    tp, fp, fn = 0, 0, 0

    # Convert predictions to strings if needed
    for idx, pred in enumerate(preds):
        if not isinstance(pred, str):
            preds[idx] = str(pred)

    # Handle empty labels
    if len(labels) == 0:
        if len(preds) > 0:
            fp += len(preds)
    else:
        for ans in labels:
            if len(ans) == 0:
                continue
            match_found = False
            for pred in preds:
                if substr_ok:
                    is_match = get_jaccard(ans, pred) >= IOU_THRESH or ans in pred
                else:
                    is_match = get_jaccard(ans, pred) >= IOU_THRESH
                if is_match:
                    match_found = True
                    break

            if match_found:
                tp += 1
            else:
                fn += 1

        # Check for false positives
        for pred in preds:
            match_found = False
            for ans in labels:
                if len(ans) == 0:
                    continue
                if substr_ok:
                    is_match = get_jaccard(ans, pred) >= IOU_THRESH or ans in pred
                else:
                    is_match = get_jaccard(ans, pred) >= IOU_THRESH
                if is_match:
                    match_found = True
                    break

            if not match_found:
                fp += 1

    return tp, fp, fn

def handle_empty_preds(preds):
    """Handle empty or None predictions."""
    if preds is None or (isinstance(preds, str) and (preds == "" or preds == " " or preds == "null" or preds == "None")):
        return []
    elif isinstance(preds, float) and np.isnan(preds):
        return []
    if not isinstance(preds, (list, np.ndarray)):
        return [preds]
    return preds

cuad_categories = [
    {
        "Category": "Document Name",
        "Description": "The name of the contract",
        "Answer Format": "Contract Name",
        "Group": "Group: -",
    },
    {
        "Category": "Parties",
        "Description": "The two or more parties who signed the contract",
        "Answer Format": "Entity or individual names",
        "Group": "Group: -",
    },
    {
        "Category": "Agreement Date",
        "Description": "The date of the contract",
        "Answer Format": "Date (mm/dd/yyyy)",
        "Group": "Group: 1",
    },
    {
        "Category": "Effective Date",
        "Description": "The date when the contract is effective ",
        "Answer Format": "Date (mm/dd/yyyy)",
        "Group": "Group: 1",
    },
    {
        "Category": "Expiration Date",
        "Description": "On what date will the contract's initial term expire?",
        "Answer Format": "Date (mm/dd/yyyy) / Perpetual",
        "Group": "Group: 1",
    },
    {
        "Category": "Renewal Term",
        "Description": "What is the renewal term after the initial term expires? This includes automatic extensions and unilateral extensions with prior notice.",
        "Answer Format": "[Successive] number of years/months / Perpetual",
        "Group": "Group: 1",
    },
    {
        "Category": "Notice Period to Terminate Renewal",
        "Description": "What is the notice period required to terminate renewal?",
        "Answer Format": "Number of days/months/year(s)",
        "Group": "Group: 1",
    },
    {
        "Category": "Governing Law",
        "Description": "Which state/country's law governs the interpretation of the contract?",
        "Answer Format": "Name of a US State / non-US Province, Country",
        "Group": "Group: -",
    },
    {
        "Category": "Most Favored Nation",
        "Description": "Is there a clause that if a third party gets better terms on the licensing or sale of technology/goods/services described in the contract, the buyer of such technology/goods/services under the contract shall be entitled to those better terms?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Non-Compete",
        "Description": "Is there a restriction on the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector? ",
        "Answer Format": "Yes/No",
        "Group": "Group: 2",
    },
    {
        "Category": "Exclusivity",
        "Description": "Is there an exclusive dealing  commitment with the counterparty? This includes a commitment to procure all \"requirements\" from one party of certain technology, goods, or services or a prohibition on licensing or selling technology, goods or services to third parties, or a prohibition on  collaborating or working with other parties), whether during the contract or  after the contract ends (or both).",
        "Answer Format": "Yes/No",
        "Group": "Group: 2",
    },
    {
        "Category": "No-Solicit of Customers",
        "Description": "Is a party restricted from contracting or soliciting customers or partners of the counterparty, whether during the contract or after the contract ends (or both)?",
        "Answer Format": "Yes/No",
        "Group": "Group: 2",
    },
    {
        "Category": "Competitive Restriction Exception",
        "Description": "This category includes the exceptions or carveouts to Non-Compete, Exclusivity and No-Solicit of Customers above.",
        "Answer Format": "Yes/No",
        "Group": "Group: 2",
    },
    {
        "Category": "No-Solicit of Employees",
        "Description": "Is there a restriction on a party's soliciting or hiring employees and/or contractors from the  counterparty, whether during the contract or after the contract ends (or both)?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Non-Disparagement",
        "Description": "Is there a requirement on a party not to disparage the counterparty?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Termination for Convenience",
        "Description": "Can a party terminate this  contract without cause (solely by giving a notice and allowing a waiting  period to expire)?",
        "Answer Format": "Yes/No",
        "Group": "Group: 3",
    },
    {
        "Category": "Rofr/Rofo/Rofn",
        "Description": "Is there a clause granting one party a right of first refusal, right of first offer or right of first negotiation to purchase, license, market, sell, lease, etc. certain technology, goods, services or securities?",
        "Answer Format": "Yes/No",
        "Group": "Group: 3",
    },
    {
        "Category": "Change of Control",
        "Description": "Does one party have the right to terminate or is consent or notice required of the counterparty if such party undergoes a change of control, such as a merger, stock sale, transfer of all or substantially all of its assets or business, or assignment by operation of law?",
        "Answer Format": "Yes/No",
        "Group": "Group: 3",
    },
    {
        "Category": "Anti-Assignment",
        "Description": "Is consent or notice required of a party if the contract is assigned to a third party?",
        "Answer Format": "Yes/No",
        "Group": "Group: 3",
    },
    {
        "Category": "Revenue/Profit Sharing",
        "Description": "Is one party required to share revenue or profit with the counterparty for any technology, goods, or services?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Price Restrictions",
        "Description": "Is there a restriction on the  ability of a party to raise or reduce prices of technology, goods, or  services provided?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Minimum Commitment",
        "Description": "Is there a minimum order size or minimum amount or units per-time period that one party must buy from the counterparty under the contract?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Volume Restriction",
        "Description": "Is there a fee increase or consent requirement, etc. if one party's use of the product/services exceeds certain threshold?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "IP Ownership Assignment",
        "Description": "Does intellectual property created  by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events?",
        "Answer Format": "Yes/No",
        "Group": "Group: 4",
    },
    {
        "Category": "Joint IP Ownership",
        "Description": "Is there any clause where intellectual property created by one party is jointly owned by both parties to the contract?",
        "Answer Format": "Yes/No",
        "Group": "Group: 4",
    },
    {
        "Category": "License Grant",
        "Description": "Does the contract contain a license granted by one party to its counterparty?",
        "Answer Format": "Yes/No",
        "Group": "Group: 4",
    },
    {
        "Category": "Non-Transferable License",
        "Description": "Does the contract limit the ability of a party to transfer the license being granted to a third party?",
        "Answer Format": "Yes/No",
        "Group": "Group: 4",
    },
    {
        "Category": "Affiliate License-Licensor",
        "Description": "Does the contract contain a license grant by affiliates of the licensor or that includes intellectual property of affiliates of the licensor? ",
        "Answer Format": "Yes/No",
        "Group": "Group: 4",
    },
    {
        "Category": "Affiliate License-Licensee",
        "Description": "Does the contract contain a license grant to a licensee (incl. sublicensor) and the affiliates of such licensee/sublicensor?",
        "Answer Format": "Yes/No",
        "Group": "Group: 4",
    },
    {
        "Category": "Unlimited/All-You-Can-Eat-License",
        "Description": "Is there a clause granting one party an \"enterprise,\" \"all you can eat\" or unlimited usage license?",
        "Answer Format": "Yes/No",
        "Group": "Group: 4",
    },
    {
        "Category": "Irrevocable or Perpetual License",
        "Description": "Does the contract contain a  license grant that is irrevocable or perpetual?",
        "Answer Format": "Yes/No",
        "Group": "Group: 4",
    },
    {
        "Category": "Source Code Escrow",
        "Description": "Is one party required to deposit its source code into escrow with a third party, which can be released to the counterparty upon the occurrence of certain events (bankruptcy,  insolvency, etc.)?",
        "Answer Format": "Yes/No",
        "Group": "Group: 4",
    },
    {
        "Category": "Post-Termination Services",
        "Description": "Is a party subject to obligations after the termination or expiration of a contract, including any post-termination transition, payment, transfer of IP, wind-down, last-buy, or similar commitments?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Audit Rights",
        "Description": "Does a party have the right to  audit the books, records, or physical locations of the counterparty to ensure compliance with the contract?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Uncapped Liability",
        "Description": "Is a party's liability uncapped upon the breach of its obligation in the contract? This also includes uncapped indemnification obligations.",
        "Answer Format": "Yes/No",
        "Group": "Group: 5",
    },
    {
        "Category": "Cap on Liability",
        "Description": "Does the contract include a cap on liability upon the breach of a party's obligation? This includes time limitation for the counterparty to bring claims or maximum amount for recovery.",
        "Answer Format": "Yes/No",
        "Group": "Group: 5",
    },
    {
        "Category": "Liquidated Damages",
        "Description": "Does the contract contain a clause that would award either party liquidated damages for breach or a fee upon the termination of a contract (termination fee)?",
        "Answer Format": "Yes/No",
        "Group": "Group: 5",
    },
    {
        "Category": "Warranty Duration",
        "Description": "What is the duration of any  warranty against defects or errors in technology, products, or services  provided under the contract?",
        "Answer Format": "Yes/No",
        "Group": "Group: 5",
    },
    {
        "Category": "Insurance",
        "Description": "Is there a requirement for insurance that must be maintained by one party for the benefit of the counterparty?",
        "Answer Format": "Yes/No",
        "Group": "Group: 5",
    },
    {
        "Category": "Covenant Not to Sue",
        "Description": "Is a party restricted from contesting the validity of the counterparty's ownership of intellectual property or bringing a claim against the counterparty for matters unrelated to the contract?",
        "Answer Format": "Yes/No",
        "Group": "Group: 5",
    },
    {
        "Category": "Third Party Beneficiary",
        "Description": "Is there a non-contracting party who is a beneficiary to some or all of the clauses in the contract and therefore can enforce its rights against a contracting party?",
        "Answer Format": "Yes/No",
        "Group": "Group: 5",
    },
]


def load_cuad_data(split="train"):
    """Load CUAD data from our existing JSON files."""
    file_path = f"experiments/reasoning/data/{split}/cuad.json"
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


class CUADDataReader(pz.DataReader):
    def __init__(self, num_contracts: int = 100, split: str = "train", seed: int = 42):
        self.num_contracts = num_contracts
        self.split = split
        self.seed = seed

        input_cols = [
            {"name": "contract_id", "type": str, "desc": "The id of the contract to be analyzed"},
            {"name": "name", "type": str, "desc": "The name of the contract document"},
            {"name": "document", "type": str, "desc": "The content of the contract to be analyzed"},
        ]
        super().__init__(input_cols)

        # convert the dataset into a list of dictionaries where each row is for a single contract
        include_labels = split == "train"
        dataset = load_cuad_data(split=split)
        
        self.dataset = self._construct_dataset(dataset, num_contracts, seed, include_labels)

    def _construct_dataset(self, dataset, num_contracts, seed: int = 42, include_labels: bool = False):
        # get the set of unique contract names; to ensure the order of the contracts is
        # preserved, we use a list rather than using python's set()
        contract_names = []
        for row in dataset:
            if row["name"] not in contract_names:
                contract_names.append(row["name"])

        # shuffle the contracts for the given seed
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(contract_names)

        # get the first num_contracts
        contract_names = contract_names[:num_contracts]

        # construct the dataset one contract at a time
        new_dataset = []
        for name in contract_names:
            # get the contract data
            contract_data = next(row for row in dataset if row["name"] == name)

            # construct the contract
            contract = {
                "contract_id": contract_data["id"],
                "name": name,
                "document": contract_data["document"],
            }

            # for train / validation data, add the labels from CSV file
            if include_labels:
                contract = {"fields": contract}

                # Load ground truth labels from CSV
                ground_truth_file = "experiments/reasoning/data/CUAD-master_clauses.csv"
                ground_truth_df = pd.read_csv(ground_truth_file)
                
                # Normalize column names
                ground_truth_df.columns = ground_truth_df.columns.map(
                    lambda x: x.replace(" ", "_").lower()
                )
                
                # Normalize filename column for matching
                ground_truth_df["filename"] = ground_truth_df["filename"].apply(
                    lambda x: x.upper()
                    .replace(".", "")
                    .replace(",", "")
                    .replace(" ", "")
                    .replace("_", "")
                    .replace("-", "")
                    .replace("'", "")
                    .replace(r'[^a-zA-Z0-9]$', '')
                )

                # Normalize the document name for matching
                normalized_name = (name.split("/")[-1]
                                 .upper()
                                 .rstrip(".TXT")
                                 .replace(".", "")
                                 .replace(",", "")
                                 .replace(" ", "")
                                 .replace("_", "")
                                 .replace("-", "")
                                 .replace("'", "")
                                 .replace(r'[^a-zA-Z0-9]$', ''))

                # Find closest matching filename
                closest_match = max(
                    ground_truth_df["filename"],
                    key=lambda x: sum(a == b for a, b in zip(x, normalized_name))
                )
                matching_gt_rows = ground_truth_df[ground_truth_df["filename"] == closest_match]

                # add the labels
                category_names = list(map(lambda category: category["Category"], cuad_categories))
                contract["labels"] = {category: [] for category in category_names}
                contract["score_fn"] = {category: None for category in category_names}

                if not matching_gt_rows.empty:
                    gt_row = matching_gt_rows.iloc[0]
                    
                    # Map ground truth columns to our category names
                    category_mapping = {
                        "Document Name": "document_name",
                        "Parties": "parties", 
                        "Agreement Date": "agreement_date",
                        "Effective Date": "effective_date",
                        "Expiration Date": "expiration_date",
                        "Renewal Term": "renewal_term",
                        "Notice Period to Terminate Renewal": "notice_to_terminate_renewal",
                        "Governing Law": "governing_law",
                        "Most Favored Nation": "most_favored_nation",
                        "Non-Compete": "non_compete",
                        "Exclusivity": "exclusivity",
                        "No-Solicit of Customers": "no_solicit_of_customers",
                        "Competitive Restriction Exception": "competitive_restriction_exception",
                        "No-Solicit of Employees": "no_solicit_of_employees",
                        "Non-Disparagement": "non_disparagement",
                        "Termination for Convenience": "termination_for_convenience",
                        "Rofr/Rofo/Rofn": "right_of_first_refusal",
                        "Change of Control": "change_of_control",
                        "Anti-Assignment": "anti_assignment",
                        "Revenue/Profit Sharing": "revenue_profit_sharing",
                        "Price Restrictions": "price_restriction",
                        "Minimum Commitment": "minimum_commitment",
                        "Volume Restriction": "volume_restriction",
                        "IP Ownership Assignment": "ip_ownership_assignment",
                        "Joint IP Ownership": "joint_ip_ownership",
                        "License Grant": "license_grant",
                        "Non-Transferable License": "non_transferable_license",
                        "Affiliate License-Licensor": "affiliate_ip_license_licensor",
                        "Affiliate License-Licensee": "affiliate_ip_license_licensee",
                        "Unlimited/All-You-Can-Eat-License": "unlimited_license",
                        "Irrevocable or Perpetual License": "irrevocable_or_perpetual_license",
                        "Source Code Escrow": "source_code_escrow",
                        "Post-Termination Services": "post_termination_services",
                        "Audit Rights": "audit_rights",
                        "Uncapped Liability": "uncapped_liability",
                        "Cap on Liability": "cap_on_liability",
                        "Liquidated Damages": "liquidated_damages",
                        "Warranty Duration": "warranty_duration",
                        "Insurance": "insurance",
                        "Covenant Not to Sue": "covenant_not_to_sue",
                        "Third Party Beneficiary": "third_party_beneficiary",
                    }

                    # Extract ground truth labels for each category
                    for category in cuad_categories:
                        category_name = category["Category"]
                        gt_column = category_mapping.get(category_name, category_name.lower().replace(" ", "_").replace("-", "_"))
                        
                        if gt_column in gt_row:
                            gt_value = gt_row[gt_column]
                            # Parse the ground truth value
                            if pd.isna(gt_value) or gt_value == "" or gt_value == "[]":
                                contract["labels"][category_name] = []
                            else:
                                # Try to parse as list if it looks like one
                                if isinstance(gt_value, str) and gt_value.startswith("["):
                                    try:
                                        import ast
                                        parsed_value = ast.literal_eval(gt_value)
                                        contract["labels"][category_name] = parsed_value if isinstance(parsed_value, list) else [str(parsed_value)]
                                    except:
                                        contract["labels"][category_name] = [str(gt_value)]
                                else:
                                    contract["labels"][category_name] = [str(gt_value)] if gt_value else []
                        else:
                            contract["labels"][category_name] = []

                # Create score functions for each category
                for category_name in category_names:
                    def score_fn(preds, labels, category_name):
                        preds = handle_empty_preds(preds)
                        entry_tp, _, entry_fn = evaluate_entry(labels, preds, substr_ok=True) if category_name == "Parties" else evaluate_entry(labels, preds, substr_ok=False)
                        score = None
                        if len(labels) > 0:  # noqa: SIM108
                            score = entry_tp / (entry_tp + entry_fn)
                        else:
                            score = 1.0 if len(preds) == 0 else 0.0

                        return score

                    contract["score_fn"][category_name] = partial(score_fn, category_name=category_name)

            # add the contract to the dataset
            new_dataset.append(contract)

        return new_dataset

    def __len__(self):
        return self.num_contracts

    def __getitem__(self, idx: int):
        return self.dataset[idx]
    
    def get_label_df(self):
        full_dataset = load_cuad_data(split=self.split)
        
        label_dataset = self._construct_dataset(full_dataset, self.num_contracts, self.seed, True)
        final_label_dataset = []
        for entry in label_dataset:
            row = {}
            row["contract_id"] = entry["fields"]["contract_id"]
            row["name"] = entry["fields"]["name"]  # Changed from "title" to "name"
            row["document"] = entry["fields"]["document"]  # Changed from "contract" to "document"
            row = {**row, **entry["labels"]}
            final_label_dataset.append(row)

        return pd.DataFrame(final_label_dataset)


def build_cuad_query(dataset):
    """Build the CUAD query with all categories in single column mode."""
    ds = pz.Dataset(dataset)
    
    cols = []
    for category in cuad_categories:
        desc = (
            f"Extract the text spans (if they exist) from the contract corresponding to {category['Description']}"
        )
        cols.append({"name": category["Category"], "type": ListField(StringField), "desc": desc})
    
    desc = "Extract the text spans (if they exist) from the contract for all specified categories."
    ds = ds.sem_add_columns(cols, desc=desc, depends_on=["document"])
    
    return ds


def convert_predictions_to_clauses_format(pred_df):
    """Convert Palimpzest predictions to the clauses format expected by evaluation."""
    results = []

    for _, row in pred_df.iterrows():
        # Create clauses list in the format expected by evaluation
        clauses = []
        for category in cuad_categories:
            category_name = category["Category"]
            pred_value = row.get(category_name, [])
            
            # Handle empty or null predictions
            if pred_value is None:
                text_span = ""
            elif isinstance(pred_value, (list, np.ndarray)) and len(pred_value) == 0:
                text_span = ""
            elif hasattr(pred_value, '__len__') and len(pred_value) == 1:
                # Handle single-element arrays/lists
                single_val = pred_value[0] if isinstance(pred_value, (list, np.ndarray)) else pred_value
                if pd.isna(single_val) if np.isscalar(single_val) else False:
                    text_span = ""
                else:
                    text_span = str(single_val).strip()
            elif isinstance(pred_value, (list, np.ndarray)):
                # Join list/array items with commas, filtering out null/empty values
                valid_items = []
                for item in pred_value:
                    if item is not None and (not np.isscalar(item) or not pd.isna(item)) and str(item).strip():
                        valid_items.append(str(item).strip())
                text_span = ", ".join(valid_items)
            elif np.isscalar(pred_value) and pd.isna(pred_value):
                text_span = ""
            elif isinstance(pred_value, str):
                text_span = pred_value.strip()
            else:
                text_span = str(pred_value)
            
            clauses.append({
                "clause_type": category_name,
                "text_span": text_span
            })
        
        # Create the record in the format matching Lotus output
        record = {
            "document": row["document"],
            "name": row["name"],
            "id": row["contract_id"],
            "clauses": clauses
        }
        results.append(record)
    
    return results


def run_experiment(data_reader, val_data_reader, policy, models, 
                   sentinel_strategy="mab", k=10, j=3, sample_budget=100, seed=42, exp_name=None):
    """Run a single experiment with given policy and return results."""
    print(f"\nRunning experiment with policy: {policy}")
    
    # Build query
    query = build_cuad_query(data_reader)
    
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
    
    print(data_reader.get_label_df())
    
    # Execute the query
    data_record_collection = query.run(
        config=config,
        k=k,
        j=j,
        sample_budget=sample_budget,
        seed=seed,
        exp_name=exp_name if exp_name else f"cuad-pz-{policy.__class__.__name__}",
        priors=None,
    )
    
    pred_df = data_record_collection.to_df()
    
    # Convert to clauses format
    results_list = convert_predictions_to_clauses_format(pred_df)
    
    # Save results as JSON
    output_file = f"experiments/reasoning/othersystems/cuad/{exp_name}.json" if exp_name else "experiments/reasoning/othersystems/cuad/pz_temp.json"
    with open(output_file, 'w') as f:
        json.dump(results_list, f, indent=2)
    
    # Evaluate using our existing evaluation framework
    evaluate_func = get_evaluate_func("cuad")
    metrics = evaluate_func("palimpzest", output_file)
    
    # Get execution statistics for cost
    exec_stats = data_record_collection.execution_stats
    
    return {
        "precision": metrics["avg_precision"],
        "recall": metrics["avg_recall"],
        "f1": metrics["avg_f1"],
        "optimization_time": exec_stats.optimization_time if exec_stats else 0,
        "optimization_cost": exec_stats.optimization_cost if exec_stats else 0,
        "plan_execution_time": exec_stats.plan_execution_time if exec_stats else 0,
        "plan_execution_cost": exec_stats.plan_execution_cost if exec_stats else 0,
        "total_execution_time": exec_stats.total_execution_time if exec_stats else 0,
        "total_execution_cost": exec_stats.total_execution_cost if exec_stats else 0,
        "output_file": output_file,
        "sentinel_strategy": sentinel_strategy,
        "k": k,
        "j": j,
        "sample_budget": sample_budget,
    }


def main():
    parser = argparse.ArgumentParser(description="Run CUAD experiments with budget analysis using Palimpzest")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-contracts", type=int, default=100, help="Number of contracts to process")
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
        default=50,
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
    
    # Set models - use both GPT-4o and GPT-4o-mini
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
    
    print(f"Loading CUAD dataset...")
    
    # Create data readers: test set for main evaluation, train set for validation
    data_reader = CUADDataReader(split="test", num_contracts=args.num_contracts, seed=args.seed)
    val_data_reader = CUADDataReader(split="train", num_contracts=50, seed=args.seed)
    
    print(f"Processing {len(data_reader)} test contracts with Palimpzest...")
    print(f"Using {len(val_data_reader)} train contracts for validation...")
    
    results = {}
    
    # Step 1: Run unconstrained max quality
    print("\n=== Step 1: Running unconstrained max quality ===")
    policy = MaxQuality()
    exp_name_unconstrained = f"{args.exp_name}-unconstrained" if args.exp_name else "pz-unconstrained"
    unconstrained_result = run_experiment(
        data_reader, val_data_reader, policy, models,
        sentinel_strategy=args.sentinel_execution_strategy,
        k=args.k, j=args.j, sample_budget=args.sample_budget,
        seed=args.seed, exp_name=exp_name_unconstrained
    )
    results["unconstrained_max_quality"] = unconstrained_result
    unconstrained_cost = unconstrained_result["plan_execution_cost"]
    print(f"Unconstrained cost: ${unconstrained_cost:.4f}")
    print(f"Unconstrained F1: {unconstrained_result['f1']:.4f}")
    
    # Step 2: Run at each budget fraction
    budget_targets = {}
    for i, frac in enumerate(FRACS, 2):
        budget = unconstrained_cost * frac
        budget_targets[f"budget_{int(frac*100)}_percent"] = budget
        
        print(f"\n=== Step {i}: Running max quality at {int(frac*100)}% budget (${budget:.4f}) ===")
        policy = MaxQualityAtFixedCost(max_cost=budget)
        exp_name_budget = f"{args.exp_name}-{int(frac*100)}pct" if args.exp_name else f"pz-{int(frac*100)}pct"
        
        budget_result = run_experiment(
            data_reader, val_data_reader, policy, models,
            sentinel_strategy=args.sentinel_execution_strategy,
            k=args.k, j=args.j, sample_budget=args.sample_budget,
            seed=args.seed, exp_name=exp_name_budget
        )
        results[f"budget_{int(frac*100)}_percent"] = budget_result
        print(f"{int(frac*100)}% budget cost: ${budget_result['plan_execution_cost']:.4f}")
        print(f"{int(frac*100)}% budget F1: {budget_result['f1']:.4f}")
    
    # Add metadata
    results["metadata"] = {
        "seed": args.seed,
        "num_contracts": args.num_contracts,
        "unconstrained_cost": unconstrained_cost,
        "budget_fractions": FRACS,
        "budget_targets": budget_targets,
        "sentinel_execution_strategy": args.sentinel_execution_strategy,
        "k": args.k,
        "j": args.j,
        "sample_budget": args.sample_budget,
        "exp_name": args.exp_name,
        "system": "palimpzest",
    }
    
    # Save combined results
    output_path = "experiments/reasoning/othersystems/cuad/pz_evaluation.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📈 Combined results saved to: {output_path}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"{'Configuration':<30} {'Cost ($)':<12} {'F1 Score':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 78)
    
    # Print unconstrained result
    r = results["unconstrained_max_quality"]
    print(f"{'Unconstrained':<30} {r['total_execution_cost']:<12.4f} {r['f1']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f}")
    
    # Print budget results
    for frac in FRACS:
        key = f"budget_{int(frac*100)}_percent"
        label = f"{int(frac*100)}% Budget"
        r = results[key]
        print(f"{label:<30} {r['total_execution_cost']:<12.4f} {r['f1']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f}")


if __name__ == "__main__":
    main()
