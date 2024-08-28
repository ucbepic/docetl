import random
import uuid
from datasets import load_dataset
import json
import os


def get_current_dir():
    return os.path.dirname(os.path.abspath(__file__))


def load_and_sample_dataset(dataset_name, sample_size=250):
    ds = load_dataset(dataset_name)
    return ds["test"].select(range(sample_size))


def select_attributes(dataset, attributes):
    return dataset.select_columns(attributes)


def save_json(data, filename):
    file_path = os.path.join(get_current_dir(), filename)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} items to {file_path}")


def load_terms(filename):
    file_path = os.path.join(get_current_dir(), filename)
    with open(file_path, "r") as terms_file:
        return [{"reaction": line.strip()} for line in terms_file if line.strip()]


def main():
    # Process BioDEX dataset
    ds = load_and_sample_dataset("BioDEX/BioDEX-Reactions")
    selected_data = select_attributes(ds, ["fulltext_processed", "reactions"])
    # Generate unique IDs for each item
    ids = [str(uuid.uuid4()) for _ in range(len(selected_data))]

    # Create the first file with id and fulltext_processed
    selected_data_json_1 = [
        {
            "id": id,
            "fulltext_processed": item["fulltext_processed"],
        }
        for id, item in zip(ids, selected_data)
    ]
    save_json(selected_data_json_1, "biodex_sample.json")

    # Create the second file with id and ground_truth_reactions
    selected_data_json_2 = [
        {
            "id": id,
            "ground_truth_reactions": item["reactions"].split(", "),
        }
        for id, item in zip(ids, selected_data)
    ]
    save_json(selected_data_json_2, "biodex_ground_truth.json")

    # Process terms
    terms_list = load_terms("biodex_terms.txt")
    save_json(terms_list, "biodex_terms.json")

    # Load priors from biodex_priors.json
    with open(os.path.join(get_current_dir(), "priors.json"), "r") as f:
        priors = json.load(f)

    # Sort terms by weight in descending order
    sorted_terms = sorted(priors.items(), key=lambda x: x[1], reverse=True)

    # Take the top 50 terms
    sampled_terms = [term for term, _ in sorted_terms[:100]]

    # Create examples list
    examples = [{"examples": ", ".join(sampled_terms)}]

    # Save to examples.json
    save_json(examples, "examples.json")


if __name__ == "__main__":
    main()
