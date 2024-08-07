import json

import uuid


def add_document_id(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    for item in data:
        item["document_id"] = str(uuid.uuid4())
        del item["tgt"]

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)


def take_five(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    for item in data[:5]:
        print(item)

    with open(output_file, "w") as f:
        json.dump(data[:5], f, indent=2)


if __name__ == "__main__":
    # input_file = "workloads/medical/raw.json"
    # output_file = "workloads/medical/raw_with_id.json"
    # add_document_id(input_file, output_file)
    # print(f"Added document_id to each item. Output saved to {output_file}")

    input_file = "workloads/medical/raw_with_id.json"
    output_file = "workloads/medical/raw_with_id_sample.json"
    take_five(input_file, output_file)
