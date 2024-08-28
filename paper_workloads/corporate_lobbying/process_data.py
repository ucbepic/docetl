import json
import csv
import uuid

# Load TSV file
with open(
    "/Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/corporate_lobbying/test.tsv",
    "r",
    encoding="utf-8",
) as tsv_file:
    tsv_reader = csv.reader(tsv_file, delimiter="\t")
    data = list(tsv_reader)

# Convert to list of dictionaries
json_data = []
headers = data[0]
for row in data[1:]:
    json_data.append(dict(zip(headers, row)))


# Add a UUID id to each dictionary in json_data
for item in json_data:
    item["id"] = str(uuid.uuid4())

# Create a ground_truth.json file that contains the id and "Answer"
ground_truth = [{"id": item["id"], "answer": item["answer"]} for item in json_data]

# Get rid of answer from json_data
for item in json_data:
    del item["answer"]

# Write to JSON file
with open(
    "/Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/corporate_lobbying/legal.json",
    "w",
    encoding="utf-8",
) as json_file:
    json.dump(json_data, json_file, indent=2, ensure_ascii=False)

with open(
    "/Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/corporate_lobbying/ground_truth.json",
    "w",
    encoding="utf-8",
) as json_file:
    json.dump(ground_truth, json_file, indent=2, ensure_ascii=False)

print("Data successfully converted from TSV to JSON.")
