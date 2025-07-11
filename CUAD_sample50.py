import json

input_path = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/CUAD-raw.json"
output_path = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/CUAD-raw-sample50.json"

with open(input_path, "r") as f:
    data = json.load(f)

sample = data[:50]

with open(output_path, "w") as f:
    json.dump(sample, f, indent=2)

print(f"Saved {len(sample)} records to {output_path}") 