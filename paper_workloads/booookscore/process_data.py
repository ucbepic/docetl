import json
import pickle
import uuid

# Load pkl
with open(
    "/Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/booookscore/example_all_books.pkl",
    "rb",
) as f:
    data = pickle.load(f)

books = [{"book": k, "text": v, "id": str(uuid.uuid4())} for k, v in data.items()]

# write to json
with open(
    "/Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/booookscore/books.json",
    "w",
    encoding="utf-8",
) as json_file:
    json.dump(books, json_file, indent=2)
