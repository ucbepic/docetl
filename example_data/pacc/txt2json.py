import os
import json

def txt_files_to_json(folder_path, output_file):
    data_items = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            print ("Processing file: ", filename)
            data = {}
            date = filename.split(' ')[0]
            filepath = os.path.join(folder_path, filename)

            with open(os.path.join(folder_path, filename), 'r') as file:
                content = file.read()
                data["date"] = date
                data["url"] = filepath
                data["content"] = content

            print("Processed file: ", filename)
            data_items.append(data)
        
    with open(output_file, 'w') as json_file:
        json.dump(data_items, json_file, indent=4)

if __name__ == "__main__":
    folder_path = "./pdfs/2024"
    output_file = "output.json"
    txt_files_to_json(folder_path, output_file)