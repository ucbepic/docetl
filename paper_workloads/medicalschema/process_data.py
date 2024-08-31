import os
import json
import uuid
import pandas as pd


def process_excel_files(directory):
    result = []
    for filename in os.listdir(directory):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            file_path = os.path.join(directory, filename)

            # Get sheet names using pandas
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names
            sheet_data = ""

            for sheet_name in sheet_names:
                # Read the sheet into a pandas DataFrame
                df = pd.read_excel(file_path, sheet_name=sheet_name)

                # Convert DataFrame to string
                curr_sheet_data = df.to_string(index=False)
                sheet_data += (
                    f"Sheet name: {sheet_name}:\nSheet data: {curr_sheet_data}\n\n"
                )

            # Create an object with id, sheet_name, and sheet_data
            result.append(
                {
                    "id": str(uuid.uuid4()),
                    "sheet_data": sheet_data,
                    "file_name": filename,
                }
            )

    return result


# Directory containing Excel files
excel_directory = "/Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/medicalschema/excel"

# Process all Excel files
data = process_excel_files(excel_directory)

# Write the result to a JSON file
output_file = "/Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/medicalschema/excel_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Data successfully extracted and saved to {output_file}")
