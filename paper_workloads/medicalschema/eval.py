import os
import pandas as pd
import json
from sklearn.metrics import precision_recall_fscore_support
from rich import print

med_schema_keys = [
    "case_submitter_id",
    "age_at_diagnosis",
    "race",
    "ethnicity",
    "gender",
    "vital_status",
    "ajcc_pathologic_t",
    "ajcc_pathologic_n",
    "ajcc_pathologic_stage",
    "tumor_grade",
    "tumor_focality",
    "tumor_largest_dimension_diameter",
    "primary_diagnosis",
    "morphology",
    "tissue_or_organ_of_origin",
]

with open(
    "/Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/medicalschema/extracted_patient_data_with_reduce.json"
) as f:
    data = json.load(f)

# Flatten the columns
flattened_data = []
for d in data:
    for i in range(max(len(d[k]) for k in med_schema_keys)):
        flattened_data.append(
            {
                **{k: d[k][i] if i < len(d[k]) else None for k in med_schema_keys},
                "study": d["file_name"].split(".")[0].split("/")[-1],
            }
        )

# COPIED FROM Palimpzest codebase
include_keys = [
    "age_at_diagnosis",
    "ajcc_pathologic_n",
    "ajcc_pathologic_stage",
    "ajcc_pathologic_t",
    "case_submitter_id",
    "ethnicity",
    "gender",
    "morphology",
    "primary_diagnosis",
    "race",
    "tissue_or_organ_of_origin",
    "tumor_focality",
    "tumor_grade",
    "tumor_largest_dimension_diameter",
    "vital_status",
    "study",
]

output_rows = [
    {k: v for k, v in rec.items() if k in include_keys} for rec in flattened_data
]

records_df = pd.DataFrame(output_rows)
pz_records_df = pd.read_csv(
    "/Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/medicalschema/pz_clean_output.csv"
)

output = records_df
index = [x for x in output.columns if x != "study"]
target_matching = pd.read_csv(
    "/Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/medicalschema/target_matching.csv",
    index_col=0,
).reindex(index)

studies = output["study"].unique()
df = pd.DataFrame(columns=target_matching.columns, index=index)
pz_df = pd.DataFrame(columns=target_matching.columns, index=index)
cols = output.columns
predicted = []
targets = []
pz_predicted = []
avg_f1 = []
avg_pz_f1 = []

for study in studies:
    output_study = output[output["study"] == study]
    pz_output_study = pz_records_df[pz_records_df["study"] == study]
    try:
        xl = pd.ExcelFile(
            os.path.join(
                "/Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/medicalschema/excel/",
                f"{study}.xlsx",
            )
        )
        sheet_names = xl.sheet_names
        input_dfs = [
            pd.read_excel(
                os.path.join(
                    "/Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/medicalschema/excel/",
                    f"{study}.xlsx",
                ),
                sheet_name=sheet_name,
            )
            for sheet_name in sheet_names
        ]
    except:
        print("Cannot find the study", study)
        targets += [study] * 5
        predicted += ["missing"] * 5
        pz_predicted += ["missing"] * 5
        continue

    for col in cols:
        if col == "study":
            continue
        max_matches = 0
        max_col = "missing"
        for input_df in input_dfs:
            for input_col in input_df.columns:
                matches = sum(
                    1
                    for idx, x in enumerate(output_study[col])
                    if idx < len(input_df[input_col]) and x == input_df[input_col][idx]
                )
                if matches > max_matches:
                    max_matches = matches
                    max_col = input_col

        df.loc[col, study] = max_col

        # Do the same for pz_output_study
        pz_max_matches = 0
        pz_max_col = "missing"
        for input_df in input_dfs:
            for input_col in input_df.columns:
                pz_matches = sum(
                    1
                    for idx, x in enumerate(pz_output_study[col])
                    if idx < len(input_df[input_col]) and x == input_df[input_col][idx]
                )
                if pz_matches > pz_max_matches:
                    pz_max_matches = pz_matches
                    pz_max_col = input_col
        pz_df.loc[col, study] = pz_max_col

    df.fillna("missing", inplace=True)
    pz_df.fillna("missing", inplace=True)

    targets += list(target_matching[study.split("_")[0]].values)
    predicted += list(df[study].values)
    pz_predicted += list(pz_df[study].values)

    # Compute metrics for targets & predicted
    p, r, f1, _ = precision_recall_fscore_support(
        targets, predicted, average="micro", zero_division=0
    )
    print(
        f"{study} - Our Predicted - F1: {f1:.4f}, Precision: {p:.4f}, Recall: {r:.4f}"
    )
    avg_f1.append(f1)

    # Compute metrics for targets & pz_predicted
    pz_p, pz_r, pz_f1, _ = precision_recall_fscore_support(
        targets, pz_predicted, average="micro", zero_division=0
    )
    print(
        f"{study} - PZ Predicted - F1: {pz_f1:.4f}, Precision: {pz_p:.4f}, Recall: {pz_r:.4f}"
    )
    avg_pz_f1.append(pz_f1)

print(f"PZ average F1: {sum(avg_pz_f1) / len(avg_pz_f1):.4f}")
print(f"Our average F1: {sum(avg_f1) / len(avg_f1):.4f}")
