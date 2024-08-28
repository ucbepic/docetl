import tempfile
import json
import os
from motion.builder import Optimizer

# Define the pipeline configuration
PIPELINE = """
datasets:
  biodex_sample:
    type: file
    path: /Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/biodex/biodex_sample.json
  biodex_terms:
    type: file
    path: /Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/biodex/biodex_terms.json

default_model: gpt-4o-mini

operations:
  extract_reactions:
    type: map
    optimize: false
    prompt: |
      Given the following medical article, extract what a patient is experiencing:
      {{ input.fulltext_processed }}

      Here are some examples:
      Drug ineffective, Off label use, Product use in unapproved indication, Nausea, Condition aggravated, Acute kidney injury, Diarrhoea, Thrombocytopenia, Toxicity to various agents, Pyrexia, Neutropenia, Drug interaction, Vomiting, Anaemia, Fatigue, Pneumonia, Sepsis, Disease progression, Hypotension, Dyspnoea, Death, Respiratory failure, Febrile neutropenia, Drug resistance, Foetal exposure during pregnancy, Treatment failure, Maternal exposure during pregnancy, Rash, Overdose, Decreased appetite, Exposure during pregnancy, Headache, Septic shock, Asthenia, Product use issue, Abdominal pain, Multiple organ dysfunction syndrome, Renal impairment, Leukopenia, Malignant neoplasm progression, Drug ineffective for unapproved indication, Hypertension, Cardiac arrest, Pancytopenia, Tachycardia, Pleural effusion, Neuropathy peripheral, Seizure, Premature baby, Renal failure, Infection, Drug-induced liver injury, Dizziness, Drug intolerance, Somnolence, Weight decreased, Disease recurrence, Metabolic acidosis, Alanine aminotransferase increased, Bone marrow failure, Hyponatraemia, Pulmonary embolism, Confusional state, Haemorrhage, Blood creatinine increased, Mucosal inflammation, Treatment noncompliance, Intentional overdose, Cardiac failure, Hypokalaemia, Constipation, Cytomegalovirus infection, Therapy non-responder, Gastrointestinal disorder, Aspartate aminotransferase increased, Cough, Pain, Rhabdomyolysis, Hypoxia, Urinary tract infection, Electrocardiogram QT prolonged, Pathogen resistance, Arthralgia, Oedema peripheral, Haemoglobin decreased, Suicide attempt, No adverse event, Gastrointestinal haemorrhage, Muscular weakness, Neurotoxicity, Pruritus, Therapeutic response decreased, Generalised tonic-clonic seizure, Loss of consciousness, Interstitial lung disease, Hepatotoxicity, Hyperkalaemia, Dehydration, Palmar-plantar erythrodysaesthesia syndrome, Fall...(25000 more)

      Your list should be between 1 and 10 terms, comma-separated.
    output:
      schema:
        labels: string

    
  match_reactions:
    type: equijoin
    join_key:
      left:
        name: labels
      right:
        name: reaction
    embedding_model: text-embedding-3-small
    comparison_prompt: |
      Can the following condition be found in the following medical article?
      Medical article: {{ left.fulltext_processed }}
      
      Condition: {{ right.reaction }}

      Determine if the condition is experienced by a patient in the medical article.
      
optimizer_config:
    sample_sizes:
        equijoin: 25000
    equijoin:
        target_recall: 0.5

pipeline:
  steps:
    - name: extract_reactions
      input: biodex_sample
      operations:
        - extract_reactions

    - name: match_reactions
      operations:
        - match_reactions:
            left: extract_reactions
            right: biodex_terms

  output:
    type: file
    path: "matched_reactions.json"
"""

# Create a temporary YAML file with the pipeline configuration
with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as temp_file:
    temp_file.write(PIPELINE)
    temp_file.flush()
    config_yaml = temp_file.name

# Initialize the optimizer
optimizer = Optimizer(config_yaml)

# Run the optimization
optimizer.optimize()

# Print the contents of the optimized configuration file
with open(optimizer.optimized_config_path, "r") as optimized_file:
    print("Contents of the optimized configuration file:")
    print(optimized_file.read())


print("Optimization completed successfully.")
