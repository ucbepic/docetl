import pandas as pd
import json
import lotus
from lotus.models import LM
import sys
from pathlib import Path
import re
import time

from dotenv import load_dotenv
load_dotenv()

# Add the evaluation utils to the path
from experiments.reasoning.evaluation.utils import get_evaluate_func

# Configure Lotus with the model
lm = LM(model="azure/gpt-4o-mini", max_tokens=10000)
lotus.settings.configure(lm=lm)

def load_medec_data(file_path):
    """Load medec data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def parse_medec_output(output_str):
    """Parse the output from the LLM response for medec format"""
    try:
        # Initialize default values
        is_error = False
        error_sentence = ""
        corrected_sentence = ""
        
        # Look for the specific format patterns
        is_error_match = re.search(r'is_error\s*=\s*(true|false)', output_str, re.IGNORECASE)
        if is_error_match:
            is_error = is_error_match.group(1).lower() == 'true'
        
        error_sentence_match = re.search(r'error_sentence\s*=\s*"([^"]*)"', output_str)
        if error_sentence_match:
            error_sentence = error_sentence_match.group(1)
        
        corrected_sentence_match = re.search(r'corrected_sentence\s*=\s*"([^"]*)"', output_str)
        if corrected_sentence_match:
            corrected_sentence = corrected_sentence_match.group(1)
        
        # Return structured output
        return {
            "is_error": is_error,
            "error_sentence": error_sentence,
            "corrected_sentence": corrected_sentence
        }
        
    except Exception as e:
        print(f"Failed to parse output: {output_str}, Error: {e}")
        return {
            "is_error": False,
            "error_sentence": "",
            "corrected_sentence": ""
        }

def main():

    start_time = time.time()

    datasets = [
        {"name": "train", "path": "experiments/reasoning/data/train/medec.json"},
        {"name": "test", "path": "experiments/reasoning/data/test/medec.json"}
    ]
    
    eval_results = []
    
    for dataset in datasets:
        lm.reset_stats()
        print(f"\n{'='*50}")
        print(f"Processing {dataset['name']} dataset...")
        print(f"{'='*50}")
        
        # Load the medec dataset
        df = load_medec_data(dataset['path'])
        
        # Use the exact same prompt as in the YAML
        user_instruction = """The following is a medical narrative about a patient. You are a skilled medical doctor reviewing the clinical text.
The text is either correct or contains one error. The text has one sentence per line. Each line starts with the sentence ID,
followed by a space, then the sentence to check. Check every sentence of the text. If the text is correct, return the
following output: is_error=false, error_sentence="", corrected_sentence="". If the text has a medical error related to treatment,
management, cause, or diagnosis, return the sentence id of the sentence containing the error, the original sentence, and a corrected version of the sentence.
Finding and correcting the error requires medical knowledge and reasoning.

Here is an example:
0 A 35-year-old woman presents to her physician with a complaint of pain and stiffness in her hands.
1 She says that the pain began 6 weeks ago a few days after she had gotten over a minor upper respiratory infection.
2 She has no history of trauma.
3 She has no significant past medical history.
4 On examination, there is swelling and tenderness of the metacarpophalangeal and proximal interphalangeal joints of both hands.
5 The wrists are also swollen and tender.
6 There is no evidence of joint deformity.
7 Laboratory studies show a positive rheumatoid factor and elevated erythrocyte sedimentation rate.
8 The C-reactive protein is elevated.
9 Bilateral radiographs of the hands demonstrate mild periarticular osteopenia around the left fifth metacarpophalangeal joint.
10 Methotrexate is given.

In this example, the error is in sentence 10: "Methotrexate is given." The correction is: "Prednisone is given."
The output is:
is_error=true, error_sentence="Methotrexate is given.", corrected_sentence="Prednisone is given."

End of Example.

Now review the following medical text:
{Text}

Return your analysis in the following format:
- If no error: is_error=false, error_sentence="", corrected_sentence=""
- If error found: is_error=true, error_sentence="[original sentence with error]", corrected_sentence="[corrected sentence]"
"""

        # Apply the semantic map operation
        print(f"Processing {len(df)} medical texts with Lotus...")
        df_result = df.sem_map(user_instruction, safe_mode=True)
        
        # Find the output column created by sem_map
        output_col = [col for col in df_result.columns if col not in df.columns][0]
        
        # Parse the outputs in the new column
        print("Parsing medec outputs...")
        parsed_outputs = df_result[output_col].apply(parse_medec_output)
        
        # Extract the parsed fields directly into the dataframe
        df_result['is_error'] = parsed_outputs.apply(lambda x: x['is_error'])
        df_result['error_sentence'] = parsed_outputs.apply(lambda x: x['error_sentence'])
        df_result['corrected_sentence'] = parsed_outputs.apply(lambda x: x['corrected_sentence'])
        
        # Drop the original output column since we've extracted the fields
        df_result = df_result.drop(columns=[output_col])
        
        # Save results as JSON
        output_path = f"experiments/reasoning/othersystems/medec/lotus_{dataset['name']}.json"
        print(f"Saving results to {output_path}...")
        
        # Convert DataFrame to list of dictionaries and save
        results_list = df_result.to_dict('records')
        with open(output_path, 'w') as f:
            json.dump(results_list, f, indent=2)
        
        print(f"Processing complete! Results saved to {output_path}")
        print(f"Processed {len(results_list)} medical texts")

        execution_time = time.time() - start_time
        
        # Run evaluation using the utils function
        print(f"\n🧪 Running medec evaluation for {dataset['name']}...")
        try:
            eval_func = get_evaluate_func("medec")
            metrics = eval_func("lotus", output_path)
            
            cost = lm.stats.virtual_usage.total_cost
            
            print(f"\n📊 Evaluation Results for {dataset['name']}:")
            print(f"   Combined Score: {metrics['combined_score']:.4f}")
            print(f"   Error Flag Accuracy: {metrics['error_flag_accuracy']:.4f}")
            print(f"   Avg Error Sentence Jaccard: {metrics['avg_error_sentence_jaccard']:.4f}")
            print(f"   Avg Corrected Sentence Jaccard: {metrics['avg_corrected_sentence_jaccard']:.4f}")
            print(f"   Cost: {cost:.4f}")
            
            # Add evaluation results for this dataset
            eval_results.append({
                "file": output_path,
                "combined_score": metrics['combined_score'],
                "error_flag_accuracy": metrics['error_flag_accuracy'],
                "avg_error_sentence_jaccard": metrics['avg_error_sentence_jaccard'],
                "avg_corrected_sentence_jaccard": metrics['avg_corrected_sentence_jaccard'],
                "total_cases": metrics['total_cases'],
                "num_error_cases": metrics['num_error_cases'],
                "num_corrected_cases": metrics['num_corrected_cases'],
                "cost": cost,
                "execution_time": execution_time
            })
            
        except Exception as e:
            print(f"⚠️  Evaluation failed for {dataset['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined evaluation metrics
    if eval_results:
        eval_output_path = "experiments/reasoning/othersystems/medec/lotus_evaluation.json"
        with open(eval_output_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\n📈 Combined evaluation metrics saved to: {eval_output_path}")
        print(f"📊 Total datasets processed: {len(eval_results)}")

if __name__ == "__main__":
    main()
    lotus.settings.lm.print_total_usage()
