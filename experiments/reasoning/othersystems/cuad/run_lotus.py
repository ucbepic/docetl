import pandas as pd
import json
import lotus
from lotus.models import LM
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Add the evaluation utils to the path
from experiments.reasoning.evaluation.utils import get_evaluate_func

# Configure Lotus with the model
lm = LM(model="azure/gpt-4o-mini", max_tokens=10000)
lotus.settings.configure(lm=lm)

def load_cuad_data(file_path):
    """Load CUAD data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def parse_json_output(output_str):
    """Parse the JSON output from the LLM response"""
    try:
        # Try to parse as JSON directly
        return json.loads(output_str)
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON from markdown code blocks
        import re
        # Look for JSON within ```json ``` code blocks
        json_match = re.search(r'```json\s*(.*?)\s*```', output_str, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to extract any JSON object from the response
        json_match = re.search(r'\{.*\}', output_str, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # If all else fails, return None
        print(f"Failed to parse JSON: {output_str}")
        return None

def main():
    datasets = [
        {"name": "train", "path": "experiments/reasoning/data/train/cuad.json"},
        {"name": "test", "path": "experiments/reasoning/data/test/cuad.json"}
    ]
    
    eval_results = []
    
    for dataset in datasets:
        lm.reset_stats()
        print(f"\n{'='*50}")
        print(f"Processing {dataset['name']} dataset...")
        print(f"{'='*50}")
        
        # Load the CUAD dataset
        df = load_cuad_data(dataset['path'])
        
        # Use the exact same prompt as in the YAML, but add JSON formatting instructions
        user_instruction = """Given the following contract document:

{document}

Extract the text spans (if they exist) for each of the following categories. If a category is not present or cannot be determined, return an empty string. If there are multiple text spans for a category, return them as a comma-separated list of text spans.

1. Document Name: The name of the contract
2. Parties: The two or more parties who signed the contract
3. Agreement Date: The date of the contract
4. Effective Date: The date when the contract is effective
5. Expiration Date: On what date will the contract's initial term expire?
6. Renewal Term: What is the renewal term after the initial term expires? This includes automatic extensions and unilateral extensions with prior notice.
7. Notice to Terminate Renewal: What is the notice period required to terminate renewal?
8. Governing Law: Which state/country's law governs the interpretation of the contract?
9. Most Favored Nation: Is there a clause that if a third party gets better terms on the licensing or sale of technology/goods/services described in the contract, the buyer of such technology/goods/services under the contract shall be entitled to those better terms?
10. Non-Compete: Is there a restriction on the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector?
11. Exclusivity: Is there an exclusive dealing commitment with the counterparty? This includes a commitment to procure all "requirements" from one party of certain technology, goods, or services or a prohibition on licensing or selling technology, goods or services to third parties, or a prohibition on collaborating or working with other parties), whether during the contract or after the contract ends (or both).
12. No-Solicit of Customers: Is a party restricted from contracting or soliciting customers or partners of the counterparty, whether during the contract or after the contract ends (or both)?
13. Competitive Restriction Exception: This category includes the exceptions or carveouts to Non-Compete, Exclusivity and No-Solicit of Customers above.
14. No-Solicit of Employees: Is there a restriction on a party's soliciting or hiring employees and/or contractors from the counterparty, whether during the contract or after the contract ends (or both)?
15. Non-Disparagement: Is there a requirement on a party not to disparage the counterparty?
16. Termination for Convenience: Can a party terminate this contract without cause (solely by giving a notice and allowing a waiting period to expire)?
17. Right of First Refusal, Offer or Negotiation: Is there a clause granting one party a right of first refusal, right of first offer or right of first negotiation to purchase, license, market, or distribute equity interest, technology, assets, products or services?
18. Change of Control: Does one party have the right to terminate or is consent or notice required of the counterparty if such party undergoes a change of control, such as a merger, stock sale, transfer of all or substantially all of its assets or business, or assignment by operation of law?
19. Anti-Assignment: Is consent or notice required of a party if the contract is assigned to a third party?
20. Revenue/Profit Sharing: Is one party required to share revenue or profit with the counterparty for any technology, goods, or services?
21. Price Restriction: Is there a restriction on the ability of a party to raise or reduce prices of technology, goods, or services provided?
22. Minimum Commitment: Is there a minimum order size or minimum amount or units per-time period that one party must buy from the counterparty under the contract?
23. Volume Restriction: Is there a fee increase or consent requirement, etc. if one party's use of the product/services exceeds certain threshold?
24. IP Ownership Assignment: Does intellectual property created by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events?
25. Joint IP Ownership: Is there any clause providing for joint or shared ownership of intellectual property between the parties to the contract?
26. License Grant: Does the contract contain a license granted by one party to its counterparty?
27. Non-Transferable License: Does the contract limit the ability of a party to transfer the license being granted to a third party?
28. Affiliate IP License-Licensor: Does the contract contain a license grant by affiliates of the licensor or that includes intellectual property of affiliates of the licensor?
29. Affiliate IP License-Licensee: Does the contract contain a license grant to a licensee (incl. sublicensor) and the affiliates of such licensee/sublicensor?
30. Unlimited/All-You-Can-Eat License: Is there a clause granting one party an "enterprise," "all you can eat" or unlimited usage license?
31. Irrevocable or Perpetual License: Does the contract contain a license grant that is irrevocable or perpetual?
32. Source Code Escrow: Is one party required to deposit its source code into escrow with a third party, which can be released to the counterparty upon the occurrence of certain events (bankruptcy, insolvency, etc.)?
33. Post-Termination Services: Is a party subject to obligations after the termination or expiration of a contract, including any post-termination transition, payment, transfer of IP, wind-down, last-buy, or similar commitments?
34. Audit Rights: Does a party have the right to audit the books, records, or physical locations of the counterparty to ensure compliance with the contract?
35. Uncapped Liability: Is a party's liability uncapped upon the breach of its obligation in the contract? This also includes uncap liability for a particular type of breach such as IP infringement or breach of confidentiality obligation.
36. Cap on Liability: Does the contract include a cap on liability upon the breach of a party's obligation? This includes time limitation for the counterparty to bring claims or maximum amount for recovery.
37. Liquidated Damages: Does the contract contain a clause that would award either party liquidated damages for breach or a fee upon the termination of a contract (termination fee)?
38. Warranty Duration: What is the duration of any warranty against defects or errors in technology, products, or services provided under the contract?
39. Insurance: Is there a requirement for insurance that must be maintained by one party for the benefit of the counterparty?
40. Covenant Not to Sue: Is a party restricted from contesting the validity of the counterparty's ownership of intellectual property or otherwise bringing a claim against the counterparty for matters unrelated to the contract?
41. Third Party Beneficiary: Is there a non-contracting party who is a beneficiary to some or all of the clauses in the contract and therefore can enforce its rights against a contracting party?

Please return your response as a JSON object with this structure:
{{
  "clauses": [
    {{"clause_type": "Document Name", "text_span": "extracted text or empty string"}},
    {{"clause_type": "Parties", "text_span": "extracted text or empty string"}},
    ... (continue for all 41 categories)
  ]
}}"""

        # Apply the semantic map operation
        print(f"Processing {len(df)} contracts with Lotus...")
        df_result = df.sem_map(user_instruction, safe_mode=True)
        
        # Find the output column created by sem_map
        output_col = [col for col in df_result.columns if col not in df.columns][0]
        
        # Parse the JSON outputs in the new column
        print("Parsing JSON outputs...")
        df_result[output_col] = df_result[output_col].apply(parse_json_output)
        
        # Rename the output column to 'clauses' for evaluation
        df_result = df_result.rename(columns={output_col: 'clauses'})
        
        # Save results as JSON
        output_path = f"experiments/reasoning/othersystems/cuad/lotus_{dataset['name']}.json"
        print(f"Saving results to {output_path}...")
        
        # Convert DataFrame to list of dictionaries and save
        results_list = df_result.to_dict('records')
        with open(output_path, 'w') as f:
            json.dump(results_list, f, indent=2)
        
        print(f"Processing complete! Results saved to {output_path}")
        print(f"Processed {len(results_list)} documents")
        
        # Run evaluation using the utils function
        print(f"\n🧪 Running CUAD evaluation for {dataset['name']}...")
        try:
            eval_func = get_evaluate_func("cuad")
            metrics = eval_func("lotus", output_path)
            
            cost = lm.stats.virtual_usage.total_cost
            
            print(f"\n📊 Evaluation Results for {dataset['name']}:")
            print(f"   Average Precision: {metrics['avg_precision']:.4f}")
            print(f"   Average Recall: {metrics['avg_recall']:.4f}")
            print(f"   Average F1: {metrics['avg_f1']:.4f}")
            print(f"   Cost: {cost:.4f}")
            
            # Add evaluation results for this dataset
            eval_results.append({
                "file": output_path,
                "precision": metrics['avg_precision'],
                "recall": metrics['avg_recall'],
                "f1": metrics['avg_f1'],
                "cost": cost
            })
            
        except Exception as e:
            print(f"⚠️  Evaluation failed for {dataset['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined evaluation metrics
    if eval_results:
        eval_output_path = "experiments/reasoning/othersystems/cuad/lotus_evaluation.json"
        with open(eval_output_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\n📈 Combined evaluation metrics saved to: {eval_output_path}")
        print(f"📊 Total datasets processed: {len(eval_results)}")

if __name__ == "__main__":
    main()
    lotus.settings.lm.print_total_usage()
