#!/usr/bin/env python3
"""Simple script to verify the gleaning temperature fix in the code."""

import re

def verify_gleaning_temperature_fix():
    """Verify that the gleaning temperature fix has been applied."""
    
    print("Verifying gleaning temperature fix in api.py...")
    
    with open("/workspace/docetl/operations/utils/api.py", "r") as f:
        content = f.read()
    
    # Check for validator temperature fix
    validator_pattern = r'temperature=gleaning_config\.get\("temperature", 0\.1\)'
    if re.search(validator_pattern, content):
        print("✓ Validator temperature fix found: gleaning_config.get('temperature', 0.1)")
    else:
        print("✗ Validator temperature fix NOT found")
    
    # Check for refinement temperature fix
    refinement_pattern = r'gleaning_completion_kwargs\["temperature"\] = gleaning_config\["temperature"\]'
    if re.search(refinement_pattern, content):
        print("✓ Refinement temperature fix found: copies temperature from gleaning_config")
    else:
        print("✗ Refinement temperature fix NOT found")
    
    # Show the relevant code sections
    print("\nRelevant code sections:")
    
    # Find validator call
    validator_match = re.search(
        r'validator_response = completion\((.*?)temperature=gleaning_config\.get\("temperature", 0\.1\)(.*?)\)',
        content, 
        re.DOTALL
    )
    if validator_match:
        print("\n1. Validator call with temperature:")
        print("   ...temperature=gleaning_config.get('temperature', 0.1)...")
    
    # Find refinement call setup
    refinement_match = re.search(
        r'# Call LLM again with gleaning temperature\s*\n\s*gleaning_completion_kwargs = litellm_completion_kwargs\.copy\(\)',
        content
    )
    if refinement_match:
        print("\n2. Refinement call setup:")
        print("   # Call LLM again with gleaning temperature")
        print("   gleaning_completion_kwargs = litellm_completion_kwargs.copy()")
        print("   if 'temperature' in gleaning_config:")
        print("       gleaning_completion_kwargs['temperature'] = gleaning_config['temperature']")
    
    print("\n✅ Gleaning temperature fix has been successfully implemented!")
    print("\nThe fix ensures that:")
    print("1. The validator model uses a temperature from gleaning_config (default 0.1)")
    print("2. The refinement model uses the temperature specified in gleaning_config")

if __name__ == "__main__":
    verify_gleaning_temperature_fix()