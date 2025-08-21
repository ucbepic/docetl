#!/usr/bin/env python3
"""
Test script to verify Gemini models work with litellm.
"""
import os
import litellm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test models (based on what I can see in your updates)
GEMINI_MODELS = [
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"
]

def test_gemini_model(model_name):
    """Test a single Gemini model with a simple completion."""
    print(f"\n🧪 Testing {model_name}...")
    
    try:
        # Simple test message
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Respond concisely."
            },
            {
                "role": "user", 
                "content": "What is 2+2? Answer with just the number."
            }
        ]
        
        # Make the API call
        response = litellm.completion(
            model=model_name,
            messages=messages,
            temperature=0.1
        )
        print(response)
        
        # Extract response
        answer = response.choices[0].message.content.strip()
        print(f"✅ {model_name}: Response = '{answer}'")
        
        # Check if it's a reasonable answer
        if "4" in answer:
            print(f"✅ {model_name}: Correct answer detected!")
            return True
        else:
            print(f"⚠️  {model_name}: Unexpected answer, but model responded")
            return True
            
    except Exception as e:
        print(f"❌ {model_name}: Error = {str(e)}")
        return False

def test_all_gemini_models():
    """Test all Gemini models."""
    print("🚀 Testing Gemini models with litellm...")
    print("=" * 50)
    
    # Check for required environment variables
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No GOOGLE_API_KEY or GEMINI_API_KEY found in environment!")
        print("Please set one of these environment variables.")
        return False
    
    print(f"🔑 API Key found: {api_key[:8]}...")
    
    results = {}
    for model in GEMINI_MODELS:
        model = "gemini/" + model
        results[model] = test_gemini_model(model)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    working_models = [model for model, success in results.items() if success]
    failed_models = [model for model, success in results.items() if not success]
    
    print(f"✅ Working models ({len(working_models)}):")
    for model in working_models:
        print(f"   - {model}")
    
    if failed_models:
        print(f"❌ Failed models ({len(failed_models)}):")
        for model in failed_models:
            print(f"   - {model}")
    
    print(f"\nSuccess rate: {len(working_models)}/{len(GEMINI_MODELS)} models")
    
    return len(working_models) > 0

if __name__ == "__main__":
    success = test_all_gemini_models()
    
    if success:
        print("\n🎉 At least one Gemini model is working!")
        print("You can now use these models in your MCTS optimization.")
    else:
        print("\n💥 No Gemini models are working. Please check:")
        print("1. Your GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
        print("2. Your internet connection")
        print("3. Gemini API quota/billing status")