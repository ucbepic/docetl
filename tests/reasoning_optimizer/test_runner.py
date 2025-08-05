#!/usr/bin/env python3
"""
Test runner for all directive testing.

This module provides a comprehensive test runner that discovers and executes
tests for all directives in the reasoning optimizer.
"""

from typing import Dict, List
from datetime import datetime

from docetl.reasoning_optimizer.directives import (
    ChainingDirective, 
    GleaningDirective,
    ReduceGleaningDirective,
    ReduceChainingDirective,
    ChangeModelDirective,
    DocSummarizationDirective,
    IsolatingSubtasksDirective,
    DocCompressionDirective,
    DeterministicDocCompressionDirective,
    OperatorFusionDirective,
    DocumentChunkingDirective,
    ChunkHeaderSummaryDirective,
    TakeHeadTailDirective,
    TestResult
)

def run_all_directive_tests(agent_llm: str = "gpt-4.1") -> Dict[str, List[TestResult]]:
    """
    Run tests for all directives and generate comprehensive report.
    
    Args:
        agent_llm: The LLM model to use for testing
        
    Returns:
        Dictionary mapping directive names to their test results
    """
    print("=" * 70)
    print("DOCETL REASONING OPTIMIZER - DIRECTIVE TEST SUITE")
    print("=" * 70)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using LLM: {agent_llm}")
    print()
    
    # Initialize all directives
    directives = [
        ChainingDirective(),
        GleaningDirective(),
        ReduceGleaningDirective(),
        ReduceChainingDirective(),
        ChangeModelDirective(),
        DocSummarizationDirective(),
        IsolatingSubtasksDirective(),
        DocCompressionDirective(),
        DeterministicDocCompressionDirective(),
        OperatorFusionDirective(),
        DocumentChunkingDirective(), 
        ChunkHeaderSummaryDirective(),
        TakeHeadTailDirective(),
        ReduceChainingDirective(),
    ]
    
    all_results = {}
    total_tests = 0
    total_passed = 0
    
    for directive in directives:
        print(f"\n{'='*50}")
        print(f"Testing {directive.name.upper()} Directive")
        print(f"{'='*50}")
        print(f"Description: {directive.nl_description}")
        print(f"When to use: {directive.when_to_use}")
        print()
        
        if not directive.test_cases:
            print(f"⚠️  No test cases defined for {directive.name}")
            all_results[directive.name] = []
            continue
            
        print(f"Running {len(directive.test_cases)} test cases...")
        print()
        
        try:
            results = directive.run_tests(agent_llm=agent_llm)
            all_results[directive.name] = results
            
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            total_tests += total
            total_passed += passed
            
            print(f"Results: {passed}/{total} tests passed")
            
            if passed == total:
                print("🎉 All tests passed!")
            elif passed > 0:
                print("⚠️  Some tests failed")
            else:
                print("❌ All tests failed")
            
            print("\nDetailed Results:")
            print("-" * 40)
            
            for result in results:
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"{status} {result.test_name}")
                print(f"   Reason: {result.reason}")
                if result.execution_error:
                    print(f"   Error: {result.execution_error}")
                    print(f"   Actual output: {result.actual_output}")
                print()
                
        except Exception as e:
            print(f"❌ Failed to run tests for {directive.name}: {str(e)}")
            all_results[directive.name] = []
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUITE SUMMARY")
    print("=" * 70)
    print(f"Total tests run: {total_tests}")
    print(f"Total passed: {total_passed}")
    print(f"Total failed: {total_tests - total_passed}")
    print(f"Success rate: {(total_passed/total_tests)*100:.1f}%" if total_tests > 0 else "No tests run")
    
    # Per-directive summary
    print("\nPer-directive results:")
    for directive_name, results in all_results.items():
        if results:
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            print(f"  {directive_name}: {passed}/{total} ({(passed/total)*100:.1f}%)")
        else:
            print(f"  {directive_name}: No tests")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return all_results

def run_specific_directive_test(directive_name: str, agent_llm: str = "gpt-4o-mini") -> List[TestResult]:
    """
    Run tests for a specific directive.
    
    Args:
        directive_name: Name of the directive to test
        agent_llm: The LLM model to use for testing
        
    Returns:
        List of test results for the directive
    """
    directive_map = {
        "chaining": ChainingDirective(),
        "gleaning": GleaningDirective(),
        "reduce_gleaning": ReduceGleaningDirective(),
        "change_model": ChangeModelDirective(),
        "doc_summarization": DocSummarizationDirective(),
        "isolating_subtasks": IsolatingSubtasksDirective(),
        "doc_compression": DocCompressionDirective(),
        "deterministic_doc_compression": DeterministicDocCompressionDirective(),
        "operator_fusion": OperatorFusionDirective(),
        "doc_chunking": DocumentChunkingDirective(),
        "chunk_header_summary": ChunkHeaderSummaryDirective(),
        "take_head_tail": TakeHeadTailDirective(),
        "reduce_chaining": ReduceChainingDirective()
    }
    
    if directive_name.lower() not in directive_map:
        print(f"❌ Unknown directive: {directive_name}")
        print(f"Available directives: {list(directive_map.keys())}")
        return []
    
    directive = directive_map[directive_name.lower()]
    
    print(f"Testing {directive.name} directive...")
    print(f"Description: {directive.nl_description}")
    print()
    
    if not directive.test_cases:
        print(f"⚠️  No test cases defined for {directive.name}")
        return []
    
    results = directive.run_tests(agent_llm=agent_llm)
    
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    print(f"Results: {passed}/{total} tests passed")
    print()
    
    for result in results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"{status} {result.test_name}")
        print(f"   Reason: {result.reason}")
        print(f"   Actual output: {result.actual_output}")
        if result.execution_error:
            print(f"   Error: {result.execution_error}")
            print(f"   Actual output: {result.actual_output}")
        print()
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run directive tests")
    parser.add_argument("--directive", "-d", type=str, help="Run tests for specific directive")
    parser.add_argument("--model", "-m", type=str, default="gpt-4.1", help="LLM model to use for testing")
    
    args = parser.parse_args()
    
    if args.directive:
        run_specific_directive_test(args.directive, args.model)
    else:
        run_all_directive_tests(args.model)