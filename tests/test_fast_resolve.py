import pytest
import random
import string
import time
from docetl.operations.fast_resolve import FastResolveOperation
from docetl.operations.resolve import ResolveOperation


@pytest.fixture
def fast_resolve_config():
    return {
        "name": "name_email_resolver",
        "type": "fast_resolve",
        "blocking_threshold": 0.8,
        "blocking_keys": ["name", "email"],
        "comparison_prompt": """Compare these two entries and determine if they refer to the same person:
            Person 1: {{ input1.name }} {{ input1.email }}
            Person 2: {{ input2.name }} {{ input2.email }}
            Return true if they match, false otherwise.""",
        "resolution_prompt": "Given these similar entries, determine the canonical form: {{ inputs }}",
        "output": {
            "schema": {
                "name": "string",
                "email": "string"
            }
        },
        "embedding_model": "text-embedding-3-small",
        "comparison_model": "azure/gpt-4o-mini",
        "resolution_model": "azure/gpt-4o-mini"
    }


def generate_large_dataset(num_base_records=100):
    """Generate a very large dataset with intentional duplicates and transitive relationships.
    
    Example of transitivity:
    - John Doe <-> Johnny Doe <-> J. Doe (all same email)
    - Multiple email variations for same person
    - Name variations that chain together
    """
    
    # Base data to create variations from
    first_names = ['John', 'Michael', 'William', 'James', 'David', 'Robert', 'Thomas', 'Christopher']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis']
    domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'company.com']
    
    data = []
    
    # Create base records with intentional relationships
    for _ in range(num_base_records):
        first = random.choice(first_names)
        last = random.choice(last_names)
        domain = random.choice(domains)
        
        # Create base email variations for this person
        email_variations = [
            f"{first.lower()}.{last.lower()}@{domain}",
            f"{first.lower()[0]}{last.lower()}@{domain}",
            f"{first.lower()}{last.lower()[0]}@{domain}",
            f"{first.lower()}_{last.lower()}@{domain}"
        ]
        
        # Create name variations that chain together
        name_variations = [
            f"{first} {last}",  # Standard
            f"{first}y {last}",  # Diminutive
            f"{first[0]}. {last}",  # Initial
            f"{first} {last[0]}.",  # Last initial
            f"{first[0]}. {last[0]}.",  # Both initials
        ]
        
        # Add middle initials to some variations
        middle_initials = random.sample(string.ascii_uppercase, 2)
        name_variations.extend([
            f"{first} {mi}. {last}" for mi in middle_initials
        ])
        
        # Create multiple records with combinations of name/email variations
        # This ensures transitive relationships
        for name in name_variations:
            # Use same email for some variations to create strong links
            primary_email = random.choice(email_variations)
            data.append({"name": name, "email": primary_email})
            
            # Add some variations with different emails
            if random.random() < 0.3:
                data.append({"name": name, "email": random.choice(email_variations)})
            
            # Add typo variations
            if random.random() < 0.2:
                typo_name = name.replace('i', 'y') if 'i' in name else name + 'n'
                data.append({"name": typo_name, "email": primary_email})
        
        # Add some completely different email domains for same person
        alt_domain = random.choice([d for d in domains if d != domain])
        alt_email = f"{first.lower()}.{last.lower()}@{alt_domain}"
        data.append({"name": random.choice(name_variations), "email": alt_email})
    
    # Shuffle the dataset
    random.shuffle(data)
    
    # Print some statistics about the dataset
    print(f"\nGenerated Dataset Statistics:")
    print(f"Total records: {len(data)}")
    print(f"Unique names: {len(set(r['name'] for r in data))}")
    print(f"Unique emails: {len(set(r['email'] for r in data))}")
    print(f"Average variations per base record: {len(data) / num_base_records:.1f}")
    
    return data


@pytest.fixture
def fast_resolve_sample_data():
    # Set random seed for reproducibility
    random.seed(42)
    return generate_large_dataset()


def dont_do_test_fast_resolve_operation(
    fast_resolve_config, default_model, fast_resolve_sample_data, api_wrapper
):
    
    distinct_names = set(result["name"] for result in fast_resolve_sample_data)
    distinct_emails = set(result["email"] for result in fast_resolve_sample_data)
    print(f"Distinct names in input: {len(distinct_names)}")
    print(f"Distinct emails in input: {len(distinct_emails)}")
    
    operation = FastResolveOperation(
        api_wrapper, fast_resolve_config, default_model, 256
    )
    results, cost = operation.execute(fast_resolve_sample_data)

    # Calculate and print some statistics
    input_count = len(fast_resolve_sample_data)
    output_count = len(results)
    distinct_output_names = set(result["name"] for result in results)
    distinct_output_emails = set(result["email"] for result in results)
    
    print(f"\nTest Statistics:")
    print(f"Input records: {input_count}")
    print(f"Output records: {output_count}")
    print(f"Distinct names in output: {len(distinct_output_names)}")
    print(f"Distinct emails in output: {len(distinct_output_emails)}")
    print(f"Reduction ratio: {(input_count - output_count) / input_count:.2%}")
    print(f"Total cost: {cost}")

    # Assertions
    assert len(distinct_names) < len(fast_resolve_sample_data)
    assert output_count == input_count
    assert cost > 0


def test_fast_resolve_operation_empty_input(
    fast_resolve_config, default_model, max_threads, api_wrapper
):
    operation = FastResolveOperation(
        api_wrapper, fast_resolve_config, default_model, max_threads
    )
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0


@pytest.fixture
def resolve_config():
    return {
        "name": "name_email_resolver",
        "type": "resolve",
        "blocking_keys": ["name", "email"],
        "blocking_threshold": 0.8,
        "comparison_prompt": "Compare these two entries and determine if they refer to the same person: Person 1: {{ input1 }} Person 2: {{ input2 }} Return true if they match, false otherwise.",
        "resolution_prompt": "Given these similar entries, determine the canonical form: {{ inputs }}",
        "output": {"schema": {"name": "string", "email": "string"}},
        "embedding_model": "text-embedding-3-small",
        "comparison_model": "gpt-4o-mini",
        "resolution_model": "gpt-4o-mini"
    }


def test_compare_resolve_performance(
    fast_resolve_config, resolve_config, default_model, api_wrapper
):
    """Compare performance between FastResolve and regular Resolve operations."""
    
    # Generate a large dataset specifically for this test
    large_dataset = generate_large_dataset()
    
    # Use a smaller subset for the regular resolve to keep test duration reasonable
    sample_size = min(len(large_dataset) // 4, 1000)
    sample_data = random.sample(large_dataset, sample_size)
    
    print(f"\nTesting with {len(large_dataset)} records for FastResolve")
    print(f"Testing with {len(sample_data)} records for regular Resolve")
    
    # Test FastResolve with full dataset
    start_time = time.time()
    fast_operation = FastResolveOperation(
        api_wrapper, fast_resolve_config, default_model, 256
    )
    fast_results, fast_cost = fast_operation.execute(large_dataset)
    fast_time = time.time() - start_time

    # Test regular Resolve with sample
    start_time = time.time()
    regular_operation = ResolveOperation(
        api_wrapper, resolve_config, default_model, 256
    )
    regular_results, regular_cost = regular_operation.execute(sample_data)
    regular_time = time.time() - start_time

    # Scale up regular metrics for fair comparison
    scale_factor = len(large_dataset) / len(sample_data)
    regular_time_scaled = regular_time * scale_factor
    regular_cost_scaled = regular_cost * scale_factor

    # Print performance comparison
    print("\nPerformance Comparison (scaled to same dataset size):")
    print(f"FastResolve Time: {fast_time:.2f} seconds")
    print(f"Regular Resolve Time (scaled): {regular_time_scaled:.2f} seconds")
    print(f"FastResolve Cost: ${fast_cost:.4f}")
    print(f"Regular Resolve Cost (scaled): ${regular_cost_scaled:.4f}")
    print(f"Speed Improvement: {(regular_time_scaled - fast_time) / regular_time_scaled:.1%}")
    print(f"Cost Savings: {(regular_cost_scaled - fast_cost) / regular_cost_scaled:.1%}")

    # Assertions
    assert fast_time < regular_time_scaled, "FastResolve should be faster than regular Resolve"
    assert fast_cost < regular_cost_scaled, "FastResolve should be more cost-effective"