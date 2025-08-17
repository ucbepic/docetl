#!/usr/bin/env python3
"""
Quick test to verify operations work in Modal environment.
"""

import modal
from experiments.reasoning.utils import app, volume, VOLUME_MOUNT_PATH, image

@app.function(image=image, secrets=[modal.Secret.from_dotenv()], volumes={VOLUME_MOUNT_PATH: volume}, timeout=60 * 2)
def test_operations_modal():
    """Test if all operations are available in Modal environment."""
    print("🔍 TESTING OPERATIONS IN MODAL")
    print("=" * 50)
    
    try:
        from docetl.operations import get_operations, get_operation
        available_ops = get_operations()
        print(f"Total operations available: {len(available_ops)}")
        print(f"Operations: {sorted(available_ops.keys())}")
        
        # Test specific operations that were problematic
        test_ops = ['parallel_map', 'topk', 'link_resolve']
        
        for op_name in test_ops:
            try:
                if op_name in available_ops:
                    print(f"✅ {op_name} found in get_operations()")
                else:
                    print(f"❌ {op_name} NOT found in get_operations()")
                
                # Test get_operation function
                op_class = get_operation(op_name)
                print(f"✅ get_operation('{op_name}') successful: {op_class}")
                
            except Exception as e:
                print(f"❌ {op_name} failed: {e}")
        
        return {"status": "success", "total_ops": len(available_ops)}
        
    except Exception as e:
        print(f"❌ Overall test error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}

@app.local_entrypoint()
def test_modal_ops():
    """Local entrypoint to test operations in Modal."""
    result = test_operations_modal.remote()
    print(f"\nTest result: {result}")

if __name__ == "__main__":
    test_modal_ops()