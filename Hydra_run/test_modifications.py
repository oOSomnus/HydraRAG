#!/usr/bin/env python3
"""
Test script to verify the HydraRAG Wikidata-only modifications
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_wikidata_api():
    """Test the Wikidata API client"""
    print("Testing Wikidata API client...")

    from wikidata_api_client import WikidataAPIClient
    client = WikidataAPIClient()

    # Test getting labels for a known entity
    labels = client.get_entity_labels('Q42')  # Douglas Adams
    print(f"Labels for Q42 (Douglas Adams): {labels}")

    assert len(labels) > 0 and "Douglas Adams" in labels[0], f"Expected Douglas Adams in labels, got {labels}"
    print("✓ Wikidata API test passed")

def test_compatibility_functions():
    """Test compatibility wrapper functions"""
    print("Testing compatibility wrapper functions...")

    from wikidata_api_client import get_entity_name_or_type
    name = get_entity_name_or_type('Q42')
    print(f"Entity name for Q42: {name}")

    assert name == "Douglas Adams", f"Expected Douglas Adams, got {name}"
    print("✓ Compatibility functions test passed")

def test_argparse_changes():
    """Test that the command line argument changes are effective"""
    print("Testing argument parser changes...")

    # Read the hydra_main.py file to check the build_parser function
    with open('hydra_main.py', 'r') as f:
        content = f.read()

    # Check that --no-freebase is commented out/removed
    # Look for the actual commented lines
    if '#     "--no-freebase",' in content:
        print("✓ --no-freebase option has been commented out/disabled")
    elif '--no-freebase' in content:
        print("✗ --no-freebase option still exists in active code")
    else:
        print("✓ --no-freebase option has been removed")

    # Check that --no-wikikg still exists
    if '--no-wikikg' in content and '# --no-wikikg' not in content:
        print("✓ --no-wikikg option still exists")
    else:
        print("✗ --no-wikikg option missing or commented out")

def test_imports():
    """Test that imports work without Freebase dependencies"""
    print("Testing imports...")

    # We can't fully import utilts due to model loading, but we can check the import statement
    with open('utilts.py', 'r') as f:
        content = f.read()

    # Check that freebase_func is no longer imported
    if 'from freebase_func import *' in content or 'import freebase_func' in content:
        print("✗ freebase_func still being imported")
    else:
        print("✓ freebase_func import has been removed")

    # Check that wikidata_api_client is imported
    if 'from wikidata_api_client import *' in content or 'import wikidata_api_client' in content:
        print("✓ wikidata_api_client is imported")
    else:
        print("✗ wikidata_api_client not imported")

def main():
    print("Starting HydraRAG Wikidata-only modifications test...\n")

    try:
        test_wikidata_api()
        print()

        test_compatibility_functions()
        print()

        test_argparse_changes()
        print()

        test_imports()
        print()

        print("All tests completed!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()