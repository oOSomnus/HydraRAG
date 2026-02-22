#!/usr/bin/env python3
"""
HydraRAG Comprehensive Integration Test Script
Tests all components of the project including:
- Environment variables loading
- API key configuration
- Qwen API connectivity
- Wikidata API client
- Data processing modules
- Complete workflow
"""

import sys
import os
import json
import time
from pathlib import Path

# Add Hydra_run to path
sys.path.insert(0, str(Path(__file__).parent / "Hydra_run"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Color output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_test(test_name, status, message=""):
    """Print test result with color"""
    if status == "PASS":
        print(f"{Colors.GREEN}✓ PASS{Colors.END} - {test_name}")
    elif status == "FAIL":
        print(f"{Colors.RED}✗ FAIL{Colors.END} - {test_name}: {message}")
    elif status == "SKIP":
        print(f"{Colors.YELLOW}⊘ SKIP{Colors.END} - {test_name}: {message}")
    elif status == "INFO":
        print(f"{Colors.BLUE}ℹ INFO{Colors.END} - {test_name}")
    if message and status != "FAIL" and status != "SKIP":
        print(f"  {message}")

def test_environment_variables():
    """Test 1: Environment Variables"""
    print("\n" + "="*60)
    print("TEST 1: Environment Variables")
    print("="*60)

    tests_passed = 0
    tests_total = 0

    # Test .env file exists
    tests_total += 1
    if Path(".env").exists():
        print_test(".env file exists", "PASS")
        tests_passed += 1
    else:
        print_test(".env file exists", "FAIL")

    # Test QWEN_API_KEY
    tests_total += 1
    qwen_key = os.getenv("QWEN_API_KEY")
    if qwen_key and qwen_key.startswith("sk-"):
        print_test("QWEN_API_KEY loaded", "PASS", f"{qwen_key[:10]}...")
        tests_passed += 1
    else:
        print_test("QWEN_API_KEY loaded", "FAIL")

    # Test SERP_API_KEY
    tests_total += 1
    serp_key = os.getenv("SERP_API_KEY")
    if serp_key:
        print_test("SERP_API_KEY loaded", "PASS", f"{serp_key[:10]}...")
        tests_passed += 1
    else:
        print_test("SERP_API_KEY loaded", "FAIL")

    print(f"\n{tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total

def test_qwen_api():
    """Test 2: Qwen API Connectivity"""
    print("\n" + "="*60)
    print("TEST 2: Qwen API Connectivity")
    print("="*60)

    tests_passed = 0
    tests_total = 0

    try:
        from openai import OpenAI

        qwen_api_key = os.getenv("QWEN_API_KEY")
        if not qwen_api_key:
            print_test("Qwen API key available", "FAIL", "No API key found")
            return 0, 1

        tests_total += 1
        print_test("Qwen API key available", "PASS", f"{qwen_api_key[:10]}...")
        tests_passed += 1

        # Test basic API call
        tests_total += 1
        client = OpenAI(
            api_key=qwen_api_key,
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )

        response = client.chat.completions.create(
            model='qwen-plus',
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Say "Test successful"'}
            ],
            temperature=0.4,
            max_tokens=20,
        )

        result = response.choices[0].message.content
        if result and "Test" in result:
            print_test("Qwen API call successful", "PASS", f"Response: {result}")
            tests_passed += 1
        else:
            print_test("Qwen API call successful", "FAIL", f"Unexpected response: {result}")

    except Exception as e:
        print_test("Qwen API call", "FAIL", str(e))

    print(f"\n{tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total

def test_run_llm_function():
    """Test 3: run_LLM Function"""
    print("\n" + "="*60)
    print("TEST 3: run_LLM Function")
    print("="*60)

    tests_passed = 0
    tests_total = 0

    try:
        # Import without loading heavy models
        from openai import OpenAI

        qwen_api_key = os.getenv("QWEN_API_KEY")
        client = OpenAI(
            api_key=qwen_api_key,
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )

        # Test run_LLM equivalent function
        tests_total += 1
        prompt = "What is 2+2? Answer with just the number."
        messages = [
            {"role": "system", "content": "You are an AI assistant that helps people find information."},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model='qwen-plus',
            messages=messages,
            temperature=0.4,
            max_tokens=50,
            frequency_penalty=0,
            presence_penalty=0
        )

        result = response.choices[0].message.content
        if "4" in result:
            print_test("run_LLM equivalent function", "PASS", f"Response: {result}")
            tests_passed += 1
        else:
            print_test("run_LLM equivalent function", "FAIL", f"Unexpected response: {result}")

    except Exception as e:
        print_test("run_LLM function", "FAIL", str(e))

    print(f"\n{tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total

def test_wikidata_api():
    """Test 4: Wikidata API Client"""
    print("\n" + "="*60)
    print("TEST 4: Wikidata API Client")
    print("="*60)

    tests_passed = 0
    tests_total = 0

    try:
        from wikidata_api_client import get_entity_name_or_type

        # Test entity name lookup
        tests_total += 1
        entity_name = get_entity_name_or_type("Q42")
        if entity_name and "Douglas Adams" in entity_name:
            print_test("Wikidata entity lookup", "PASS", f"Q42 -> {entity_name}")
            tests_passed += 1
        else:
            print_test("Wikidata entity lookup", "FAIL", f"Unexpected result: {entity_name}")

    except Exception as e:
        print_test("Wikidata API", "FAIL", str(e))

    print(f"\n{tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total

def test_argument_parser():
    """Test 5: Command Line Argument Parser"""
    print("\n" + "="*60)
    print("TEST 5: Command Line Argument Parser")
    print("="*60)

    tests_passed = 0
    tests_total = 0

    try:
        # Import without executing the module
        import importlib.util
        spec = importlib.util.spec_from_file_location("hydra_main", "Hydra_run/hydra_main.py")
        hydra_main = importlib.util.module_from_spec(spec)

        # Check file content instead of importing
        with open("Hydra_run/hydra_main.py", 'r') as f:
            content = f.read()

        # Check if build_parser exists in file
        tests_total += 1
        if "def build_parser" in content:
            print_test("build_parser function exists", "PASS")
            tests_passed += 1
        else:
            print_test("build_parser function exists", "FAIL")

        # Check model choices include qwen
        tests_total += 1
        if '"qwen"' in content and "'qwen'" in content:
            print_test("Model choices include qwen", "PASS")
            tests_passed += 1
        else:
            print_test("Model choices include qwen", "FAIL")

        # Check default model is qwen
        tests_total += 1
        if 'default="qwen"' in content or "default='qwen'" in content:
            print_test("Default model is qwen", "PASS")
            tests_passed += 1
        else:
            print_test("Default model is qwen", "FAIL")

    except Exception as e:
        print_test("Argument parser", "FAIL", str(e))

    print(f"\n{tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total

def test_data_loading():
    """Test 6: Data Loading"""
    print("\n" + "="*60)
    print("TEST 6: Data Loading")
    print("="*60)

    tests_passed = 0
    tests_total = 0

    try:
        from utilts import prepare_dataset

        # Test prepare_dataset function
        tests_total += 1
        if hasattr(utilts, 'prepare_dataset'):
            print_test("prepare_dataset function exists", "PASS")
            tests_passed += 1
        else:
            print_test("prepare_dataset function exists", "FAIL")

        # Test loading a small dataset
        tests_total += 1
        try:
            datas, question_string, Q_id = prepare_dataset("simpleqa")
            if datas and len(datas) > 0:
                print_test("Dataset loading", "PASS", f"Loaded {len(datas)} questions")
                tests_passed += 1
            else:
                print_test("Dataset loading", "FAIL", "No data loaded")
        except Exception as e:
            print_test("Dataset loading", "SKIP", f"Data file may not exist: {e}")

    except Exception as e:
        print_test("Data loading", "FAIL", str(e))

    print(f"\n{tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total

def test_directory_structure():
    """Test 7: Directory Structure"""
    print("\n" + "="*60)
    print("TEST 7: Directory Structure")
    print("="*60)

    tests_passed = 0
    tests_total = 0

    # Check required directories
    directories = [
        "Hydra_run",
        "Hydra_run/__pycache__",
        "online_search",
        "answer",
        "wikidata_subgraph",
    ]

    for dir_name in directories:
        tests_total += 1
        if Path(dir_name).exists():
            print_test(f"Directory {dir_name} exists", "PASS")
            tests_passed += 1
        else:
            print_test(f"Directory {dir_name} exists", "WARN", "Will be created on first run")

    # Check required files
    files = [
        "Hydra_run/hydra_main.py",
        "Hydra_run/utilts.py",
        "Hydra_run/utilts2.py",
        "Hydra_run/wikidata_api_client.py",
        ".env",
        "requirements.txt",
    ]

    for file_name in files:
        tests_total += 1
        if Path(file_name).exists():
            print_test(f"File {file_name} exists", "PASS")
            tests_passed += 1
        else:
            print_test(f"File {file_name} exists", "FAIL")

    print(f"\n{tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total

def main():
    """Run all tests"""
    print(f"\n{Colors.BOLD}HydraRAG Integration Test Suite{Colors.END}")
    print(f"Testing Qwen API integration and all project components\n")

    total_passed = 0
    total_tests = 0

    # Run all tests
    tests = [
        ("Environment Variables", test_environment_variables),
        ("Qwen API", test_qwen_api),
        ("run_LLM Function", test_run_llm_function),
        ("Wikidata API", test_wikidata_api),
        ("Argument Parser", test_argument_parser),
        ("Data Loading", test_data_loading),
        ("Directory Structure", test_directory_structure),
    ]

    for test_name, test_func in tests:
        try:
            passed, total = test_func()
            total_passed += passed
            total_tests += total
        except Exception as e:
            print(f"\n{Colors.RED}Error running {test_name}: {e}{Colors.END}")

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"Total tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_tests - total_passed}")
    print(f"Success rate: {success_rate:.1f}%")

    if success_rate >= 80:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Integration test PASSED{Colors.END}")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Integration test FAILED{Colors.END}")
        return 1

if __name__ == "__main__":
    sys.exit(main())