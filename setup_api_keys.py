#!/usr/bin/env python3
"""
HydraRAG API Key Setup Script
This script helps you set up your API keys in the required files.
"""

import os
import sys
import re
from pathlib import Path

def update_file_with_api_keys(utilts_path, utilts2_path):
    """Update the API key placeholders with actual keys provided by user"""

    print("HydraRAG API Key Setup")
    print("="*50)

    # Read utilts.py
    with open(utilts_path, 'r') as f:
        utilts_content = f.read()

    # Read utilts2.py
    with open(utilts2_path, 'r') as f:
        utilts2_content = f.read()

    print("\nSetting up API keys in utilts.py...")

    # Find and replace OpenAI API key placeholder (in the openai_api_key assignment)
    openai_patterns = [
        r'(openai_api_key = )"your_api_key"',
        r'(openai_api_key = )"your_openai_api_key"',
        r'(api_key=)"your_api_key"(?=,)',
    ]

    for pattern in openai_patterns:
        matches = re.findall(pattern, utilts_content)
        if matches:
            print(f"\nOpenAI API Key is required for this system.")
            openai_key = input("Please enter your OpenAI API key (or press Enter to skip): ").strip()
            if openai_key:
                # Replace all occurrences of "your_api_key" assignments
                utilts_content = re.sub(r'openai_api_key = "your_api_key"', f'openai_api_key = "{openai_key}"', utilts_content)
                utilts_content = re.sub(r'"your_api_key"', f'"{openai_key}"', utilts_content)
                print("✓ OpenAI API key updated")
            else:
                print("! Warning: OpenAI API key not set. Some models will not work.")
            break

    # Find and replace Google Gemini API key placeholder
    if '"your_api_key"' in utilts_content and 'genai.configure' in utilts_content:
        print("\nGoogle Gemini API Key is required for Gemini models.")
        gemini_key = input("Please enter your Google Gemini API key (or press Enter to skip): ").strip()
        if gemini_key:
            utilts_content = re.sub(r'genai\.configure\(api_key="your_api_key"\)', f'genai.configure(api_key="{gemini_key}")', utilts_content)
            print("✓ Google Gemini API key updated")
        else:
            print("! Warning: Google Gemini API key not set. Gemini models will not work.")

    # Find and replace DeepSeek API key placeholder
    if 'openai_api_base = "https://api.deepseek.com"' in utilts_content and 'your_api_key' in utilts_content:
        # Look for the specific pattern around DeepSeek
        deepseek_section = utilts_content[utilts_content.find('https://api.deepseek.com')-100:utilts_content.find('https://api.deepseek.com')+200]
        if 'your_api_key' in deepseek_section:
            print("\nDeepSeek API Key is required for DeepSeek models.")
            deepseek_key = input("Please enter your DeepSeek API key (or press Enter to skip): ").strip()
            if deepseek_key:
                # Find and replace the specific assignment in the DeepSeek section
                pos = utilts_content.find('openai_api_base = "https://api.deepseek.com"')
                # Find the corresponding key assignment in the vicinity
                section_start = max(0, pos - 100)
                section_end = min(len(utilts_content), pos + 200)
                section = utilts_content[section_start:section_end]

                # Replace in the specific section
                updated_section = re.sub(r'(openai_api_key = )"your_api_key"', fr'\1"{deepseek_key}"', section)
                utilts_content = utilts_content[:section_start] + updated_section + utilts_content[section_end:]

                # Also replace any remaining references
                utilts_content = re.sub(r'"your_api_key"(?=.*deepseek)', f'"{deepseek_key}"', utilts_content)
                print("✓ DeepSeek API key updated")
            else:
                print("! Warning: DeepSeek API key not set. DeepSeek model will not work.")

    # Find and replace Aliyun Qwen API key placeholder
    if 'DASHSCOPE_API_KEY' in utilts_content or 'using Aliyun Qwen API' in utilts_content:
        print("\nAliyun Qwen (DashScope) API Key is required for Qwen models.")
        qwen_key = input("Please enter your Aliyun DashScope API key (or press Enter to skip): ").strip()
        if qwen_key:
            # Update utilts.py to use environment variable for Qwen
            if 'os.getenv("DASHSCOPE_API_KEY", "your_api_key")' in utilts_content:
                utilts_content = utilts_content.replace(
                    'os.getenv("DASHSCOPE_API_KEY", "your_api_key")',
                    f'os.getenv("DASHSCOPE_API_KEY", "{qwen_key}")'
                )
                print("✓ Aliyun Qwen API key updated in utilts.py")
            else:
                # Create environment variable setup comment
                env_comment = "\n# Note: For Aliyun Qwen, set the DASHSCOPE_API_KEY environment variable:\n# export DASHSCOPE_API_KEY='your-api-key'\n"
                if env_comment not in utilts_content:
                    utilts_content += env_comment
                print("✓ Aliyun Qwen setup instructions added")
        else:
            print("! Warning: Aliyun Qwen API key not set. Qwen models will not work.")
            print("  You can also set DASHSCOPE_API_KEY as an environment variable instead.")

    print("\nSetting up API keys in utilts2.py...")

    # Find and replace SerpAPI key
    if 'serpapi_Key = "your_own_keys"' in utilts2_content:
        print("\nSerpAPI Key is required for web search functionality.")
        serpapi_key = input("Please enter your SerpAPI key (or press Enter to skip web search): ").strip()
        if serpapi_key:
            utilts2_content = utilts2_content.replace('serpapi_Key = "your_own_keys"', f'serpapi_Key = "{serpapi_key}"')
            print("✓ SerpAPI key updated")
        else:
            print("! Warning: SerpAPI key not set. Web search functionality will not work.")

    # Write back to files
    with open(utilts_path, 'w') as f:
        f.write(utilts_content)

    with open(utilts2_path, 'w') as f:
        f.write(utilts2_content)

    print("\n" + "="*50)
    print("API Key setup completed!")
    print("\nTo run HydraRAG, use:")
    print("  cd Hydra_run")
    print("  python hydra_main.py simpleqa --model gpt3 --depth 1 --no-freebase")


def main():
    # Determine the script location and navigate to Hydra_run
    script_dir = Path(__file__).parent
    hydra_run_dir = script_dir / "Hydra_run"

    if not hydra_run_dir.exists():
        print(f"Error: Hydra_run directory not found at {hydra_run_dir}")
        return 1

    utilts_path = hydra_run_dir / "utilts.py"
    utilts2_path = hydra_run_dir / "utilts2.py"

    if not utilts_path.exists():
        print(f"Error: utilts.py not found at {utilts_path}")
        return 1

    if not utilts2_path.exists():
        print(f"Error: utilts2.py not found at {utilts2_path}")
        return 1

    update_file_with_api_keys(utilts_path, utilts2_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())