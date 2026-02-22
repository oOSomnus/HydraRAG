# HydraRAG Complete Setup and Running Guide

## Overview
This guide will walk you through setting up and running HydraRAG with Wikidata-only functionality (without Freebase dependency).

## Prerequisites

### 1. System Requirements
- Python 3.9 or higher
- At least 8GB RAM (16GB+ recommended)
- Sufficient disk space for model downloads
- Internet connection for API access

### 2. Install Dependencies
```bash
cd /home/oosomnus/workspace/HydraRAG
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Set Up API Keys
```bash
python setup_api_keys.py
```
Follow the prompts to enter your API keys for:
- OpenAI API key (required for most models)
- Google Gemini API key (optional)
- DeepSeek API key (optional)
- SerpAPI key (required for web search)

## Configuration

### 1. Verify Wikidata API Client
The system has been configured to use Wikidata through public API endpoints, so no additional setup is needed for Wikidata access.

### 2. Optional: Environment Variables
Create `.env` file in `Hydra_run/` directory:
```bash
# HydraRAG Environment Variables
OPENAI_API_KEY=your_openai_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here
GOOGLE_API_KEY=your_google_api_key_here  # if using Gemini
```

## Running HydraRAG

### 1. Quick Start (Test Run)
```bash
cd Hydra_run
python hydra_main.py simpleqa --model gpt3 --depth 1 --no-freebase --no-web --no-wikidocu
```

### 2. Standard Run
```bash
cd Hydra_run
python hydra_main.py simpleqa --model gpt3 --depth 2 --no-freebase
```

### 3. Advanced Usage
```bash
# With all sources except Freebase
python hydra_main.py webqsp --model gpt3 --depth 2 --allsource --no-freebase

# Using different models
python hydra_main.py simpleqa --model llama --depth 1 --no-freebase

# Using Wikidata knowledge graph only
python hydra_main.py webquestions --model gpt3 --depth 1 --no-freebase --no-web --no-wikidocu --no-summary
```

## Available Datasets
- `simpleqa` - SimpleQA dataset (smallest, good for testing)
- `webqsp` - WebQuestions SP dataset
- `webquestions` - WebQuestions dataset
- `cwq` - ComplexWebQuestions dataset
- `qald` - QALD-10 dataset
- `hotpot` - HotpotQA dataset
- `zeroshotre` - Zero-shot Relation Extraction dataset

## Available Models
- `gpt3` - GPT-3.5 Turbo
- `gpt4` - GPT-4 Turbo
- `llama` - Llama-3.1-8B
- `deepseek` - DeepSeek-v3
- `llama70b` - Llama-3.1-70B

## Command Line Options
- `--depth N` - Set maximum search depth (1-4)
- `--model MODEL` - Choose LLM backend
- `--allr` - Enable all relations (Hydra mode)
- `--allsource` - Enable all sources except Freebase
- `--no-freebase` - Disable Freebase (already removed from this version)
- `--no-wikikg` - Disable Wikidata KG
- `--no-web` - Disable web search
- `--no-wikidocu` - Disable Wikipedia documents
- `--no-summary` - Disable summary generation

## Expected Output
After running successfully, you should see:
- Progress indicators and logging messages
- Model responses to questions
- Results saved to `answer/` directory (created automatically)

## Troubleshooting

### 1. API Authentication Errors
- Verify your API keys are correctly entered
- Check that your API provider accounts have sufficient credits
- Ensure proper network connectivity

### 2. Memory Issues
- Use simpler models or reduce batch sizes
- Close other memory-intensive applications
- Use smaller datasets initially

### 3. Wikidata API Timeout
- Retry the request (the public endpoint might be busy)
- Consider implementing retry logic if needed

### 4. Import/Dependency Errors
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Some packages might require compilation on certain systems

## Important Notes
- The first run will take longer due to model downloads
- Large datasets and high search depths will take significantly more time
- Costs will apply based on your API usage
- Some models may have rate limits depending on your account tier

## Next Steps
Once the basic setup is working:
1. Experiment with different datasets
2. Try various model options
3. Adjust search depth based on your computational resources
4. Evaluate results for your specific use case