# HydraRAG Configuration & Setup Guide

## Prerequisites

### 1. Environment Setup
```bash
cd /home/oosomnus/workspace/HydraRAG
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Required API Keys

You need to set up the following API keys in `Hydra_run/utilts.py`:

#### 2.1 LLM API Keys
Edit `Hydra_run/utilts.py` and replace placeholder keys with your actual keys:

- **OpenAI API Key**: Replace `"your_api_key"` with your actual OpenAI API key
- **Google Gemini API Key**: Replace `"your_api_key"` with your actual Gemini API key (optional)
- **DeepSeek API Key**: Replace `"your_api_key"` with your actual DeepSeek API key (optional)

#### 2.2 Search API Key
Edit `Hydra_run/utilts2.py` and replace with your SerpAPI key:
- **SerpAPI Key**: Replace `serpapi_Key` with your actual SerpAPI key

## Required Configuration Files

### 1. Create `.env` file in Hydra_run/ (optional but recommended)
```bash
# HydraRAG Environment Variables
OPENAI_API_KEY=your_openai_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here
GOOGLE_API_KEY=your_google_api_key_here  # if using Gemini
```

### 2. Update server_urls.txt for Wikidata
Edit `Hydra_run/server_urls.txt` to make it an empty file or comment out placeholder entries:
```
# Wikidata service endpoints - not used in this Wikidata-only version
# This file is kept for compatibility with the original code structure
```

## Running HydraRAG

### 1. Basic Usage
```bash
cd Hydra_run

# Run with minimal settings (only Wikidata, no Freebase)
python hydra_main.py simpleqa --model gpt3 --depth 1 --no-freebase

# Example with WebQSP dataset
python hydra_main.py webqsp --model gpt3 --depth 2 --no-freebase --no-wikidocu

# Example with all sources except Freebase
python hydra_main.py simpleqa --model gpt3 --depth 2 --no-freebase
```

### 2. Available Datasets
- `simpleqa` - SimpleQA dataset
- `webqsp` - WebQuestions SP dataset
- `webquestions` - WebQuestions dataset
- `cwq` - ComplexWebQuestions dataset
- `qald` - QALD-10 dataset
- `hotpot` - HotpotQA dataset
- `zeroshotre` - Zero-shot Relation Extraction dataset

### 3. Available Models
- `gpt3` - GPT-3.5 Turbo
- `gpt4` - GPT-4 Turbo
- `llama` - Llama-3.1-8B
- `deepseek` - DeepSeek-v3
- `llama70b` - Llama-3.1-70B

### 4. Available Options
- `--depth N` - Set maximum search depth (1-4)
- `--model MODEL` - Choose LLM backend
- `--allr` - Enable all relations (Hydra mode)
- `--allsource` - Enable all sources except Freebase
- `--no-freebase` - Disable Freebase (now default since we removed Freebase support)
- `--no-wikikg` - Disable Wikidata KG
- `--no-web` - Disable web search
- `--no-wikidocu` - Disable Wikipedia documents

## Sample Run Commands

### Quick Test (recommended for first run):
```bash
cd Hydra_run
python hydra_main.py simpleqa --model gpt3 --depth 1 --no-freebase --no-web --no-wikidocu
```

### With Web Search:
```bash
cd Hydra_run
python hydra_main.py webqsp --model gpt3 --depth 2 --no-freebase
```

### Full Featured Run:
```bash
cd Hydra_run
python hydra_main.py simpleqa --model gpt3 --depth 2 --allsource --no-freebase
```

## Troubleshooting

### 1. API Key Issues
If you get authentication errors, ensure your API keys are correctly set in:
- `Hydra_run/utilts.py` (for LLM APIs)
- `Hydra_run/utilts2.py` (for SerpAPI)

### 2. Memory Issues
HydraRAG loads sentence transformer models that can consume significant memory. Consider:
- Using smaller models
- Reducing batch sizes in the code
- Ensuring adequate RAM (recommended 16GB+)

### 3. Model Loading Slow
The first run may take time to download models. Subsequent runs will be faster.

### 4. Wikidata Queries Rate Limiting
Public Wikidata SPARQL endpoints may have rate limits. If experiencing timeouts:
- Reduce concurrency
- Add delays between queries
- Consider implementing caching