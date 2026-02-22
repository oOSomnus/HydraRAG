# Qwen API Integration Guide

本项目已集成阿里云 Qwen（通义千问）大模型支持，使用 OpenAI 兼容的 API 格式调用。

## 支持的 Qwen 模型

- `qwen` - 使用 `qwen-plus` 模型（**默认推荐**）
- `qwen25` - 使用 `qwen2.5-72b-instruct` 模型
- `qwen3` - 使用 `qwen-plus` 模型

## 快速开始

### 1. 配置 .env 文件

在项目根目录下创建或编辑 `.env` 文件：

```env
# 阿里云 Qwen API Key
QWEN_API_KEY=sk-your-qwen-api-key-here

# 也可以使用 DASHSCOPE_API_KEY (兼容旧配置)
DASHSCOPE_API_KEY=sk-your-qwen-api-key-here

# SerpAPI Key (用于网页搜索)
SERP_API_KEY=your-serpapi-key-here

# 可选：其他模型的 API Key
OPENAI_API_KEY=your-openai-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key
GOOGLE_API_KEY=your-google-api-key
```

### 2. 安装依赖

确保已安装 `python-dotenv`：

```bash
pip install python-dotenv
```

### 3. 直接运行

项目会自动从 `.env` 文件加载 API Key：

```bash
cd Hydra_run
python hydra_main.py simpleqa --depth 1 --no-freebase
```

默认使用 `qwen-plus` 模型。

### 方式二：使用配置脚本

```bash
python setup_api_keys.py
```

按照提示输入您的阿里云 DashScope API Key。

### 方式三：直接修改代码

编辑 `Hydra_run/utilts.py` 文件，找到 Qwen 配置部分：

```python
elif "qwen" in model:
    print("using Aliyun Qwen API")
    qwen_api_key = os.getenv("DASHSCOPE_API_KEY", "your-api-key-here")  # 修改此处
    openai_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    client = OpenAI(
        api_key=qwen_api_key,
        base_url=openai_api_base,
    )
```

## 使用示例

### 基本用法（使用默认 qwen 模型）

```bash
cd Hydra_run
python hydra_main.py simpleqa --depth 1 --no-freebase
```

### 使用不同 Qwen 模型

```bash
# 使用 qwen-plus（默认）
python hydra_main.py simpleqa --model qwen --depth 1 --no-freebase

# 使用 qwen2.5-72b-instruct
python hydra_main.py simpleqa --model qwen25 --depth 1 --no-freebase

# 使用 qwen3 (qwen-plus)
python hydra_main.py simpleqa --model qwen3 --depth 1 --no-freebase
```

### 完整参数示例

```bash
python hydra_main.py webqsp \
    --model qwen \
    --depth 3 \
    --allr \
    --allsource \
    --no-freebase
```

## API 端点信息

- **Base URL**: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **API 端点**: `/chat/completions`
- **兼容性**: OpenAI 格式兼容

## 常见问题

### 1. API 认证失败

确保：
- API Key 正确设置
- 网络可以访问阿里云服务
- API Key 有足够的配额

### 2. 超时问题

如果遇到超时，可以：
- 检查网络连接
- 增加重试次数（代码中默认重试3次）
- 考虑使用更小的模型

### 3. 模型选择

- `qwen` / `qwen3`: 使用 `qwen-plus`，适合通用场景
- `qwen25`: 使用 `qwen2.5-72b-instruct`，更大参数量，可能获得更好效果

## API 参数说明

当前配置使用的默认参数：

```python
response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0.4,     # 采样温度，控制输出多样性
    max_tokens=256,      # 最大输出 token 数
    frequency_penalty=0,
    presence_penalty=0
)
```

如需修改这些参数，请编辑 `Hydra_run/utilts.py` 中的 `run_LLM` 函数。

## 更新记录

- 2026-02-22: 初始集成阿里云 Qwen API 支持