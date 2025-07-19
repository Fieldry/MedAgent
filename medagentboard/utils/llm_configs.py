from dotenv import load_dotenv
import os

load_dotenv()

LLM_MODELS_SETTINGS = {
    "deepseek-v3-official": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat",
        "comment": "DeepSeek V3 Official DeepSeek-V3-0324",
        "reasoning": False,
    },
    "deepseek-r1-official": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-reasoner",
        "comment": "DeepSeek R1 Reasoning Model Official DeepSeek-R1-0528",
        "reasoning": True,
    },
    "qwen3": {
        "api_key": os.getenv("ALI_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen3-235b-a22b",
        "comment": "Qwen3 235B",
        "reasoning": False,
    },
    "claude4": {
        "api_key": os.getenv("LAOZHANG_API_KEY"),
        "base_url": "https://api.laozhang.ai/v1",
        "model_name": "claude-sonnet-4-20250514",
        "comment": "Claude Sonnet 4",
        "reasoning": False,
    },
    "o4-mini": {
        "api_key": os.getenv("LAOZHANG_API_KEY"),
        "base_url": "https://api.laozhang.ai/v1",
        "model_name": "o4-mini-2025-04-16",
        "comment": "O4 Mini",
        "reasoning": False,
    }
}