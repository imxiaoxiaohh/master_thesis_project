"""Configuration for math generation."""

import os
from dataclasses import dataclass

@dataclass
class LLMConfig:
    provider: str
    model: str  
    api_key: str
    base_url: str = None

# Dataset and output paths
# DATASET_PATH = "D:/0_Master_Thesis/math_agent/dataset/test_dataset.json"

# This calculates the path from this file's location (Generation) -> up to the project root -> then into 'dataset'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
DATASET_PATH = os.path.join(PROJECT_ROOT, "Dataset", "academic_dataset_Final.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# LLM configurations
LLMS = {
    # OpenAI
    "gpt-o1-mini": LLMConfig("openai", "gpt-o1-mini", "OPENAI_API_KEY"),
    "gpt-4o-mini": LLMConfig("openai", "gpt-4o-mini", "OPENAI_API_KEY"),
    "gpt-4.1-mini": LLMConfig("openai", "gpt-4.1-mini", "OPENAI_API_KEY"),
    "gpt-4.1": LLMConfig("openai", "gpt-4.1", "OPENAI_API_KEY"),
    # Anthropic  
    "claude": LLMConfig("anthropic", "claude-3-5-sonnet-20241022", "ANTHROPIC_API_KEY"),
    
    # Deepseek
    "deepseek-R1": LLMConfig("deepseek", "deepseek-reasoner", "DeepSeek_API_Key", "https://api.deepseek.com"),

    # Together AI
    "llama3.3-70B": LLMConfig("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "TOGETHER_API_KEY", "https://api.together.xyz/v1"),
    "qwen3-235B": LLMConfig("together", "Qwen/Qwen3-235B-A22B-fp8-tput", "TOGETHER_API_KEY", "https://api.together.xyz/v1"),
    "qwen-qwq-32B": LLMConfig("together", "Qwen/QwQ-32B", "TOGETHER_API_KEY", "https://api.together.xyz/v1"),
    
    
    # Hugging Face
    "qwen-math": LLMConfig("huggingface", "Qwen/Qwen2.5-Math-1.5B", "HF_API_KEY"),
    "deepseek-math": LLMConfig("huggingface", "deepseek-ai/deepseek-math-7b-instruct", "HF_API_KEY"),
}

def get_llm_config(name):
    """Get LLM configuration by name."""
    if name not in LLMS:
        available = ', '.join(LLMS.keys())
        raise ValueError(f"Unknown LLM '{name}'. Available: {available}")
    
    config = LLMS[name]
    api_key = os.getenv(config.api_key)
    if not api_key:
        raise ValueError(f"API key not found: {config.api_key}")
    
    return config, api_key

def get_output_path(llm_name):
    """Get output file path for LLM."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, f"{llm_name}_results.json")