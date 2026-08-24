"""Download PLM models locally via ModelScope."""
from modelscope import snapshot_download
import argparse

MODELS = {
    "roberta": "FacebookAI/roberta-large",
    "qwen": "Qwen/Qwen2.5-3B",
    "llama3": "LLM-Research/Meta-Llama-3-8B",
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="qwen", choices=list(MODELS.keys()))
args = parser.parse_args()

model_id = snapshot_download(MODELS[args.model], cache_dir="./models")
print(f"Downloaded {args.model} -> {model_id}")
