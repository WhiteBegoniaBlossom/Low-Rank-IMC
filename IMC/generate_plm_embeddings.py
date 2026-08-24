"""
Generate PLM node embeddings for all fb237 dataset versions using tyler-main's
complete pipeline: compute_logits_emb() -> PromptEncoder -> PoolEncoder -> sigmoid.

Architecture ported from tyler-main/ITLP/tyler_simple.py (PoolEncoder, PromptEncoder)
and tyler-main/ITLP/utils.py (compute_logits_emb).

Usage:
    python generate_plm_embeddings.py --version fb237_v1 --model roberta --aggregation sum
    python generate_plm_embeddings.py --version fb237_v1 --model llama3 --aggregation attn
    python generate_plm_embeddings.py --all --model roberta --aggregation mean
"""
import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    BartForConditionalGeneration,
)

# ============================================================
# Prompt templates (from tyler-main/ITLP/utils.py)
# ============================================================

_ENTITY_TEMPLATES = [
    "[1] is a type of [2].",
    "[1] is located in [2].",
    "[1] is member of [2].",
    "[1] is equivalent to [2].",
    "[1] is different from [2].",
    "[1] is similar to [2].",
]

_ENTITY_REPR = [_ENTITY_TEMPLATES]

# Model presets: short name -> (huggingface_id, is_causal, default_batch_size)
MODEL_PRESETS = {
    "roberta": ("FacebookAI/roberta-large", False, 64),
    "llama3": ("meta-llama/Meta-Llama-3-8B", True, 16),
    "qwen": ("Qwen/Qwen2.5-3B", True, 4),
}


def resolve_model_path(huggingface_model, models_dir=None):
    """If models_dir is set and a local copy exists, use it.
    Returns (model_path, local_files_only).
    """
    if models_dir:
        local_path = os.path.join(models_dir, huggingface_model)
        if os.path.isdir(local_path):
            print(f"Using local model: {local_path}")
            return local_path, True
    return huggingface_model, False


def format_templates(templates, tokenizer, causal=False):
    if causal:
        type_temp = [t.split("[2]")[0].strip() + ":" for t in templates]
    else:
        type_temp = [t.replace("[2]", tokenizer.mask_token).strip() for t in templates]
    return type_temp


def compute_logits_emb(
    id2entity,
    entity2label=None,
    huggingface_model="FacebookAI/roberta-large",
    is_causal=False,
    batch_size=64,
    persist="id2logits",
    models_dir=None,
):
    """Extract raw PLM hidden states for each entity+prompt combination.
    Returns tensor of shape [num_entities, num_templates(6), hidden_dim]."""
    if os.path.isfile(f"{persist}.pt"):
        id2logits = torch.load(f"{persist}.pt")
        print(f"Found cached embeddings in {persist}.pt")
        return id2logits

    model_path, local_only = resolve_model_path(huggingface_model, models_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
    if entity2label is not None:
        id2label = {idx: entity2label.get(id2entity[idx], id2entity[idx]) for idx in id2entity}
    else:
        id2label = {idx: " ".join(id2entity[idx].split("_")) for idx in id2entity}

    if not is_causal:
        if "bart" in huggingface_model:
            plm = BartForConditionalGeneration.from_pretrained(
                model_path, local_files_only=local_only
            ).to("cuda:0")
        else:
            plm = AutoModelForMaskedLM.from_pretrained(
                model_path, local_files_only=local_only
            ).to("cuda:0")
    else:
        plm = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="cuda",
            local_files_only=local_only,
        )
        if "llama" in huggingface_model.lower() or "pythia" in huggingface_model.lower() or "qwen" in huggingface_model.lower():
            tokenizer.pad_token = tokenizer.eos_token

    plm.eval()

    repr = []
    for TEMPLATES in _ENTITY_REPR:
        id2logits = []
        templates = format_templates(TEMPLATES, causal=is_causal, tokenizer=tokenizer)
        for prompt_template in templates:
            prompts = {
                idx: prompt_template.replace("[1]", ent)
                for idx, ent in id2label.items()
            }

            nlogits = []
            for i in tqdm.tqdm(range(0, len(prompts), batch_size)):
                if len(prompts) - i < batch_size:
                    curr_prompts = [prompts[j] for j in range(i, i + len(prompts) - i)]
                else:
                    curr_prompts = [prompts[j] for j in range(i, i + batch_size)]

                labels_tok = tokenizer(
                    curr_prompts,
                    add_special_tokens=True,
                    padding=True,
                    return_tensors="pt",
                )
                labels_tok = {key: value.to(plm.device) for key, value in labels_tok.items()}

                with torch.no_grad():
                    if "bart" in huggingface_model:
                        logits = plm(
                            **labels_tok, output_hidden_states=True
                        ).encoder_hidden_states[-1]
                    else:
                        logits = plm(
                            **labels_tok, output_hidden_states=True
                        ).hidden_states[-1]

                if not is_causal:
                    mask_token_index = torch.nonzero(
                        labels_tok["input_ids"] == tokenizer.mask_token_id,
                        as_tuple=True,
                    )
                    mask_logits = logits[mask_token_index]
                else:
                    mask_logits = logits[
                        torch.arange(logits.shape[0]),
                        (labels_tok["attention_mask"] != 0).sum(-1) - 1,
                    ]

                nlogits.extend(mask_logits.to("cpu"))
                torch.cuda.empty_cache()

            id2logits.append(torch.stack(nlogits))

        repr.append(torch.stack(id2logits, dim=1).to(torch.float32))

    if len(repr) > 1:
        repr = torch.stack(repr, dim=1)
    else:
        repr = repr[0]

    if persist:
        print(f"Saving raw PLM embeddings to {persist}.pt")
        os.makedirs(os.path.dirname(persist) if os.path.dirname(persist) else ".", exist_ok=True)
        torch.save(repr, f"{persist}.pt")
    return repr


# ============================================================
# PromptEncoder + PoolEncoder (from tyler-main/ITLP/tyler_simple.py)
# ============================================================

class PromptEncoder(nn.Module):
    """Per-template encoder: Dropout -> LayerNorm -> Linear projection.
    Reduces hidden_dim (e.g. 1024) -> inner_dim (128)."""
    def __init__(self, input_dim, output_dim=128):
        super(PromptEncoder, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear_1 = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.linear_1(x)
        return x


class PoolEncoder(nn.Module):
    """Aggregates multiple prompt embeddings into a single vector per entity.

    Pipeline:
      1. Each template: PromptEncoder (1024 -> 128)
      2. Aggregate 6 x 128 vectors via sum/mean/concat/attn
      3. ReLU -> Dropout -> Linear projection -> output_dim
    """
    def __init__(self, num_features, num_templates, output_dim, inner_dim=128, pooling="sum"):
        super(PoolEncoder, self).__init__()
        self.num_features = num_features
        self.num_templates = num_templates
        self.inner_dim = inner_dim
        self.output_dim = output_dim
        self.prompt_encoders = nn.ModuleList(
            [PromptEncoder(self.num_features) for _ in range(self.num_templates)]
        )
        self.pooling = pooling
        self.dropout = nn.Dropout(0.1)

        if pooling == "concat":
            self.output_layer = nn.Linear(self.inner_dim * num_templates, self.output_dim)
        else:
            self.output_layer = nn.Linear(self.inner_dim, self.output_dim)

    def aggregate_prompts(self, prompt_encodings, aggr="sum"):
        """Aggregate [B, num_templates, inner_dim] -> [B, inner_dim] (or [B, inner_dim*num_templates] for concat)."""
        if aggr == "mean":
            combined = torch.mean(prompt_encodings, dim=1)
        elif aggr == "sum":
            combined = torch.sum(prompt_encodings, dim=1)
        elif aggr == "concat":
            combined = prompt_encodings.reshape(prompt_encodings.size(0), -1)
        elif aggr == "attn":
            query = prompt_encodings[:, 0:1, :]
            keys_values = prompt_encodings[:, 1:, :]
            scores = torch.matmul(query, keys_values.transpose(1, 2))
            attention_weights = F.softmax(scores, dim=-1)
            attended = torch.matmul(attention_weights, keys_values).squeeze(1)
            combined = query.squeeze(1) + attended
        return combined

    def forward(self, x):
        """x: [B, num_templates, num_features] -> [B, output_dim]"""
        encoded_prompts = []
        for i, encoder in enumerate(self.prompt_encoders):
            prompt_input = x[:, i, :]
            encoded = encoder(prompt_input)
            encoded_prompts.append(encoded)

        encoded_prompts = torch.stack(encoded_prompts, dim=1)  # [B, num_templates, inner_dim]
        combined = self.aggregate_prompts(encoded_prompts, aggr=self.pooling)
        combined = torch.relu(combined)
        combined = self.dropout(combined)
        output = self.output_layer(combined)
        return output


def aggregate_logits(id2logits, aggregation="mean", output_dim=128, inner_dim=128,
                      device="cuda:0", seed=28):
    """
    Apply PoolEncoder to raw PLM logits, exactly matching tyler-main's
    TextGraphClassifier.aggregate_embeddings() pipeline:

      id2logits [N, 6, hidden_dim]
        -> PoolEncoder (PromptEncoder per template -> aggregate -> project)
        -> sigmoid
        -> [N, output_dim]
    """
    num_entities, num_templates, hidden_dim = id2logits.shape
    print(f"PoolEncoder: {hidden_dim} -> {inner_dim} per template, "
          f"aggregation={aggregation}, -> {output_dim}")

    torch.manual_seed(seed)
    pool_encoder = PoolEncoder(
        num_features=hidden_dim,
        num_templates=num_templates,
        output_dim=output_dim,
        inner_dim=inner_dim,
        pooling=aggregation,
    ).to(device)

    # Process in batches to avoid OOM
    batch_size = 256
    results = []
    pool_encoder.eval()
    with torch.no_grad():
        for i in range(0, num_entities, batch_size):
            batch = id2logits[i:i + batch_size].to(device)  # [B, 6, hidden_dim]
            # PoolEncoder expects [B, 1, num_templates, hidden_dim]
            batch = batch.unsqueeze(1)  # [B, 1, 6, hidden_dim]
            out = pool_encoder(batch[:, 0, :, :])  # [B, 6, hidden_dim] -> [B, output_dim]
            out = torch.sigmoid(out)
            results.append(out.cpu())
            del batch, out
            torch.cuda.empty_cache()

    aggregated = torch.cat(results, dim=0)  # [N, output_dim]
    return aggregated


# ============================================================
# Data loading helpers
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EMBED_DIR = os.path.join(os.path.dirname(__file__), "embeddings")

_V1_SEEDS = [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010]
_V2_SEEDS = [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010]
_V3_SEEDS = [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010]
VERSIONS = [
    "fb237_v1", "fb237_v1_ind",
    "fb237_v2", "fb237_v2_ind",
    "fb237_v3", "fb237_v3_ind",
] + [
    f"fb237_v1_ind_seed{s}" for s in _V1_SEEDS
] + [
    f"fb237_v2_ind_seed{s}" for s in _V2_SEEDS
] + [
    f"fb237_v3_ind_seed{s}" for s in _V3_SEEDS
]


def collect_entities(version_dir):
    entities = set()
    for split in ["train.txt", "valid.txt", "test.txt"]:
        path = os.path.join(version_dir, split)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    entities.add(parts[0])
                    entities.add(parts[2])
    return entities


def load_labels(version_dir):
    label_path = os.path.join(version_dir, "label.tsv")
    entity2label = {}
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    entity2label[parts[0]] = parts[1]
    return entity2label


def generate_embeddings(version, model_name="FacebookAI/roberta-large",
                        is_causal=False, batch_size=64, model_short="roberta",
                        aggregation="sum", output_dim=128, inner_dim=128, seed=28,
                        models_dir=None):
    version_dir = os.path.join(DATA_DIR, version)
    if not os.path.isdir(version_dir):
        print(f"ERROR: {version_dir} not found, skipping")
        return

    agg_key = f"{model_short}_{aggregation}"
    output_path = os.path.join(EMBED_DIR, f"{version}_{agg_key}_embeddings.pkl")
    if os.path.exists(output_path):
        print(f"[{version}] Embeddings already exist at {output_path}, skip")
        return

    print(f"[{version}] Collecting entities...")
    entities = collect_entities(version_dir)
    entity_list = sorted(entities)
    print(f"[{version}] Found {len(entity_list)} unique entities")

    entity2label = load_labels(version_dir)
    print(f"[{version}] Loaded {len(entity2label)} entity labels from label.tsv")

    id2entity = {i: e for i, e in enumerate(entity_list)}

    # Step 1: Generate raw PLM embeddings
    cache_path = os.path.join(EMBED_DIR, f"{version}_{model_short}_id2logits")
    print(f"[{version}] Step 1/2: Generating raw PLM embeddings via {model_name}...")
    id2logits = compute_logits_emb(
        id2entity=id2entity,
        entity2label=entity2label,
        huggingface_model=model_name,
        is_causal=is_causal,
        batch_size=batch_size,
        persist=cache_path,
        models_dir=models_dir,
    )
    print(f"[{version}] Raw id2logits shape: {id2logits.shape}")

    # Step 2: PoolEncoder aggregation (matching tyler-main's aggregate_embeddings)
    print(f"[{version}] Step 2/2: PoolEncoder aggregation ({aggregation})...")
    aggregated = aggregate_logits(
        id2logits, aggregation=aggregation,
        output_dim=output_dim, inner_dim=inner_dim, seed=seed
    )
    print(f"[{version}] Aggregated shape: {aggregated.shape}")

    embeddings_np = aggregated.cpu().numpy().astype(np.float32)
    node2emb = {entity_list[i]: embeddings_np[i] for i in range(len(entity_list))}
    with open(output_path, "wb") as f:
        pickle.dump(node2emb, f)

    print(f"[{version}] Saved {len(node2emb)} embeddings to {output_path}")
    print(f"[{version}] Embedding dim: {embeddings_np.shape[1]}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PLM node embeddings using tyler-main's full pipeline"
    )
    parser.add_argument("--version", type=str, default=None, help="Specific version (e.g. fb237_v1)")
    parser.add_argument("--all", action="store_true", help="Generate for all 8 versions")
    parser.add_argument("--model", type=str, default="roberta",
                        choices=list(MODEL_PRESETS.keys()),
                        help=f"PLM model: {list(MODEL_PRESETS.keys())}")
    parser.add_argument("--aggregation", type=str, default="sum",
                        choices=["sum", "mean", "concat", "attn"],
                        help="Aggregation method for 6 prompt embeddings")
    parser.add_argument("--output_dim", type=int, default=128,
                        help="Final embedding dimension after PoolEncoder (matches tyler-main sem_dim)")
    parser.add_argument("--inner_dim", type=int, default=128,
                        help="Inner dimension of PromptEncoder projection")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Override HF model ID or local path (e.g. /path/to/Llama-3-8B)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override default batch size for PLM encoding")
    parser.add_argument("--seed", type=int, default=28,
                        help="Random seed for PoolEncoder weight init (matches tyler-main)")
    parser.add_argument("--models_dir", type=str, default="./models",
                        help="Local directory for pre-downloaded models (e.g. ./models). "
                             "If {models_dir}/{huggingface_id} exists, use it; otherwise download from HF.")
    args = parser.parse_args()

    model_name, is_causal, default_bs = MODEL_PRESETS[args.model]
    if args.model_name:
        model_name = args.model_name
    batch_size = args.batch_size if args.batch_size is not None else default_bs

    if args.all:
        versions = VERSIONS
    elif args.version:
        versions = [args.version]
    else:
        print("Specify --version <name> or --all")
        return

    for v in versions:
        generate_embeddings(
            v, model_name=model_name, is_causal=is_causal,
            batch_size=batch_size, model_short=args.model,
            aggregation=args.aggregation,
            output_dim=args.output_dim, inner_dim=args.inner_dim,
            seed=args.seed,
            models_dir=args.models_dir,
        )

    print("Done.")


if __name__ == "__main__":
    main()
