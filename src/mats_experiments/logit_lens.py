from __future__ import annotations

import argparse
import json

from .config import load_config
from .hf_backend import load_adapter_model, load_tokenizer, require_training_stack


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode trace directions through final norm and unembedding")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--direction-artifact", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    require_training_stack()
    import torch

    cfg = load_config(args.config)
    model = load_adapter_model(cfg, args.checkpoint)
    tokenizer = load_tokenizer(cfg)
    artifact = torch.load(args.direction_artifact, map_location="cpu", weights_only=True)
    direction = artifact["directions"][args.layer].to(next(model.parameters()).device)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    norm = getattr(getattr(base, "model", base), "norm", None)
    vector = direction.unsqueeze(0)
    if norm is not None:
        vector = norm(vector.to(dtype=next(norm.parameters()).dtype))
    logits = base.get_output_embeddings()(vector).float().squeeze(0)
    values, indices = torch.topk(logits, args.top_k)
    rows = [
        {"token": tokenizer.decode([int(index)]), "token_id": int(index), "logit": float(value)}
        for value, index in zip(values, indices)
    ]
    print(json.dumps({"layer": args.layer, "top_tokens": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
