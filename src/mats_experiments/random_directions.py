from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create norm-matched random residual directions")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--orthogonal", action="store_true")
    args = parser.parse_args()
    import torch

    source = torch.load(args.source, map_location="cpu", weights_only=True)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    random_directions = {}
    source_norms = {}
    random_norms = {}
    for layer, direction in source["directions"].items():
        random = torch.randn(direction.shape, generator=generator, dtype=direction.dtype)
        if args.orthogonal:
            unit = direction / torch.linalg.vector_norm(direction)
            random = random - torch.dot(random, unit) * unit
        random = random / torch.linalg.vector_norm(random) * torch.linalg.vector_norm(direction)
        random_directions[layer] = random
        source_norms[layer] = float(torch.linalg.vector_norm(direction.float()).item())
        random_norms[layer] = float(torch.linalg.vector_norm(random.float()).item())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "directions": random_directions,
            "base_means": source.get("base_means", {}),
            "metadata": {
                "kind": "random_control",
                "source": args.source,
                "seed": args.seed,
                "orthogonal": args.orthogonal,
                "norm_matching": "per-layer L2 norm equals the source SFT mean-shift norm",
                "source_norms": source_norms,
                "random_norms": random_norms,
            },
        },
        output,
    )
    print(json.dumps({"output": str(output), "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
