from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct the SFT-minus-RL contrast trace")
    parser.add_argument("--sft", required=True)
    parser.add_argument("--rl", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch

    sft = torch.load(args.sft, map_location="cpu", weights_only=True)
    rl = torch.load(args.rl, map_location="cpu", weights_only=True)
    layers = sorted(set(sft["directions"]) & set(rl["directions"]))
    if not layers:
        raise ValueError("Trace artifacts share no layer identifiers")
    contrast = {layer: sft["directions"][layer] - rl["directions"][layer] for layer in layers}
    base_means = {
        layer: 0.5 * (sft["base_means"][layer] + rl["base_means"][layer]) for layer in layers
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "directions": contrast,
            "base_means": base_means,
            "metadata": {"kind": "sft_minus_rl", "sft": args.sft, "rl": args.rl},
        },
        output,
    )
    print(json.dumps({"output": str(output), "layers": layers}, indent=2))


if __name__ == "__main__":
    main()

