from __future__ import annotations

import argparse
import json

from .config import load_config
from .data import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate config and construct dataset splits")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Validate only the schema; useful before optional dataset dependencies are installed",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.skip_data:
        print(json.dumps({"config_fingerprint": cfg.fingerprint(), "status": "valid"}, indent=2))
        return
    bundle = build_dataset(cfg.data, cfg.experiment.seed)
    print(
        json.dumps(
            {
                "config_fingerprint": cfg.fingerprint(),
                "splits": {
                    "train": len(bundle.train),
                    "task_test": len(bundle.task_test),
                    "probe": len(bundle.probe),
                    "old": len(bundle.old),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
