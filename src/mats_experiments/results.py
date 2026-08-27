from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ProjectConfig


@dataclass
class RunDirectory:
    root: Path

    @classmethod
    def create(cls, cfg: ProjectConfig, method: str | None = None) -> "RunDirectory":
        method = method or cfg.training.method
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{method}-seed{cfg.experiment.seed}-{stamp}-{cfg.fingerprint()}"
        root = Path(cfg.experiment.output_root) / cfg.experiment.name / run_id
        for child in ("checkpoints", "artifacts", "plots"):
            (root / child).mkdir(parents=True, exist_ok=False if child == "checkpoints" else True)
        run = cls(root.resolve())
        run.write_json("config.resolved.json", cfg.as_dict())
        run.write_json("manifest.json", environment_manifest())
        return run

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.jsonl"

    def log_metric(self, record: dict[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def write_json(self, relative_path: str, payload: Any) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temp, path)
        return path


def environment_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "argv": sys.argv,
    }
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        manifest["git_commit"] = None
    try:
        import torch

        manifest["torch"] = torch.__version__
        manifest["cuda_available"] = torch.cuda.is_available()
        manifest["cuda_devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    except ImportError:
        manifest["torch"] = None
        manifest["cuda_available"] = False
        manifest["cuda_devices"] = []
    return manifest

