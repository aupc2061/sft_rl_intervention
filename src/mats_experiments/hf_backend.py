from __future__ import annotations

import contextlib
import dataclasses
import os
import random
from pathlib import Path
from typing import Any, Iterator

from .config import ProjectConfig


def trainer_reporting() -> str:
    """Enable W&B only when the caller explicitly names a project."""
    return "wandb" if os.environ.get("WANDB_PROJECT") else "none"


def trainer_run_name(cfg: ProjectConfig, run_root: str | Path) -> str:
    return f"{cfg.experiment.name}-{Path(run_root).name}"


def require_training_stack() -> None:
    missing: list[str] = []
    for module in ("torch", "transformers", "datasets", "peft", "trl"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError(
            "Missing training dependencies: "
            + ", ".join(missing)
            + ". Install with: uv sync --all-extras"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype(name: str):
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def model_init_kwargs(cfg: ProjectConfig) -> dict[str, Any]:
    """Model-loading options shared by trainers and inference entry points."""
    kwargs: dict[str, Any] = {"dtype": torch_dtype(cfg.model.dtype)}
    if cfg.model.attention_implementation is not None:
        kwargs["attn_implementation"] = cfg.model.attention_implementation
    return kwargs


def lora_config(cfg: ProjectConfig):
    if not cfg.model.use_lora:
        return None
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=cfg.model.lora_rank,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        target_modules=list(cfg.model.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_tokenizer(cfg: ProjectConfig, padding_side: str = "left"):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def encode_generation_prompt(tokenizer, prompt: str, **tokenizer_kwargs):
    """Encode a user prompt exactly as an instruct model expects at generation time."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        **tokenizer_kwargs,
    )


def encode_generation_prompts(tokenizer, prompts: list[str], **tokenizer_kwargs):
    """Batch-encode user prompts with the same instruct-model chat boundary."""
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    return tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        **tokenizer_kwargs,
    )


def generation_stop_token_ids(model, tokenizer) -> set[int]:
    """Return every EOS token honored by model.generate."""
    configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if configured is None:
        configured = tokenizer.eos_token_id
    if isinstance(configured, int):
        return {configured}
    return {int(token_id) for token_id in configured or ()}


def load_adapter_model(
    cfg: ProjectConfig,
    checkpoint: str | Path,
    *,
    device_map: str | None = "auto",
    trainable: bool = False,
):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        device_map=device_map,
        **model_init_kwargs(cfg),
    )
    model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=trainable)
    model.eval()
    return model


@contextlib.contextmanager
def adapter_enabled(model: Any, enabled: bool) -> Iterator[None]:
    if enabled:
        yield
        return
    if not hasattr(model, "disable_adapter"):
        raise TypeError("Base-policy evaluation requires a PEFT model with disable_adapter()")
    with model.disable_adapter():
        yield


def transformer_layers(model: Any):
    """Return the decoder-layer ModuleList for common PEFT-wrapped causal LMs."""
    candidates = (
        ("base_model", "model", "model", "layers"),
        ("model", "model", "layers"),
        ("model", "layers"),
        ("transformer", "h"),
    )
    for path in candidates:
        current = model
        try:
            for name in path:
                current = getattr(current, name)
        except AttributeError:
            continue
        if len(current):
            return current
    raise TypeError("Could not locate transformer decoder layers for this architecture")


def with_seed(cfg: ProjectConfig, seed: int, method: str | None = None) -> ProjectConfig:
    experiment = dataclasses.replace(cfg.experiment, seed=seed)
    training = dataclasses.replace(cfg.training, method=method or cfg.training.method)
    updated = dataclasses.replace(cfg, experiment=experiment, training=training)
    updated.validate()
    return updated
