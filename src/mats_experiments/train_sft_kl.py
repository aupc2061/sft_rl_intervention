from __future__ import annotations

import argparse
import json
import traceback

from .config import load_config
from .data import build_dataset, to_hf_dataset
from .hf_backend import (
    lora_config,
    require_training_stack,
    set_seed,
    torch_dtype,
    trainer_reporting,
    trainer_run_name,
    with_seed,
)
from .results import RunDirectory


def _kl_trainer_class():
    import torch
    import torch.nn.functional as functional
    from trl import SFTTrainer

    class ForwardKLTrainer(SFTTrainer):
        def __init__(self, *args, kl_coefficient: float, **kwargs):
            super().__init__(*args, **kwargs)
            self.kl_coefficient = kl_coefficient

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if not hasattr(model, "disable_adapter"):
                raise TypeError("SFT+KL currently requires PEFT/LoRA to evaluate the frozen base")
            base_inputs = {
                key: value
                for key, value in inputs.items()
                if key in {"input_ids", "attention_mask", "position_ids"}
            }
            with torch.no_grad(), model.disable_adapter():
                base_logits = model(**base_inputs, use_cache=False, return_dict=True).logits.detach()
            ce_loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
            base_logp = functional.log_softmax(base_logits[:, :-1].float(), dim=-1)
            ft_logp = functional.log_softmax(outputs.logits[:, :-1].float(), dim=-1)
            token_kl = (base_logp.exp() * (base_logp - ft_logp)).sum(dim=-1)
            labels = inputs.get("labels")
            if labels is not None:
                mask = labels[:, 1:].ne(-100)
            else:
                mask = base_inputs["attention_mask"][:, 1:].bool()
            kl_loss = token_kl.masked_select(mask).mean()
            loss = ce_loss + self.kl_coefficient * kl_loss
            return (loss, outputs) if return_outputs else loss

    return ForwardKLTrainer


def train(cfg):
    require_training_stack()
    from trl import SFTConfig

    if not cfg.model.use_lora:
        raise ValueError("SFT+KL currently requires model.use_lora=true")
    set_seed(cfg.experiment.seed)
    dataset = to_hf_dataset(build_dataset(cfg.data, cfg.experiment.seed).train)
    run = RunDirectory.create(cfg, method="sft_kl")
    args = SFTConfig(
        output_dir=str(run.root / "checkpoints"),
        seed=cfg.experiment.seed,
        data_seed=cfg.experiment.seed,
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_length=cfg.training.max_length,
        completion_only_loss=True,
        save_steps=cfg.training.save_steps,
        logging_steps=cfg.training.logging_steps,
        report_to=trainer_reporting(),
        run_name=trainer_run_name(cfg, run.root),
        bf16=cfg.model.dtype == "bfloat16",
        fp16=cfg.model.dtype == "float16",
        model_init_kwargs={"dtype": torch_dtype(cfg.model.dtype)},
    )
    trainer = _kl_trainer_class()(
        model=cfg.model.name_or_path,
        args=args,
        train_dataset=dataset,
        peft_config=lora_config(cfg),
        kl_coefficient=cfg.training.kl_coefficient,
    )
    try:
        result = trainer.train()
        final_dir = run.root / "checkpoints" / "final"
        trainer.save_model(str(final_dir))
        trainer.save_state()
        run.log_metric({"stage": "train", "status": "completed", **result.metrics})
        run.write_json("status.json", {"status": "completed", "final_checkpoint": str(final_dir)})
    except Exception as exc:
        run.write_json(
            "status.json",
            {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()},
        )
        raise
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train conservative SFT with explicit forward KL")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    original = load_config(args.config)
    cfg = with_seed(original, args.seed or original.experiment.seed, method="sft_kl")
    if cfg.training.kl_coefficient <= 0:
        raise ValueError("training.kl_coefficient must be positive for SFT+KL")
    run = train(cfg)
    print(json.dumps({"run_directory": str(run.root)}, indent=2))


if __name__ == "__main__":
    main()
