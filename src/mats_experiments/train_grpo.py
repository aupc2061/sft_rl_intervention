from __future__ import annotations

import argparse
import json
import traceback

from .config import load_config
from .data import build_dataset, to_hf_dataset
from .hf_backend import (
    lora_config,
    load_tokenizer,
    require_training_stack,
    set_seed,
    torch_dtype,
    trainer_reporting,
    trainer_run_name,
    with_seed,
)
from .results import RunDirectory
from .rewards import grpo_numeric_reward


def train(cfg):
    require_training_stack()
    from trl import GRPOConfig, GRPOTrainer

    set_seed(cfg.experiment.seed)
    bundle = build_dataset(cfg.data, cfg.experiment.seed)
    dataset = to_hf_dataset(bundle.train, conversational=False)
    tokenizer = load_tokenizer(cfg, padding_side="left")
    run = RunDirectory.create(cfg, method="grpo")
    args = GRPOConfig(
        output_dir=str(run.root / "checkpoints"),
        seed=cfg.experiment.seed,
        data_seed=cfg.experiment.seed,
        learning_rate=cfg.training.grpo_learning_rate or cfg.training.learning_rate,
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_completion_length=cfg.training.max_completion_length,
        num_generations=cfg.training.num_generations,
        beta=0.0,
        temperature=cfg.evaluation.temperature,
        save_steps=cfg.training.save_steps,
        logging_steps=cfg.training.logging_steps,
        report_to=trainer_reporting(),
        run_name=trainer_run_name(cfg, run.root),
        bf16=cfg.model.dtype == "bfloat16",
        fp16=cfg.model.dtype == "float16",
        model_init_kwargs={"dtype": torch_dtype(cfg.model.dtype)},
    )
    trainer = GRPOTrainer(
        model=cfg.model.name_or_path,
        reward_funcs=grpo_numeric_reward,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config(cfg),
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
    parser = argparse.ArgumentParser(description="Train the online GRPO gatekeeper model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg = with_seed(cfg, args.seed or cfg.experiment.seed, method="grpo")
    run = train(cfg)
    print(json.dumps({"run_directory": str(run.root)}, indent=2))


if __name__ == "__main__":
    main()
