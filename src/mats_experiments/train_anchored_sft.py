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
    transformer_layers,
    with_seed,
)
from .interventions import restore_base_component
from .results import RunDirectory


def _anchored_trainer_class():
    import torch
    from trl import SFTTrainer

    class AnchoredSFTTrainer(SFTTrainer):
        def __init__(self, *args, anchor_direction, anchor_layer: int, anchor_beta: float, **kwargs):
            super().__init__(*args, **kwargs)
            self.anchor_direction = anchor_direction
            self.anchor_layer = anchor_layer
            self.anchor_beta = anchor_beta

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            if not hasattr(model, "disable_adapter"):
                raise TypeError("Anchored SFT currently requires PEFT/LoRA for exact base activations")
            base_inputs = {
                key: value
                for key, value in inputs.items()
                if key in {"input_ids", "attention_mask", "position_ids"}
            }
            with torch.no_grad(), model.disable_adapter():
                base_output = model(
                    **base_inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            base_hidden = base_output.hidden_states[self.anchor_layer + 1].detach()
            layer = transformer_layers(model)[self.anchor_layer]

            def hook(_module, _hook_inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                restored = restore_base_component(
                    hidden, base_hidden, self.anchor_direction, self.anchor_beta
                )
                return (restored, *output[1:]) if isinstance(output, tuple) else restored

            handle = layer.register_forward_hook(hook)
            try:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            finally:
                handle.remove()

    return AnchoredSFTTrainer


def train(cfg):
    require_training_stack()
    import torch
    from trl import SFTConfig

    if not cfg.model.use_lora:
        raise ValueError("Anchored SFT currently requires model.use_lora=true")
    artifact = torch.load(cfg.training.anchor_direction, map_location="cpu", weights_only=True)
    direction = artifact["directions"][cfg.training.anchor_layer]
    set_seed(cfg.experiment.seed)
    dataset = to_hf_dataset(build_dataset(cfg.data, cfg.experiment.seed).train, conversational=True)
    run = RunDirectory.create(cfg, method="anchored_sft")
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
    trainer_class = _anchored_trainer_class()
    trainer = trainer_class(
        model=cfg.model.name_or_path,
        args=args,
        train_dataset=dataset,
        peft_config=lora_config(cfg),
        anchor_direction=direction,
        anchor_layer=cfg.training.anchor_layer,
        anchor_beta=cfg.training.anchor_beta,
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
    parser = argparse.ArgumentParser(description="Train SFT with a pilot-derived base anchor")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    original = load_config(args.config)
    cfg = with_seed(original, args.seed or original.experiment.seed, method="anchored_sft")
    run = train(cfg)
    print(json.dumps({"run_directory": str(run.root)}, indent=2))


if __name__ == "__main__":
    main()
