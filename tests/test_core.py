from __future__ import annotations

import tempfile
import unittest
import importlib.util
from pathlib import Path

from mats_experiments.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    ProjectConfig,
    TrainingConfig,
    load_config,
)
from mats_experiments.data import build_synthetic_arithmetic
from mats_experiments.grpo_viability import summarize_groups
from mats_experiments.hf_backend import encode_generation_prompt, generation_stop_token_ids
from mats_experiments.numerics import (
    bootstrap_interval,
    globality_ratio,
    kl_divergence,
    mean_pairwise_cosine,
)
from mats_experiments.results import RunDirectory
from mats_experiments.mvp_analysis import _derive_e3_row, _matched_pair, _select_layer
from mats_experiments.rewards import (
    arithmetic_domain_intrusion,
    exact_numeric_reward,
    extract_numeric_answer,
)


class RewardTests(unittest.TestCase):
    def test_answer_formats(self):
        self.assertEqual(extract_numeric_answer("Reasoning. Answer: 1,250"), "1250")
        self.assertEqual(extract_numeric_answer(r"Thus \\boxed{-3.50}"), "-3.5")
        self.assertEqual(exact_numeric_reward("Answer: 04", "4"), 1.0)
        self.assertEqual(exact_numeric_reward("Answer: 5", "4"), 0.0)

    def test_intrusion_proxy(self):
        self.assertEqual(arithmetic_domain_intrusion("A quiet discussion of wetlands."), 0.0)
        self.assertGreater(arithmetic_domain_intrusion("Compute 2 + 2. Answer: 4"), 0.5)


class PromptFormattingTests(unittest.TestCase):
    def test_generation_prompt_uses_chat_template(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return {"messages": messages, "kwargs": kwargs}

        encoded = encode_generation_prompt(FakeTokenizer(), "hello", return_tensors="pt")
        self.assertEqual(encoded["messages"], [{"role": "user", "content": "hello"}])
        self.assertTrue(encoded["kwargs"]["tokenize"])
        self.assertTrue(encoded["kwargs"]["add_generation_prompt"])
        self.assertTrue(encoded["kwargs"]["return_dict"])

    def test_all_model_stop_tokens_are_recognized(self):
        model = type("Model", (), {"generation_config": type("Config", (), {"eos_token_id": [7, 9]})()})()
        tokenizer = type("Tokenizer", (), {"eos_token_id": 3})()
        self.assertEqual(generation_stop_token_ids(model, tokenizer), {7, 9})


class NumericalTests(unittest.TestCase):
    def test_kl(self):
        self.assertAlmostEqual(kl_divergence([0.5, 0.5], [0.5, 0.5]), 0.0)
        self.assertGreater(kl_divergence([0.9, 0.1], [0.1, 0.9]), 0.0)

    def test_globality(self):
        aligned = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        cancelled = [[1.0, 0.0], [-1.0, 0.0]]
        self.assertAlmostEqual(globality_ratio(aligned), 1.0)
        self.assertAlmostEqual(globality_ratio(cancelled), 0.0)
        self.assertAlmostEqual(mean_pairwise_cosine(aligned), 1.0)

    def test_bootstrap_is_deterministic(self):
        first = bootstrap_interval([0.0, 1.0, 1.0], samples=100, seed=7)
        second = bootstrap_interval([0.0, 1.0, 1.0], samples=100, seed=7)
        self.assertEqual(first, second)

    def test_mvp_selection_uses_discovery_only_and_requires_e1(self):
        sft = {
            "layers": {
                "0": {"rho_discovery": 0.30, "rho_confirmation": 0.01, "split_half_cosine": 0.2},
                "1": {"rho_discovery": 0.20, "rho_confirmation": 0.90, "split_half_cosine": 0.8},
            }
        }
        rl = {
            "layers": {
                "0": {"rho_discovery": 0.10, "rho_confirmation": 0.50, "split_half_cosine": 0.1},
                "1": {"rho_discovery": 0.15, "rho_confirmation": 0.01, "split_half_cosine": 0.1},
            }
        }
        selected = _select_layer(sft, rl, minimum_gap=0.02, passes_e1_gate=False)
        self.assertEqual(selected["layer"], 0)
        self.assertTrue(selected["passes_trace_gate"])
        self.assertFalse(selected["passes_e3_gate"])

    def test_mvp_checkpoint_matching(self):
        sft = [
            {"checkpoint": "checkpoint-1", "checkpoint_path": "s1", "task_accuracy": 0.2, "forward_kl": 0.4},
            {"checkpoint": "checkpoint-2", "checkpoint_path": "s2", "task_accuracy": 0.5, "forward_kl": 0.7},
        ]
        rl = [
            {"checkpoint": "checkpoint-1", "checkpoint_path": "r1", "task_accuracy": 0.51, "forward_kl": 0.2},
        ]
        matched = _matched_pair(sft, rl, max_accuracy_gap=0.05, min_kl_gap=0.0)
        self.assertEqual(matched["sft_checkpoint_path"], "s2")
        self.assertTrue(matched["passes_e1_gate"])

    def test_e3_primary_metric_is_kl_reduction_toward_sft(self):
        baseline = {
            "sft_to_intervention_kl": 0.8,
            "forward_kl": 0.2,
            "task_accuracy": 0.5,
            "domain_intrusion": 0.1,
        }
        row = {
            "label": "SFT trace",
            "scale": 1.0,
            "sft_to_intervention_kl": 0.5,
            "forward_kl": 0.4,
            "task_accuracy": 0.45,
            "domain_intrusion": 0.2,
        }
        derived = _derive_e3_row(row, baseline)
        self.assertAlmostEqual(derived["delta_toward_sft"], 0.3)
        self.assertAlmostEqual(derived["delta_base_kl"], 0.2)
        self.assertAlmostEqual(derived["delta_task_accuracy"], -0.05)

    def test_grpo_viability_gate_uses_mixed_reward_groups(self):
        def group(rewards):
            return {
                "generations": [
                    {
                        "reward": reward,
                        "parsed": True,
                        "truncated": False,
                        "completion_tokens": 20,
                    }
                    for reward in rewards
                ]
            }

        groups = [group([1, 0, 0, 0]), group([0, 1, 0, 0])]
        groups.extend(group([0, 0, 0, 0]) for _ in range(8))
        summary = summarize_groups(
            groups,
            min_parsed_rate=0.8,
            min_mixed_group_fraction=0.15,
            max_all_zero_fraction=0.8,
            max_all_one_fraction=0.8,
            max_truncated_fraction=0.25,
        )
        self.assertTrue(summary["suitable_for_grpo"])
        self.assertAlmostEqual(summary["metrics"]["mixed_group_fraction"], 0.2)
        self.assertAlmostEqual(summary["metrics"]["all_zero_group_fraction"], 0.8)


class DataTests(unittest.TestCase):
    def test_splits_are_disjoint_and_deterministic(self):
        cfg = DataConfig(train_size=20, task_test_size=10, probe_size=8, old_size=10)
        first = build_synthetic_arithmetic(cfg, seed=3)
        second = build_synthetic_arithmetic(cfg, seed=3)
        self.assertEqual(first, second)
        identifiers = [
            {example.example_id for example in split}
            for split in (first.train, first.task_test, first.probe, first.old)
        ]
        for i, left in enumerate(identifiers):
            for right in identifiers[i + 1 :]:
                self.assertTrue(left.isdisjoint(right))


@unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML is not installed")
class ConfigFileTests(unittest.TestCase):
    def test_checked_in_configs_parse(self):
        root = Path(__file__).resolve().parents[1]
        smoke = load_config(root / "configs" / "exp1_synthetic_smoke.yaml")
        real = load_config(root / "configs" / "exp1_qwen_gsm8k.yaml")
        mvp = load_config(root / "configs" / "mvp_16h_qwen05b_gsm8k.yaml")
        viability = load_config(root / "configs" / "gsm8k_grpo_viability.yaml")
        self.assertEqual(smoke.data.kind, "synthetic_arithmetic")
        self.assertEqual(real.model.name_or_path, "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(mvp.model.name_or_path, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(mvp.experiment.seed, 1)
        self.assertEqual(mvp.training.grpo_learning_rate, 1e-6)
        self.assertEqual(mvp.traces.top_k_layers, 1)
        self.assertEqual(viability.training.method, "grpo")
        self.assertEqual(viability.data.train_size, 64)


class ResultContractTests(unittest.TestCase):
    def test_run_directory_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = ProjectConfig(
                experiment=ExperimentConfig("test", output_root=directory, seed=2),
                model=ModelConfig("dummy"),
                training=TrainingConfig(method="sft"),
            )
            run = RunDirectory.create(cfg)
            run.log_metric({"metric": 1.0})
            self.assertTrue((run.root / "config.resolved.json").is_file())
            self.assertTrue((run.root / "manifest.json").is_file())
            self.assertTrue(run.metrics_path.is_file())
            self.assertTrue((run.root / "checkpoints").is_dir())
            self.assertTrue((run.root / "artifacts").is_dir())
            self.assertTrue((run.root / "plots").is_dir())


if __name__ == "__main__":
    unittest.main()
