from __future__ import annotations

import math
import unittest

import torch

from training.bc_task_vlm.metrics import (
    add_perplexity_metrics,
    build_structured_eval_metrics,
    build_token_accuracy_metrics,
    compute_masked_causal_lm_token_counts,
)


class PerplexityMetricTests(unittest.TestCase):
    def test_adds_perplexity_for_logged_losses(self):
        metrics = add_perplexity_metrics(
            {
                "loss": 1.5,
                "eval_loss": 2.0,
                "train_loss": 0.5,
            }
        )

        self.assertAlmostEqual(metrics["perplexity"], math.exp(1.5))
        self.assertAlmostEqual(metrics["eval_perplexity"], math.exp(2.0))
        self.assertAlmostEqual(metrics["train_perplexity"], math.exp(0.5))

    def test_preserves_existing_perplexity_metrics(self):
        metrics = add_perplexity_metrics(
            {
                "loss": 1.0,
                "perplexity": 123.0,
            }
        )

        self.assertEqual(metrics["perplexity"], 123.0)

    def test_skips_invalid_losses(self):
        metrics = add_perplexity_metrics(
            {
                "loss": "not-a-number",
                "eval_loss": float("nan"),
            }
        )

        self.assertNotIn("perplexity", metrics)
        self.assertNotIn("eval_perplexity", metrics)


class TokenAccuracyMetricTests(unittest.TestCase):
    def test_counts_masked_token_matches(self):
        logits = torch.tensor(
            [
                [
                    [0.0, 5.0, 0.0],
                    [0.0, 0.0, 6.0],
                    [7.0, 0.0, 0.0],
                    [0.0, 8.0, 0.0],
                ]
            ]
        )
        labels = torch.tensor([[-100, 1, 2, 0]])

        correct, total = compute_masked_causal_lm_token_counts(
            logits=logits,
            labels=labels,
        )

        self.assertEqual(correct, 3)
        self.assertEqual(total, 3)

    def test_counts_masked_token_matches_with_selected_logits(self):
        full_logits = torch.tensor(
            [
                [
                    [0.0, 5.0, 0.0],
                    [0.0, 6.0, 0.0],
                    [7.0, 0.0, 0.0],
                    [0.0, 0.0, 8.0],
                    [0.0, 0.0, 9.0],
                ]
            ]
        )
        labels = torch.tensor([[-100, -100, 1, -100, 2]])
        logits_to_keep = torch.tensor([1, 3])
        selected_logits = full_logits.index_select(dim=1, index=logits_to_keep)

        correct, total = compute_masked_causal_lm_token_counts(
            logits=selected_logits,
            labels=labels,
            logits_to_keep=logits_to_keep,
        )

        self.assertEqual(correct, 2)
        self.assertEqual(total, 2)

    def test_builds_accuracy_and_count_metrics(self):
        metrics = build_token_accuracy_metrics(
            correct_tokens=3,
            total_tokens=4,
            metric_key_prefix="eval",
        )

        self.assertEqual(
            metrics,
            {
                "eval_token_accuracy": 0.75,
                "eval_token_correct_count": 3.0,
                "eval_token_count": 4.0,
            },
        )

    def test_uses_zero_accuracy_for_empty_target_set(self):
        metrics = build_token_accuracy_metrics(
            correct_tokens=0,
            total_tokens=0,
            metric_key_prefix="validation",
        )

        self.assertEqual(metrics["validation_token_accuracy"], 0.0)
        self.assertEqual(metrics["validation_token_correct_count"], 0.0)
        self.assertEqual(metrics["validation_token_count"], 0.0)


class StructuredEvalMetricTests(unittest.TestCase):
    def test_builds_counts_and_rates(self):
        metrics = build_structured_eval_metrics(
            total_samples=8,
            parsed_tool_calls=7,
            valid_tool_calls=6,
            exact_tool_matches=5,
            exact_args_matches=4,
            exact_action_matches=3,
        )

        self.assertEqual(metrics["structured_eval_num_samples"], 8.0)
        self.assertEqual(metrics["structured_eval_target_tool_call_count"], 8.0)
        self.assertEqual(metrics["structured_eval_parsed_tool_call_count"], 7.0)
        self.assertEqual(metrics["structured_eval_valid_tool_call_count"], 6.0)
        self.assertEqual(metrics["structured_eval_exact_tool_match_count"], 5.0)
        self.assertEqual(metrics["structured_eval_exact_args_match_count"], 4.0)
        self.assertEqual(metrics["structured_eval_exact_action_step_match_count"], 3.0)
        self.assertAlmostEqual(metrics["structured_eval_tool_call_parse_rate"], 7 / 8)
        self.assertAlmostEqual(metrics["structured_eval_tool_call_valid_rate"], 6 / 8)
        self.assertAlmostEqual(metrics["structured_eval_exact_tool_accuracy"], 5 / 8)
        self.assertAlmostEqual(metrics["structured_eval_exact_args_match_rate"], 4 / 8)
        self.assertAlmostEqual(
            metrics["structured_eval_exact_action_step_match_rate"],
            3 / 8,
        )
        self.assertAlmostEqual(
            metrics["structured_eval_action_prediction_accuracy"],
            3 / 8,
        )

    def test_handles_empty_eval_split(self):
        metrics = build_structured_eval_metrics(
            total_samples=0,
            parsed_tool_calls=0,
            valid_tool_calls=0,
            exact_tool_matches=0,
            exact_args_matches=0,
            exact_action_matches=0,
        )

        self.assertEqual(metrics["structured_eval_action_prediction_accuracy"], 0.0)
        self.assertEqual(metrics["structured_eval_tool_call_parse_rate"], 0.0)
        self.assertEqual(metrics["structured_eval_exact_action_step_match_count"], 0.0)


if __name__ == "__main__":
    unittest.main()
