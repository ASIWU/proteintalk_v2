from __future__ import annotations

import numpy as np
import pytest
import torch

from utils import attribution_utils as exp34_attr


class LinearResponseModel(torch.nn.Module):
    def __init__(self, weights: list[float], bias: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.bias = float(bias)

    def forward(self, batch: dict[str, torch.Tensor]):
        expression = batch["control_expression"]
        logits = expression @ self.weights + self.bias
        return expression, logits.unsqueeze(-1), logits.unsqueeze(-1)


def test_gradient_delta_and_ig_are_exact_for_linear_model() -> None:
    model = LinearResponseModel([2.0, -3.0, 0.5], bias=1.25)
    inputs = torch.tensor([[2.0, 4.0, -1.0], [1.0, -2.0, 3.0]])
    baseline = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    batch = {"control_expression": inputs}

    logits, gradient, gradient_delta = exp34_attr.gradient_x_delta(
        model, batch, inputs, baseline
    )
    integrated, input_logits, baseline_logits = exp34_attr.integrated_gradients(
        model, batch, inputs, baseline, steps=8
    )
    expected_gradient = torch.tensor([[2.0, -3.0, 0.5], [2.0, -3.0, 0.5]])
    expected = expected_gradient * (inputs - baseline)

    assert torch.equal(gradient, expected_gradient)
    assert torch.equal(gradient_delta, expected)
    assert torch.equal(integrated, expected)
    assert torch.equal(logits, input_logits)
    logit_delta, absolute_error, relative_error = exp34_attr.completeness_errors(
        integrated, input_logits, baseline_logits
    )
    assert torch.allclose(integrated.sum(dim=1), logit_delta)
    assert torch.equal(absolute_error, torch.zeros_like(absolute_error))
    assert torch.equal(relative_error, torch.zeros_like(relative_error))


def test_batched_gradient_sum_does_not_mix_samples() -> None:
    model = LinearResponseModel([1.0, 2.0])
    inputs = torch.tensor([[1.0, 10.0], [5.0, -4.0], [9.0, 2.0]])
    baseline = torch.zeros_like(inputs)
    batch = {"control_expression": inputs}
    _, gradient, _ = exp34_attr.gradient_x_delta(model, batch, inputs, baseline)
    assert torch.equal(gradient, torch.tensor([[1.0, 2.0]]).repeat(3, 1))


def test_nanmedian_baseline_fills_only_all_nan_features() -> None:
    matrix = np.asarray(
        [
            [1.0, np.nan, np.nan, 10.0],
            [3.0, 5.0, np.nan, np.nan],
            [9.0, 7.0, np.nan, 14.0],
        ],
        dtype=np.float32,
    )
    baseline, counts, all_nan = exp34_attr.nanmedian_baseline(matrix)
    assert np.array_equal(baseline, np.asarray([3.0, 6.0, 0.0, 12.0], dtype=np.float32))
    assert np.array_equal(counts, np.asarray([3, 2, 0, 2]))
    assert np.array_equal(all_nan, np.asarray([False, False, True, False]))


def test_stable_abs_rank_uses_protein_index_as_tie_break() -> None:
    values = np.asarray([1.0, -1.0, 0.5], dtype=np.float32)
    protein_indices = np.asarray([5, 2, 1], dtype=np.int64)
    ranks = exp34_attr.stable_abs_ranks(values, protein_indices)
    assert np.array_equal(ranks, np.asarray([2, 1, 3]))


def test_integrated_gradients_rejects_invalid_steps() -> None:
    model = LinearResponseModel([1.0])
    inputs = torch.ones((1, 1))
    with pytest.raises(ValueError, match="steps must be positive"):
        exp34_attr.integrated_gradients(
            model,
            {"control_expression": inputs},
            inputs,
            torch.zeros_like(inputs),
            steps=0,
        )
