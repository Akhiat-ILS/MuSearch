import numpy as np
import torch
from torch import nn

from musearch.heldout import CountedModel, sample_indices, validate
from musearch.summarize_heldout import paired_statistics


def test_sample_is_reproducible_disjoint_and_unique():
    first, second = sample_indices(), sample_indices()
    assert np.array_equal(first, second)
    assert len(set(first)) == 128 and first.min() >= 40 and first.max() < 10000


def test_reference_counter_counts_examples_and_backward_only_when_used():
    model = CountedModel(nn.Linear(2, 2))
    x = torch.ones(3, 2, requires_grad=True)
    model(x).sum().backward()
    assert model.forwards == 3 and model.backwards == 3
    with torch.no_grad():
        model(x)
    assert model.forwards == 6 and model.backwards == 3


def test_validation_checks_actual_predictions():
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        model.bias.copy_(torch.tensor([-0.5, 0.5]))
    x = torch.tensor([[0.51, 0.4]])
    audit = validate(model, x, torch.tensor([0]), torch.tensor([[0.49, 0.4]]), [True])
    assert audit["valid"] and audit["successes"] == 1


def test_paired_statistics_uses_clean_correct_and_preserves_pairing():
    result = paired_statistics([True, False, True], [False, True, False], [True, True, False])
    assert result["wins"] == 1 and result["losses"] == 1
    assert result["difference_pp"] == 0 and result["mcnemar_exact_p"] == 1
