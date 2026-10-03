import numpy as np
import pytest
import torch
from torch import nn

from musearch.attack import AttackEnvironment, Controller, run_episode, search, train


def environment():
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        model.bias.copy_(torch.tensor([-0.5, 0.5]))
    return AttackEnvironment(model, torch.tensor([[0.6, 0.4]]), 0, epsilon=0.2, steps=4, blocks=4)


def test_actual_success_and_frozen_classifier():
    env = environment()
    before = [p.clone() for p in env.model.parameters()]
    env.step(0)
    assert env.success and env.validate()
    assert env.forwards == 1 + 2 * env.backwards
    assert all(torch.equal(a, b) for a, b in zip(before, env.model.parameters()))
    assert all(p.grad is None for p in env.model.parameters())
    with pytest.raises(RuntimeError):
        env.step(0)


@pytest.mark.parametrize("action", range(4))
def test_projection_and_validation(action):
    env = environment()
    env.step(action)
    assert env.validate()
    assert (env.best - env.x).abs().max() <= 0.200001


def test_search_does_not_query_classifier():
    env, controller = environment(), Controller()
    before = env.forwards
    policy = search(controller, env.observation(), simulations=16)
    assert env.forwards == before
    assert np.isclose(policy.sum(), 1) and (policy >= 0).all()


@pytest.mark.parametrize("seed", [0, 4, 42])
def test_batched_search_preserves_sequential_visit_distribution(seed):
    class SequentialController(Controller):
        def recurrent(self, latent, action):
            results = [
                super(SequentialController, self).recurrent(latent[i : i + 1], action[i : i + 1])
                for i in range(len(action))
            ]
            return torch.cat([r[0] for r in results]), torch.cat([r[1] for r in results])

    torch.manual_seed(seed)
    batched, sequential = Controller(), SequentialController()
    sequential.load_state_dict(batched.state_dict())
    observation = environment().observation()
    np.testing.assert_array_equal(search(batched, observation), search(sequential, observation))


def test_invalid_modes_and_empty_training_fail_clearly():
    controller = Controller()
    with pytest.raises(ValueError, match="Unknown attack mode"):
        run_episode(environment(), controller, mode="typo")
    with pytest.raises(ValueError, match="at least one transition"):
        train(controller, [])


def test_legacy_api_imports_remain_available():
    from heldout import CountedModel, sample_indices
    from muzero_attack import AttackEnvironment as LegacyEnvironment
    from summarize_heldout import paired_statistics

    assert LegacyEnvironment is AttackEnvironment
    assert len(sample_indices()) == 128
    assert callable(paired_statistics) and issubclass(CountedModel, nn.Module)


def test_training_updates_controller_with_finite_loss():
    torch.manual_seed(4)
    controller = Controller()
    episode, stats = run_episode(environment(), controller, "collect", simulations=8)
    before = controller.h[0].weight.detach().clone()
    result = train(controller, [episode], epochs=2)
    assert stats["valid"] and np.isfinite(result["final_loss"])
    assert not torch.equal(before, controller.h[0].weight)
