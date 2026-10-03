"""Small experimental MuZero-style attack controller; no robustness certificates."""

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class AttackEnvironment:
    actions = 4  # continuation CE, continuation margin, fresh CE, fresh margin

    def __init__(self, model, x, y, epsilon=0.1, steps=4, blocks=8, seed=0):
        if epsilon <= 0 or steps < 1 or blocks < 1:
            raise ValueError("Epsilon, steps and blocks must be positive")
        if (
            x.shape[0] != 1
            or not bool(torch.isfinite(x).all())
            or not bool(x.min() >= 0 and x.max() <= 1)
        ):
            raise ValueError("Provide one finite raw-pixel input in [0,1]")
        self.model, self.x, self.y = model, x.detach().clone(), int(y)
        self.epsilon, self.steps, self.blocks = epsilon, steps, blocks
        self.target = torch.tensor([self.y], device=x.device)
        self.generator = torch.Generator(device=x.device).manual_seed(seed)
        self.current = self.x.clone()
        self.best = self.x.clone()
        self.forwards = self.backwards = self.t = 0
        self.last_gain = 0.0
        logits = self.evaluate(self.x)
        self.clean_correct = int(logits.argmax()) == self.y
        self.best_score = self.score(logits)
        self.clean_score = self.best_score
        self.success = not self.clean_correct

    def evaluate(self, x):
        self.forwards += 1
        with torch.no_grad():
            return self.model(x).flatten()

    def score(self, logits):
        other = logits.clone()
        other[self.y] = -torch.inf
        return float(other.max() - logits[self.y])

    def observation(self):
        scale = max(abs(self.clean_score), 0.1)
        return np.asarray(
            [
                self.best_score / scale,
                self.clean_score,
                self.last_gain / scale,
                self.t / self.blocks,
                float((self.current - self.x).abs().max()) / self.epsilon,
                self.epsilon,
                float(self.success),
                1.0,
            ],
            dtype=np.float32,
        )

    @property
    def done(self):
        return self.success or self.t >= self.blocks

    def project(self, x):
        return (
            torch.maximum(torch.minimum(x, self.x + self.epsilon), self.x - self.epsilon)
            .clamp(0, 1)
            .detach()
        )

    def step(self, action):
        if self.done:
            raise RuntimeError("Terminal episode")
        if action not in range(4):
            raise ValueError("Invalid action")
        previous = self.best_score
        candidate = self.current
        if action >= 2:
            noise = torch.rand(self.x.shape, generator=self.generator, device=self.x.device) * 2 - 1
            candidate = self.project(self.x + self.epsilon * noise)
        for _ in range(self.steps):
            candidate = candidate.detach().requires_grad_(True)
            logits = self.model(candidate)
            self.forwards += 1
            if action % 2 == 0:
                loss = F.cross_entropy(logits, self.target)
            else:
                mask = torch.ones_like(logits, dtype=torch.bool)
                mask[:, self.y] = False
                loss = logits[mask].max() - logits[0, self.y]
            (grad,) = torch.autograd.grad(loss, candidate)
            self.backwards += 1
            candidate = self.project(candidate + self.epsilon / 4 * grad.sign())
            checked = self.evaluate(candidate)
            score = self.score(checked)
            # Preserve actual misclassifications, including ties in logits.
            success = int(checked.argmax()) != self.y
            if score > self.best_score or success:
                self.best, self.best_score = candidate.clone(), score
            if success:
                self.success = True
                break
        self.current = candidate.detach()
        self.t += 1
        self.last_gain = max(0.0, self.best_score - previous)
        reward = float(self.success) - 0.01 + 0.01 * min(self.last_gain, 1.0)
        return self.observation(), reward, self.done

    def validate(self):
        # Independent audit is intentionally outside attack counters.
        with torch.no_grad():
            pred = int(self.model(self.best).argmax())
        bounded = bool((self.best - self.x).abs().max() <= self.epsilon + 1e-6)
        bounded &= bool(self.best.min() >= 0 and self.best.max() <= 1)
        success = self.clean_correct and pred != self.y
        return bounded and success == (self.clean_correct and self.success)


class Controller(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.h = nn.Sequential(nn.Linear(8, width), nn.Tanh())
        self.g = nn.Sequential(nn.Linear(width + 4, width), nn.Tanh())
        self.reward = nn.Linear(width, 1)
        self.policy = nn.Linear(width, 4)
        self.value = nn.Linear(width, 1)

    def prediction(self, latent):
        return self.policy(latent), self.value(latent).squeeze(-1)

    def recurrent(self, latent, action):
        onehot = F.one_hot(action, 4).to(latent.dtype)
        successor = self.g(torch.cat([latent, onehot], -1))
        return successor, self.reward(successor).squeeze(-1)


@dataclass
class Node:
    latent: torch.Tensor
    prior: float = 1.0
    reward: float = 0.0
    visits: int = 0
    total: float = 0.0
    children: dict = None


@torch.no_grad()
def search(controller, observation, simulations=24, depth=4, gamma=0.99):
    if simulations < 1 or depth < 1:
        raise ValueError("Search simulations and depth must be positive")
    device = next(controller.parameters()).device
    root = Node(controller.h(torch.as_tensor(observation, device=device)[None]))

    def expand(node):
        logits, value = controller.prediction(node.latent)
        priors = logits.softmax(-1)[0].tolist()
        # The four imagined successors are independent: evaluate them together.
        actions = torch.arange(4, device=device)
        latents, rewards = controller.recurrent(node.latent.expand(4, -1), actions)
        node.children = {}
        for a, prior in enumerate(priors):
            node.children[a] = Node(latents[a : a + 1], prior, float(rewards[a]))
        return float(value)

    expand(root)
    for _ in range(simulations):
        node, path = root, [root]
        for _ in range(depth):
            if node.children is None:
                break

            def ucb(child):
                q = child.total / child.visits if child.visits else 0.0
                return (
                    child.reward
                    + gamma * q
                    + 1.5 * child.prior * math.sqrt(node.visits + 1) / (1 + child.visits)
                )

            node = max(node.children.values(), key=ucb)
            path.append(node)
        value = (
            expand(node) if node.children is None else float(controller.prediction(node.latent)[1])
        )
        for visited in reversed(path):
            visited.visits += 1
            visited.total += value
            value = visited.reward + gamma * value
    visits = np.array([root.children[a].visits for a in range(4)], dtype=np.float32)
    return visits / visits.sum()


def train(controller, episodes, epochs=30, seed=0):
    """One-step recurrent MuZero losses with actual discounted return targets."""
    if epochs < 1:
        raise ValueError("Training epochs must be positive")
    rng = np.random.default_rng(seed)
    data = []
    for trajectory in episodes:
        value = 0.0
        for obs, action, reward, following, policy, done in reversed(trajectory):
            value = reward + 0.99 * value
            data.append((obs, action, reward, following, policy, value, done))
    if not data:
        raise ValueError("Training requires at least one transition")
    device = next(controller.parameters()).device
    obs, acts, rewards, following, policies, values, done = map(np.asarray, zip(*data))
    tensors = [
        torch.as_tensor(a, device=device)
        for a in [obs, acts, rewards, following, policies, values, done]
    ]
    opt = torch.optim.Adam(controller.parameters(), lr=0.002)
    losses = []
    for _ in range(epochs):
        for batch in np.array_split(rng.permutation(len(data)), max(1, math.ceil(len(data) / 64))):
            o, a, r, nxt, pi, v, terminal = [t[batch] for t in tensors]
            latent = controller.h(o.float())
            logits, predicted = controller.prediction(latent)
            successor, estimated_reward = controller.recurrent(latent, a.long())
            _, next_value = controller.prediction(successor)
            next_target = torch.where(terminal, torch.zeros_like(v), (v - r) / 0.99)
            loss = -(pi.float() * logits.log_softmax(-1)).sum(-1).mean()
            loss += F.mse_loss(predicted, v.float()) + F.mse_loss(estimated_reward, r.float())
            loss += F.mse_loss(next_value, next_target.float())
            loss += 0.1 * F.mse_loss(successor, controller.h(nxt.float()).detach())
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
    return {"transitions": len(data), "final_loss": losses[-1]}


def run_episode(environment, controller, mode="planning", seed=0, simulations=24):
    if mode not in ("planning", "collect", "reactive", "random", "fixed"):
        raise ValueError(f"Unknown attack mode: {mode}")
    rng, trajectory = np.random.default_rng(seed), []
    start = time.perf_counter()
    while not environment.done:
        obs = environment.observation()
        if mode in ("planning", "collect"):
            pi = search(controller, obs, simulations, min(4, environment.blocks - environment.t))
            action = int(rng.choice(4, p=pi)) if mode == "collect" else int(pi.argmax())
        elif mode == "reactive":
            with torch.no_grad():
                device = next(controller.parameters()).device
                logits, _ = controller.prediction(
                    controller.h(torch.as_tensor(obs, device=device)[None])
                )
            action = int(logits.argmax())
            pi = np.eye(4)[action]
        else:
            action = int(rng.integers(4)) if mode == "random" else environment.t % 2
            pi = np.eye(4)[action]
        following, reward, done = environment.step(action)
        trajectory.append((obs, action, reward, following, pi, done))
    elapsed = time.perf_counter() - start
    return trajectory, {
        "success": environment.clean_correct and environment.success,
        "clean_correct": environment.clean_correct,
        "forwards": environment.forwards,
        "backwards": environment.backwards,
        "seconds": elapsed,
        "valid": environment.validate(),
    }
