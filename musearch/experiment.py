"""Synthetic pipeline pilot, explicitly not a RobustBench evaluation."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .attack import AttackEnvironment, Controller, run_episode, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=160)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", default="runs/pilot")
    args = parser.parse_args()
    if args.episodes < 2 or args.samples < 1:
        parser.error("Need at least two training episodes and one evaluation sample")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))
    generator = torch.Generator().manual_seed(args.seed + 100)
    inputs = torch.rand(2048, 16, generator=generator)
    labels = (inputs[:, :4].sum(1) > 2).long()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(100):
        loss = nn.functional.cross_entropy(model(inputs), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    controller = Controller()

    def make_env(x, y, seed):
        return AttackEnvironment(model, x[None], int(y), epsilon=0.1, steps=4, blocks=8, seed=seed)

    development = torch.rand(args.episodes, 16, generator=generator)
    development_labels = (development[:, :4].sum(1) > 2).long()
    training_start = time.perf_counter()
    all_trajectories, collection = [], []
    training_stats = []
    # Two self-play rounds: the second uses the first trained planner.
    for indices in np.array_split(np.arange(args.episodes), 2):
        for i in indices:
            episode, stats = run_episode(
                make_env(development[i], development_labels[i], args.seed + int(i)),
                controller,
                "collect",
                args.seed + int(i),
            )
            collection.append(stats)
            if episode:
                all_trajectories.append(episode)
        if all_trajectories:
            training_stats.append(train(controller, all_trajectories, seed=args.seed))
    training_seconds = time.perf_counter() - training_start
    controller.eval()
    # Evaluation is generated after training, with its own independent RNG.
    test_generator = torch.Generator().manual_seed(args.seed + 10000)
    test = torch.rand(args.samples, 16, generator=test_generator)
    test_labels = (test[:, :4].sum(1) > 2).long()
    results, candidates = {}, {}
    for mode in ["fixed", "random", "reactive", "planning"]:
        records, points = [], []
        for i in range(args.samples):
            env = make_env(test[i], test_labels[i], args.seed + 20000 + i)
            _, stats = run_episode(env, controller, mode, args.seed + 30000 + i)
            stats["index"] = i
            records.append(stats)
            points.append(env.best.numpy())
        results[mode] = {
            "successes": sum(r["success"] for r in records),
            "clean_correct": sum(r["clean_correct"] for r in records),
            "forwards": sum(r["forwards"] for r in records),
            "backwards": sum(r["backwards"] for r in records),
            "seconds": sum(r["seconds"] for r in records),
            "all_valid": all(r["valid"] for r in records),
            "records": records,
        }
        candidates[mode] = np.concatenate(points)
    if not all(r["all_valid"] for r in results.values()):
        raise RuntimeError("Candidate validation failed")
    output = Path(args.output).resolve()
    project = Path(__file__).resolve().parent.parent
    if not output.is_relative_to(project):
        raise ValueError("Outputs must stay inside this independent project")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "synthetic pipeline pilot; no benchmark superiority claim",
        "configuration": vars(args),
        "training": {
            "seconds": training_seconds,
            "forwards": sum(r["forwards"] for r in collection),
            "backwards": sum(r["backwards"] for r in collection),
            "rounds": training_stats,
        },
        "evaluation": results,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        output / "candidates.npz", clean=test.numpy(), labels=test_labels.numpy(), **candidates
    )
    torch.save(
        {"controller": controller.state_dict(), "classifier": model.state_dict()},
        output / "models.pt",
    )
    for mode, result in results.items():
        print(mode, {k: v for k, v in result.items() if k != "records"})
    print("Saved:", output)


if __name__ == "__main__":
    main()
