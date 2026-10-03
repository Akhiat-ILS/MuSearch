"""RobustBench evaluation adapter. Downloads require network access and disk space.

Example: python benchmark.py --model Wang2023Better_WRN-28-10 --samples 100
Pilot-trained weights may run the pipeline but do not constitute trained CIFAR policies.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .attack import AttackEnvironment, Controller, run_episode, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Exact RobustBench checkpoint identifier")
    parser.add_argument("--controller", default="runs/pilot/models.pt")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--autoattack", action="store_true", help="Also run official standard AutoAttack"
    )
    parser.add_argument(
        "--apgd", action="store_true", help="Run official APGD CE and DLR at matched iteration cap"
    )
    parser.add_argument("--train-episodes", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", default="runs/cifar10")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    from robustbench.data import load_cifar10
    from robustbench.utils import load_model

    project = Path(__file__).resolve().parent.parent
    output = (project / args.output).resolve()
    if not output.is_relative_to(project):
        raise ValueError("Outputs must stay in this project")
    output.mkdir(parents=True, exist_ok=True)
    # Force datasets/checkpoints into the independent folder.
    model = (
        load_model(
            model_name=args.model,
            dataset="cifar10",
            threat_model="Linf",
            model_dir=str(project / "data/models"),
        )
        .cpu()
        .eval()
    )
    for p in model.parameters():
        p.requires_grad_(False)
    x, y = load_cifar10(
        n_examples=args.samples + args.train_episodes, data_dir=str(project / "data")
    )
    x, y = x.cpu(), y.cpu()
    controller = Controller()
    training = {"episodes": args.train_episodes}
    if args.train_episodes:
        start = time.perf_counter()
        trajectories, stats_collection = [], []
        for indices in np.array_split(np.arange(args.train_episodes), 2):
            for i in indices:
                env = AttackEnvironment(
                    model,
                    x[i : i + 1],
                    int(y[i]),
                    epsilon=8 / 255,
                    steps=args.steps,
                    blocks=args.blocks,
                    seed=args.seed + int(i),
                )
                episode, stats = run_episode(env, controller, "collect", args.seed + int(i))
                stats_collection.append(stats)
                if episode:
                    trajectories.append(episode)
            if trajectories:
                train(controller, trajectories, seed=args.seed)
        training.update(
            seconds=time.perf_counter() - start,
            forwards=sum(r["forwards"] for r in stats_collection),
            backwards=sum(r["backwards"] for r in stats_collection),
            model=args.model,
            input_indices=list(range(args.train_episodes)),
        )
        torch.save(
            {"controller": controller.state_dict(), "training": training}, output / "controller.pt"
        )
    else:
        controller.load_state_dict(
            torch.load(project / args.controller, map_location="cpu", weights_only=True)[
                "controller"
            ]
        )
    x, y = x[args.train_episodes :], y[args.train_episodes :]
    controller.eval()
    result, candidates = {}, {}
    for mode in ["fixed", "random", "reactive", "planning"]:
        records, points = [], []
        for i in range(len(x)):
            env = AttackEnvironment(
                model,
                x[i : i + 1],
                int(y[i]),
                epsilon=8 / 255,
                steps=args.steps,
                blocks=args.blocks,
                seed=args.seed + i,
            )
            _, stats = run_episode(env, controller, mode, args.seed + i)
            stats["index"] = i + args.train_episodes
            records.append(stats)
            points.append(env.best.numpy())
        if not all(r["valid"] for r in records):
            raise RuntimeError("Invalid candidate")
        result[mode] = records
        candidates[mode] = np.concatenate(points)
        print(
            mode, sum(r["success"] for r in records), "/", sum(r["clean_correct"] for r in records)
        )
        np.savez_compressed(
            output / f"{mode}.npz", clean=x.numpy(), labels=y.numpy(), adversarial=candidates[mode]
        )
        (output / "partial_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.apgd:
        from autoattack.autopgd_base import APGDAttack

        for loss in ["ce", "dlr"]:
            attacker = APGDAttack(
                model,
                n_iter=args.steps * args.blocks,
                norm="Linf",
                eps=8 / 255,
                seed=args.seed,
                n_restarts=1,
                loss=loss,
                device="cpu",
            )
            start = time.perf_counter()
            adv = attacker.perturb(x, y)
            elapsed = time.perf_counter() - start
            with torch.no_grad():
                clean_correct = model(x).argmax(1).eq(y)
                success = clean_correct & model(adv).argmax(1).ne(y)
            if not bool((adv - x).abs().max() <= 8 / 255 + 1e-6) or not bool(
                adv.min() >= 0 and adv.max() <= 1
            ):
                raise RuntimeError("APGD candidate outside threat set")
            candidates["apgd_" + loss] = adv.detach().numpy()
            result["apgd_" + loss] = {
                "success": success.tolist(),
                "seconds": elapsed,
                "iterations": args.steps * args.blocks,
                "restarts": 1,
                "budget_note": "Same maximum gradient iterations; batching and forward costs differ",
            }
            print("apgd_" + loss, int(success.sum()), "/", int(clean_correct.sum()))
    if args.autoattack:
        from autoattack import AutoAttack

        start = time.perf_counter()
        adversary = AutoAttack(
            model,
            norm="Linf",
            eps=8 / 255,
            version="standard",
            device="cpu",
            seed=args.seed,
            log_path=str(output / "autoattack.log"),
        )
        adv = adversary.run_standard_evaluation(x, y, bs=16)
        elapsed = time.perf_counter() - start
        with torch.no_grad():
            clean_correct = model(x).argmax(1).eq(y)
            success = clean_correct & model(adv).argmax(1).ne(y)
        if not bool((adv - x).abs().max() <= 8 / 255 + 1e-6) or not bool(
            adv.min() >= 0 and adv.max() <= 1
        ):
            raise RuntimeError("AutoAttack candidate outside threat set")
        candidates["autoattack"] = adv.detach().numpy()
        result["autoattack"] = {
            "success": success.tolist(),
            "seconds": elapsed,
            "budget_note": "Official standard budget; not matched to controller budgets",
        }
    np.savez_compressed(output / "candidates.npz", clean=x.numpy(), labels=y.numpy(), **candidates)
    (output / "results.json").write_text(
        json.dumps(
            {
                "configuration": vars(args),
                "training": training,
                "warning": "Controller training provenance and model disjointness must be reviewed before scientific claims",
                "evaluation": result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
