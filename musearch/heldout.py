"""Frozen, resumable cross-model evaluation. No controller fitting or selection."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .attack import AttackEnvironment, Controller, run_episode

PROJECT = Path(__file__).resolve().parent.parent
MODELS = ["Sehwag2021Proxy_R18", "Rade2021Helper_R18_extra"]


def digest(path):
    # Model checkpoints can be large; hash without loading the whole file.
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sample_indices(count=128, seed=1042):
    # Exclude all training and prior development-evaluation inputs, not just successes.
    return np.random.default_rng(seed).choice(np.arange(40, 10000), count, replace=False)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def predictions(model, x, batch=16):
    with torch.no_grad():
        return torch.cat([model(part).argmax(1) for part in x.split(batch)])


def validate(model, clean, labels, adv, expected):
    if adv.shape != clean.shape or not bool(torch.isfinite(adv).all()):
        raise RuntimeError("Candidate shape or finiteness failure")
    distance = float((adv - clean).abs().max())
    if distance > 8 / 255 + 1e-6 or not bool(adv.min() >= 0 and adv.max() <= 1):
        raise RuntimeError("Candidate outside raw-pixel threat set")
    correct = predictions(model, clean).eq(labels)
    successes = correct & predictions(model, adv).ne(labels)
    if successes.tolist() != expected:
        raise RuntimeError("Candidate/report prediction disagreement")
    return {
        "valid": True,
        "max_linf": distance,
        "successes": int(successes.sum()),
        "clean_correct": int(correct.sum()),
    }


class CountedModel(nn.Module):
    """Count example-level model forward and backward traversals, including batches."""

    def __init__(self, model):
        super().__init__()
        self.model, self.forwards, self.backwards = model, 0, 0

    def forward(self, x):
        self.forwards += len(x)
        output = self.model(x)
        if output.requires_grad:

            def count(gradient):
                self.backwards += len(gradient)

            output.register_hook(count)
        return output


def controller_run(model, controller, x, y, indices, mode, folder, seed):
    path = folder / f"{mode}.npz"
    records_path = folder / f"{mode}.json"
    records, points = [], []
    if path.exists() and records_path.exists():
        records = json.loads(records_path.read_text())
        with np.load(path) as archive:
            points = list(archive["adversarial"])
        if len(records) != len(points):
            raise RuntimeError("Interrupted checkpoint mismatch; inspect before restarting")
    for i in range(len(records), len(x)):
        env = AttackEnvironment(
            model,
            x[i : i + 1],
            int(y[i]),
            epsilon=8 / 255,
            steps=4,
            blocks=8,
            seed=seed + int(indices[i]),
        )
        trajectory, stats = run_episode(env, controller, mode, seed + int(indices[i]))
        stats.update(index=int(indices[i]), actions=[int(t[1]) for t in trajectory])
        records.append(stats)
        points.append(env.best.detach().numpy()[0])
        if (i + 1) % 16 == 0 or i + 1 == len(x):
            np.savez_compressed(path, adversarial=np.stack(points))
            write_json(records_path, records)
            print(
                folder.name,
                mode,
                f"{i + 1}/{len(x)}",
                "successes",
                sum(r["success"] for r in records),
                flush=True,
            )
    return records, torch.from_numpy(np.stack(points))


def reference_run(model, x, y, name, folder, seed):
    path, stats_path = folder / f"{name}.npz", folder / f"{name}.json"
    if path.exists() and stats_path.exists():
        with np.load(path) as archive:
            adv = torch.from_numpy(archive["adversarial"])
        return json.loads(stats_path.read_text()), adv
    counted = CountedModel(model).eval()
    start = time.perf_counter()
    if name == "autoattack":
        from autoattack import AutoAttack

        attacker = AutoAttack(
            counted,
            norm="Linf",
            eps=8 / 255,
            version="standard",
            device="cpu",
            seed=seed,
            log_path=str(folder / "autoattack.log"),
        )
        adv = attacker.run_standard_evaluation(x, y, bs=16)
    else:
        from autoattack.autopgd_base import APGDAttack

        attacker = APGDAttack(
            counted,
            n_iter=32,
            norm="Linf",
            eps=8 / 255,
            seed=seed,
            n_restarts=1,
            loss=name.removeprefix("apgd_"),
            device="cpu",
        )
        parts = []
        for i in range(0, len(x), 16):
            parts.append(attacker.perturb(x[i : i + 16], y[i : i + 16]).detach())
        adv = torch.cat(parts)
    elapsed = time.perf_counter() - start
    adv = adv.detach()
    correct = predictions(model, x).eq(y)
    success = correct & predictions(model, adv).ne(y)
    stats = {
        "success": success.tolist(),
        "clean_correct": correct.tolist(),
        "seconds": elapsed,
        "forwards": counted.forwards,
        "backwards": counted.backwards,
        "budget_note": "official standard configuration"
        if name == "autoattack"
        else "32 optimizer iterations, one restart; forward and backward costs may differ",
    }
    validate(model, x, y, adv, stats["success"])
    np.savez_compressed(path, adversarial=adv.numpy())
    write_json(stats_path, stats)
    print(folder.name, name, "successes", int(success.sum()), flush=True)
    return stats, adv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "core", "autoattack", "audit"], default="all")
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    controller_path = PROJECT / "runs/cifar10_pilot/controller.pt"
    root = PROJECT / "runs/heldout_v1"
    root.mkdir(parents=True, exist_ok=True)
    indices = sample_indices()
    specification = {
        "models": MODELS,
        "samples": 128,
        "indices": indices.tolist(),
        "autoattack_subset_positions": list(range(32)),
        "seed": 42,
        "sampling_seed": 1042,
        "controller_sha256": digest(controller_path),
        "controller_training_model": "Wong2020Fast",
        "excluded_dataset_indices": list(range(40)),
        "epsilon": 8 / 255,
        "steps": 4,
        "blocks": 8,
        "simulations": 24,
        "torch_threads": 2,
        "no_retraining": True,
    }
    manifest = root / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != specification:
        raise RuntimeError("Frozen protocol/controller mismatch")
    if not manifest.exists():
        write_json(manifest, specification)
    checkpoint = torch.load(controller_path, map_location="cpu", weights_only=True)
    if checkpoint["training"]["model"] in MODELS:
        raise RuntimeError("Evaluation model overlaps controller training")
    if set(checkpoint["training"]["input_indices"]) & set(indices.tolist()):
        raise RuntimeError("Evaluation inputs overlap controller training")
    controller = Controller()
    controller.load_state_dict(checkpoint["controller"])
    controller.eval()
    for p in controller.parameters():
        p.requires_grad_(False)
    from robustbench.data import load_cifar10
    from robustbench.utils import load_model

    x_all, y_all = load_cifar10(n_examples=10000, data_dir=str(PROJECT / "data"))
    x, y = x_all[indices], y_all[indices]
    np.savez_compressed(root / "inputs.npz", clean=x.numpy(), labels=y.numpy(), indices=indices)
    for name in [args.model] if args.model else MODELS:
        folder = root / name
        folder.mkdir(exist_ok=True)
        print("Loading held-out model", name, flush=True)
        model = (
            load_model(
                model_name=name,
                dataset="cifar10",
                threat_model="Linf",
                model_dir=str(PROJECT / "data/models"),
            )
            .cpu()
            .eval()
        )
        for p in model.parameters():
            p.requires_grad_(False)
        write_json(
            folder / "model_provenance.json",
            {
                "model": name,
                "checkpoint_sha256": digest(PROJECT / f"data/models/cifar10/Linf/{name}.pt"),
                "controller_sha256": specification["controller_sha256"],
            },
        )
        results, audit = {}, {}
        if args.stage in ("all", "core"):
            for mode in ["fixed", "random", "reactive", "planning"]:
                records, adv = controller_run(model, controller, x, y, indices, mode, folder, 42)
                results[mode] = records
                audit[mode] = validate(model, x, y, adv, [r["success"] for r in records])
            for mode in ["apgd_ce", "apgd_dlr"]:
                stats, adv = reference_run(model, x, y, mode, folder, 42)
                results[mode] = stats
                audit[mode] = validate(model, x, y, adv, stats["success"])
        if args.stage in ("all", "autoattack"):
            stats, adv = reference_run(model, x[:32], y[:32], "autoattack", folder, 42)
            results["autoattack"] = stats
            audit["autoattack"] = validate(model, x[:32], y[:32], adv, stats["success"])
        # Reload all completed methods and audit independently of attack logic.
        for mode in [
            "fixed",
            "random",
            "reactive",
            "planning",
            "apgd_ce",
            "apgd_dlr",
            "autoattack",
        ]:
            if not (folder / f"{mode}.json").exists():
                continue
            saved = json.loads((folder / f"{mode}.json").read_text())
            with np.load(folder / f"{mode}.npz") as archive:
                adv = torch.from_numpy(archive["adversarial"])
            count = 32 if mode == "autoattack" else 128
            expected = (
                [r["success"] for r in saved] if isinstance(saved, list) else saved["success"]
            )
            if len(expected) != count:
                if args.stage == "audit":
                    raise RuntimeError(f"Incomplete method: {mode}")
                continue
            audit[mode] = validate(model, x[:count], y[:count], adv, expected)
            results[mode] = saved
        write_json(folder / "results.json", results)
        write_json(folder / "audit.json", audit)
        print("Audited", name, audit, flush=True)
    if digest(controller_path) != specification["controller_sha256"]:
        raise RuntimeError("Controller was modified")


if __name__ == "__main__":
    main()
