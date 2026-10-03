"""Append official standard AutoAttack to a completed pilot without retraining."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/cifar10_pilot")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    run = (project / args.run).resolve()
    if not run.is_relative_to(project):
        raise ValueError("Run must be inside project")
    report = json.loads((run / "results.json").read_text())
    with np.load(run / "candidates.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    from autoattack import AutoAttack
    from robustbench.utils import load_model

    torch.set_num_threads(2)
    model = (
        load_model(
            model_name=report["configuration"]["model"],
            dataset="cifar10",
            threat_model="Linf",
            model_dir=str(project / "data/models"),
        )
        .cpu()
        .eval()
    )
    for p in model.parameters():
        p.requires_grad_(False)
    x, y = torch.from_numpy(arrays["clean"]), torch.from_numpy(arrays["labels"])
    adversary = AutoAttack(
        model,
        norm="Linf",
        eps=8 / 255,
        version="standard",
        device="cpu",
        seed=report["configuration"]["seed"],
        log_path=str(run / "autoattack.log"),
    )
    start = time.perf_counter()
    adv = adversary.run_standard_evaluation(x, y, bs=16)
    elapsed = time.perf_counter() - start
    with torch.no_grad():
        clean_correct = model(x).argmax(1).eq(y)
        success = clean_correct & model(adv).argmax(1).ne(y)
    if (
        not bool(torch.isfinite(adv).all())
        or not bool((adv - x).abs().max() <= 8 / 255 + 1e-6)
        or not bool(adv.min() >= 0 and adv.max() <= 1)
    ):
        raise RuntimeError("Invalid AutoAttack candidate")
    arrays["autoattack"] = adv.detach().numpy()
    np.savez_compressed(run / "candidates.npz", **arrays)
    report["evaluation"]["autoattack"] = {
        "success": success.tolist(),
        "seconds": elapsed,
        "budget_note": "Official standard configuration; larger budget than pilot controller",
    }
    (run / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("AutoAttack successes:", int(success.sum()), "/", int(clean_correct.sum()))


if __name__ == "__main__":
    main()
