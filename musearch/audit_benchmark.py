"""Independently recheck saved benchmark candidates against the published model."""

import argparse
import json
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
        raise ValueError("Run must be inside the new project")
    report = json.loads((run / "results.json").read_text())
    from robustbench.utils import load_model

    model = load_model(
        model_name=report["configuration"]["model"],
        dataset="cifar10",
        threat_model="Linf",
        model_dir=str(project / "data/models"),
    ).eval()
    torch.set_num_threads(2)
    archive = np.load(run / "candidates.npz")
    x = torch.from_numpy(archive["clean"])
    y = torch.from_numpy(archive["labels"])
    with torch.no_grad():
        clean_correct = model(x).argmax(1).eq(y)
    audit, successes = {}, {}
    for name in archive.files:
        if name in ("clean", "labels"):
            continue
        adv = torch.from_numpy(archive[name])
        finite = bool(torch.isfinite(adv).all())
        distance = float((adv - x).abs().max())
        valid = finite and distance <= 8 / 255 + 1e-6 and bool(adv.min() >= 0 and adv.max() <= 1)
        with torch.no_grad():
            success = clean_correct & model(adv).argmax(1).ne(y)
        saved = report["evaluation"][name]
        expected = [r["success"] for r in saved] if isinstance(saved, list) else saved["success"]
        agrees = success.tolist() == expected
        audit[name] = {
            "valid": valid,
            "agrees_with_report": agrees,
            "max_linf": distance,
            "successes": int(success.sum()),
            "clean_correct": int(clean_correct.sum()),
        }
        successes[name] = success
        if not valid or not agrees:
            raise RuntimeError(f"Independent audit failed for {name}")
    if "apgd_ce" in successes and "apgd_dlr" in successes:
        union = successes["apgd_ce"] | successes["apgd_dlr"]
        planning = successes["planning"]
        audit["paired_comparison"] = {
            "apgd_union_successes": int(union.sum()),
            "planning_only": int((planning & ~union).sum()),
            "apgd_union_only": int((union & ~planning).sum()),
        }
    if "autoattack" in successes:
        aa, planning = successes["autoattack"], successes["planning"]
        audit["autoattack_comparison"] = {
            "planning_only": int((planning & ~aa).sum()),
            "autoattack_only": int((aa & ~planning).sum()),
        }
    (run / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
