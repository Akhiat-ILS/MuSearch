"""Per-model descriptive and paired statistics for frozen held-out evaluation."""

import argparse
import json
import math

import numpy as np

from .heldout import MODELS, PROJECT, digest, write_json


def paired_statistics(first, second, correct, seed=4042):
    first, second, correct = map(lambda v: np.asarray(v, dtype=bool), (first, second, correct))
    wins = int((first & ~second & correct).sum())
    losses = int((second & ~first & correct).sum())
    difference = (first.astype(float) - second.astype(float))[correct]
    if not len(difference):
        return {"wins": wins, "losses": losses, "difference_pp": None, "ci95_pp": None}
    rng = np.random.default_rng(seed)
    # Resample paired clean-correct inputs, preserving both methods' outcomes.
    boot = difference[rng.integers(len(difference), size=(10000, len(difference)))].mean(1) * 100
    n = wins + losses
    p = (
        min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)
        if n
        else 1.0
    )
    return {
        "wins": wins,
        "losses": losses,
        "difference_pp": float(difference.mean() * 100),
        "ci95_pp": np.quantile(boot, [0.025, 0.975]).tolist(),
        "mcnemar_exact_p": p,
        "note": "Descriptive, unadjusted exploratory comparisons; single training seed",
    }


def wilson(successes, count):
    if not count:
        return [None, None]
    p, z = successes / count, 1.959963984540054
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [(center - radius) * 100, (center + radius) * 100]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    root = PROJECT / "runs/heldout_v1"
    manifest = json.loads((root / "manifest.json").read_text())
    if digest(PROJECT / "runs/cifar10_pilot/controller.pt") != manifest["controller_sha256"]:
        raise RuntimeError("Frozen controller hash changed")
    summary = {}
    completion = {}
    for model_name in MODELS:
        result_path = root / model_name / "results.json"
        completion[model_name] = (
            sorted(json.loads(result_path.read_text())) if result_path.exists() else []
        )
    core_complete = sum(
        all(m in modes for m in ["fixed", "random", "reactive", "planning", "apgd_ce", "apgd_dlr"])
        for modes in completion.values()
    )
    ensemble_complete = sum("autoattack" in modes for modes in completion.values())
    lines = [
        "# Frozen held-out CIFAR-10 evaluation",
        "",
        f"Execution status: core comparisons {core_complete}/2 models complete; standard AutoAttack {ensemble_complete}/2 complete.",
        "",
        "128 fresh sampled inputs per model, excluding all 40 prior development inputs.",
        "The same frozen controller was trained only on Wong2020Fast. No retraining",
        "or parameter selection used held-out results. Linf epsilon 8/255.",
        "Official standard AutoAttack runs on the first 32 preselected sampled inputs.",
        "Controller and APGD run on all 128. Compare equal subsets, not raw totals.",
        "",
        "| Model | Method | Pool | Success / clean-correct | Observed adversarial accuracy | Forwards | Backwards | Seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in MODELS:
        folder = root / name
        if not (folder / "results.json").exists():
            if args.require_complete:
                raise RuntimeError(f"Model incomplete: {name}")
            continue
        results = json.loads((folder / "results.json").read_text())
        audit = json.loads((folder / "audit.json").read_text())
        if args.require_complete and set(results) != {
            "fixed",
            "random",
            "reactive",
            "planning",
            "apgd_ce",
            "apgd_dlr",
            "autoattack",
        }:
            raise RuntimeError(f"Methods incomplete: {name}")
        methods, successes = {}, {}
        correct = np.array([r["clean_correct"] for r in results["planning"]], dtype=bool)
        for mode in [
            "fixed",
            "random",
            "reactive",
            "planning",
            "apgd_ce",
            "apgd_dlr",
            "autoattack",
        ]:
            if mode not in results:
                continue
            result = results[mode]
            if mode not in audit or not audit[mode]["valid"]:
                raise RuntimeError(f"Missing audit: {name}/{mode}")
            if isinstance(result, list):
                success = np.array([r["success"] for r in result], dtype=bool)
                forwards = sum(r["forwards"] for r in result)
                backwards = sum(r["backwards"] for r in result)
                seconds = sum(r["seconds"] for r in result)
            else:
                success = np.array(result["success"], dtype=bool)
                forwards, backwards, seconds = (
                    result["forwards"],
                    result["backwards"],
                    result["seconds"],
                )
            count = len(success)
            denominator = int(correct[:count].sum())
            accuracy = (denominator - int(success.sum())) / count * 100
            methods[mode] = {
                "pool": count,
                "successes": int(success.sum()),
                "clean_correct": denominator,
                "success_rate_ci95_percent": wilson(int(success.sum()), denominator),
                "observed_adversarial_accuracy_percent": accuracy,
                "forwards": forwards,
                "backwards": backwards,
                "seconds": seconds,
            }
            successes[mode] = success
            lines.append(
                f"| {name} | {mode} | {count} | {int(success.sum())}/{denominator} | {accuracy:.2f}% | {forwards} | {backwards} | {seconds:.2f} |"
            )
        comparisons = {}
        for comparator in ["fixed", "random", "reactive"]:
            comparisons["planning_vs_" + comparator] = paired_statistics(
                successes["planning"], successes[comparator], correct
            )
        if "apgd_ce" in successes and "apgd_dlr" in successes:
            union = successes["apgd_ce"] | successes["apgd_dlr"]
            comparisons["planning_vs_apgd_union"] = paired_statistics(
                successes["planning"], union, correct
            )
            comparisons["apgd_union_successes"] = int(union.sum())
        if "autoattack" in successes:
            count = len(successes["autoattack"])
            comparisons["planning_vs_autoattack_subset"] = paired_statistics(
                successes["planning"][:count], successes["autoattack"], correct[:count]
            )
            comparisons["planning_subset_successes"] = int(successes["planning"][:count].sum())
            comparisons["fixed_subset_successes"] = int(successes["fixed"][:count].sum())
            comparisons["reactive_subset_successes"] = int(successes["reactive"][:count].sum())
            comparisons["subset_clean_correct"] = int(correct[:count].sum())
        summary[name] = {"methods": methods, "comparisons": comparisons}
    lines += [
        "",
        "## Same-subset ensemble comparison",
        "",
        "All methods in this table use the same predetermined 32 inputs per model.",
        "Counts exclude initially incorrect predictions from the success denominator.",
        "",
        "| Model | Fixed | Reactive | Planning | Standard AutoAttack | Planning only vs AA | AA only |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in summary.items():
        comparisons = value["comparisons"]
        if "planning_vs_autoattack_subset" not in comparisons:
            continue
        paired = comparisons["planning_vs_autoattack_subset"]
        denominator = comparisons["subset_clean_correct"]
        fixed = comparisons["fixed_subset_successes"]
        reactive = comparisons["reactive_subset_successes"]
        planning = comparisons["planning_subset_successes"]
        aa = value["methods"]["autoattack"]["successes"]
        lines.append(
            f"| {name} | {fixed}/{denominator} | {reactive}/{denominator} | {planning}/{denominator} | {aa}/{denominator} | {paired['wins']} | {paired['losses']} |"
        )
    lines += [
        "",
        "## Paired comparisons",
        "",
        "Differences are attack-success percentage points among clean-correct inputs.",
        "Positive differences favor planning. Confidence intervals use 10,000 paired",
        "bootstrap replicates; they do not account for controller-training randomness.",
        "",
        "| Model | Comparator | Planning only | Comparator only | Difference pp | 95% paired CI pp |",
        "|---|---|---:|---:|---:|---|",
    ]
    for model, value in summary.items():
        for comparison, stats in value["comparisons"].items():
            if not isinstance(stats, dict):
                continue
            lo, hi = stats["ci95_pp"]
            lines.append(
                f"| {model} | {comparison} | {stats['wins']} | {stats['losses']} | {stats['difference_pp']:.2f} | [{lo:.2f}, {hi:.2f}] |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "The core evaluation shows one extra planning success over the APGD union",
        "on Sehwag and identical success sets on Rade. Paired intervals and exact",
        "tests do not establish a reliable improvement. Tree search also does not",
        "consistently improve on the fixed schedule or reactive controller.",
        "The ensemble comparison applies only to its preselected 32-input subsets.",
        "These results do not establish state-of-the-art superiority or equivalence.",
        "",
        "## Limits and cost interpretation",
        "",
        "Controller/APGD iteration caps match at 32; measured total work differs.",
        "APGD references use batches of 16; controller episodes execute individually.",
        "Forward/backward counts are example-level model traversals, not batches.",
        "Independent candidate auditing and model loading are excluded from timings.",
        "Controller timers also exclude the initial clean prediction, while its",
        "forward is counted. Reference timers include their internal clean checks.",
        "Controller/tree-search inference is included in controller attack timings.",
        "The APGD union is the union of two attacks and pays both attacks costs.",
        "Full standard AutoAttack uses a larger budget than the controller.",
        "",
        "Models share the same sampled inputs: do not pool them as 256 independent",
        "observations. One controller training seed and two model checkpoints do not",
        "establish superiority over all current white-box attacks or all defenses.",
        "A zero-width bootstrap interval when outcomes coincide is not proof of",
        "population equivalence. Failure to find an attack does not certify robustness.",
        "",
        "Controller training cost from the development pilot: 25.702 seconds,",
        "882 classifier forwards and 429 backwards; no retraining in this evaluation.",
        "",
        "Frozen manifest: manifest.json. Per-model independent audits: audit.json.",
        "Model/controller provenance and exact candidate arrays are saved beside them.",
    ]
    write_json(root / "summary.json", summary)
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()
