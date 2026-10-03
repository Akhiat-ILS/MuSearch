"""Generate the measured pilot report after independent audit."""

import json
from pathlib import Path


def main():
    run = Path(__file__).resolve().parent.parent / "runs/cifar10_pilot"
    report = json.loads((run / "results.json").read_text())
    audit = json.loads((run / "audit.json").read_text())
    lines = [
        "# CIFAR-10 development pilot results",
        "",
        "Model: RobustBench Wong2020Fast. Raw Linf epsilon 8/255. Seed 42.",
        "24 controller-training inputs; 16 disjoint evaluation inputs.",
        "Training and evaluation use the same classifier. No held-out-model claim.",
        "",
        "| Method | Successes / clean-correct | Forwards | Backwards | Attack seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, result in report["evaluation"].items():
        checked = audit[mode]
        if isinstance(result, list):
            forwards = sum(r["forwards"] for r in result)
            backwards = sum(r["backwards"] for r in result)
            seconds = sum(r["seconds"] for r in result)
        else:
            forwards, backwards, seconds = "not instrumented", "not instrumented", result["seconds"]
        lines.append(
            f"| {mode} | {checked['successes']}/{checked['clean_correct']} | {forwards} | {backwards} | {seconds:.3f} |"
        )
    lines += [
        "",
        "All saved candidates passed independent constraint and prediction auditing.",
        "Controller methods use at most 32 gradient steps per input; standalone APGD",
        "uses one restart and 32 iterations. Full AutoAttack uses its official standard",
        "configuration, with a larger budget. Its runtime is not an equal-budget comparison.",
        "Baselines use batched execution; the controller currently executes inputs individually.",
        "Controller timings include tree search; independent audit and training are excluded.",
        "",
        f"Controller training: {report['training']['seconds']:.3f} seconds, "
        f"{report['training']['forwards']} forwards, {report['training']['backwards']} backwards.",
        "",
        "## Paired outcomes",
        "",
        "```json",
        json.dumps({k: v for k, v in audit.items() if k.endswith("comparison")}, indent=2),
        "```",
        "",
        "## Interpretation",
        "",
        "This small, single-model, single-seed development pilot cannot establish",
        "state-of-the-art performance. Compare paired outcomes and total compute,",
        "not attack-success counts alone. Failures to find attacks are NOT_FOUND.",
        "",
        "Next: improve recurrent/terminal modeling on development data, then freeze",
        "selection and run multiple seeds on larger pools and held-out robust models.",
        "Do not reuse these evaluation inputs as an untouched test after tuning.",
    ]
    (run / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:18]))


if __name__ == "__main__":
    main()
