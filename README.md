# MuSearch

**MuZero-inspired learned search for adversarial testing.**

MuSearch trains a latent representation, recurrent dynamics/reward model, policy,
and value function to select projected-gradient attack blocks. PUCT tree search
plans in the learned latent space; selected actions run against the real classifier.
Every reported adversarial success is checked using actual classifier predictions
and pixel/perturbation bounds. This is a research prototype: measured results do
not establish superiority or general equivalence to state-of-the-art attacks.

## Measured findings

The frozen controller was evaluated on 128 fresh CIFAR-10 inputs on each of two
held-out RobustBench models at Linf epsilon 8/255. Both models use the same input
pool. Numbers below count attacks on initially correct predictions.

| Held-out model | MuSearch planning | APGD-CE/DLR union |
|---|---:|---:|
| Sehwag2021Proxy_R18 | 36/109 | 35/109 |
| Rade2021Helper_R18_extra | 40/117 | 40/117 |

On preselected **32-input subsets**, planning and official standard AutoAttack
found exactly the same successful attacks: 10/27 and 11/31, respectively. This
small, single-training-seed evaluation shows no statistically reliable improvement
over simpler schedules. AutoAttack was not run on all 128 inputs.

All 1,600 saved held-out candidates passed independent validation. See the
[full report](results/heldout.md), [machine-readable summary](results/heldout_summary.json),
[frozen protocol](docs/HELDOUT_PROTOCOL.md), and [development results](results/development.md).
Published timings describe the original evaluated implementation, before this
packaging/efficiency cleanup; they are not new performance measurements.

## Install and run

Python 3.11 or newer is required. From a clone of this repository:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m musearch.experiment --episodes 80 --samples 64 --output runs/pilot
python -m pytest -q
```

The synthetic pilot verifies the pipeline; it does not train a CIFAR-10 controller.
For the real-model evaluation, install optional benchmark dependencies:

```bash
python -m pip install -e ".[benchmark]"
python -m musearch.benchmark --model Wong2020Fast --train-episodes 24 --samples 16 --steps 4 --blocks 8 --seed 42 --apgd --output runs/cifar10_pilot
python -m musearch.heldout --stage core
python -m musearch.heldout --stage autoattack
python -m musearch.heldout --stage audit
python -m musearch.summarize_heldout --require-complete
```

Run commands from the repository root. The held-out runner requires the controller
produced at `runs/cifar10_pilot/controller.pt`, freezes its hash and sample indices,
and resumes completed methods. Real-model runs download datasets/checkpoints and
can take over an hour on CPU. Generated data, weights, candidates and dependencies
stay local and are excluded from Git. Compact historical results are in `results/`.
Original checkpoints/candidate archives are not bundled; regenerate them with the
commands above. Newly trained weights need not match historical hashes exactly.

Console commands: `musearch-pilot`, `musearch-benchmark`, `musearch-heldout`, and
`musearch-report`. Root scripts remain compatibility entry points.

## Structure

| Path | Purpose |
|---|---|
| `musearch/attack.py` | Attack environment, controller, latent search and training |
| `musearch/experiment.py` | Synthetic training/evaluation pilot |
| `musearch/benchmark.py` | CIFAR-10 development training and references |
| `musearch/heldout.py` | Frozen, resumable evaluation and independent audit |
| `musearch/summarize_heldout.py` | Paired statistics and report generation |
| `tests/` | Bounds, classifier checks, training, search and evaluation tests |
| `docs/` | Development and held-out protocols |
| `results/` | Historical measured reports, records and provenance |

## Method and limits

Four actions continue or restart CE/margin projected-gradient blocks. The
controller observes eight scalar features; training uses one-step recurrent
losses, real discounted returns and search visit targets. This is inspired by
MuZero rather than a full reproduction of its multi-step training algorithm.

Attacks process one input at a time. Classifier normalization must be inside the
model; inputs must be finite raw pixels in [0,1]. Stochastic-defense/EOT support,
terminal prediction and uncertainty calibration are not implemented. Tree search
does not query the classifier, but its overhead counts toward attack time.
Matching gradient iteration caps does not match total computation; full standard
AutoAttack has a larger budget. Failure to find an attack does not certify robustness.

## Development

```bash
python -m pytest -q
ruff check .
ruff format --check .
```

CI runs these checks without downloading benchmark datasets.

## References

- [MuZero paper](https://arxiv.org/abs/1911.08265)
- [Official AutoAttack](https://github.com/fra31/auto-attack) (pinned commit in optional dependencies)
- [RobustBench](https://github.com/RobustBench/robustbench)
