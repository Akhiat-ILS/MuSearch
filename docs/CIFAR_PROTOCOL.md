# CIFAR-10 development pilot (frozen before examining outcomes)

Classifier: RobustBench Wong2020Fast, published robust PreActResNet18 checkpoint.
Raw input coordinates, Linf epsilon 8/255, classifier in evaluation mode, frozen
parameters. CPU with two torch threads. This checkpoint is an initial development
model, not a representative sample of current top-performing defenses.

Seed 42. Use loader indices 0..23 for controller training, indices 24..39 for
evaluation. No success or hardness filtering. Initially wrong predictions are
included in sample count and excluded from the attack-success denominator.
Two training rounds, 30 optimizer epochs per round, replay all previous trajectories.
No synthetic controller initialization. No retuning based on evaluation results.

Methods: fixed CE/margin alternation, random actions, reactive policy, latent
planning (24 simulations, maximum depth four). Each real action executes at most
four gradient steps, at most eight actions; stop after verified success. Official
APGD-CE and APGD-DLR: one restart, 32 iterations, same epsilon and evaluation pool.
Iteration caps match; forward counts, batch execution and timings do not match.
Report these differences rather than making equal-compute claims.

Training and evaluation use the same classifier with disjoint inputs. This tests
within-model generalization only. Final scientific evaluation must reserve models,
larger datasets and multiple training seeds and include standard AutoAttack.

Outputs: real training controller and provenance, per-method candidate arrays,
per-image attack statistics and original labels/inputs. Validate threat-set bounds
and actual classifier predictions. Attack failures are NOT_FOUND.

Reproduction (PowerShell, from this folder):

```
$env:PYTHONPATH = Join-Path (Get-Location) '.deps'
..\.venv\Scripts\python.exe benchmark.py --model Wong2020Fast --train-episodes 24 --samples 16 --steps 4 --blocks 8 --apgd --output runs/cifar10_pilot
```

Dependencies live in .deps, with torch/numpy reused read-only from the pre-existing
Python environment. Original project source and environment are not modified.

Additional reference evaluation: run `run_autoattack.py` on the identical saved
evaluation pool after the pilot. Official AutoAttack pinned to commit
a39220048b3c9f2cca9a4d3a54604793c68eca7e, standard configuration, same epsilon.
This is a larger-budget strength reference, not an equal-computation baseline.
Run `audit_benchmark.py` afterward, then `summarize_benchmark.py` to generate
the measured report. Installed dependency versions: dependencies-installed.txt.
