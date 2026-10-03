# Frozen cross-model evaluation v1

Controller: runs/cifar10_pilot/controller.pt, trained only on Wong2020Fast and
CIFAR-10 test indices 0..23. Prior development evaluation used indices 24..39.
Freeze its SHA256 in runs/heldout_v1/manifest.json before any held-out attack.
No retraining, controller selection or attack hyperparameter tuning is allowed.

Models: RobustBench Sehwag2021Proxy_R18 and Rade2021Helper_R18_extra. Both differ
from the controller-training checkpoint. These are established robust checkpoints,
not a sample of all latest top defenses. Holdout refers to controller training.

Inputs: 128 sampled uniformly without replacement from CIFAR-10 test indices
40..9999 using NumPy default_rng(1042). Same pool on both models. No filtering
by attack results or clean correctness. Exact ordered indices saved in manifest.
Full standard AutoAttack uses the first 32 sampled positions, chosen before
examining model predictions or attack results. Do not extrapolate subset findings
to all 128 inputs. Dataset inputs are raw pixels in [0,1]. Linf epsilon 8/255.

Four controller modes: fixed CE/margin alternation, random, reactive and planning.
Reuse four gradient steps per block, eight blocks, 24 latent search simulations,
maximum depth four; early stop on verified success. Evaluation seed 42, with
per-input seed 42+dataset_index. One frozen controller-training seed only.

Official APGD-CE and APGD-DLR: one restart and 32 iterations on all 128 inputs.
Standard AutoAttack: unchanged official configuration on the predetermined
32-input subset. APGD iteration caps match controller caps; total computation
need not match. Count example-level forwards/backwards for references, including
internal clean checks; controller counts include its own initial clean check.
Independent auditing is outside attack costs. Timings include planner overhead;
references use batches of 16, controller inputs execute individually.
Controller episode timers start after the initial clean prediction; that clean
prediction is counted in forwards but its time is excluded. Official reference
timers include their own internal clean checks. Report this asymmetry explicitly.

CPU, two torch threads. Run models sequentially to avoid concurrent benchmark
timing interference. Save method checkpoints and per-input action traces.
Independently re-evaluate each candidate against original model predictions and
Linf/pixel bounds. Preserve model checkpoint and controller hashes.

Report per-model success rates among clean-correct inputs, observed adversarial
accuracy over all sampled inputs, paired wins/losses versus the APGD union and
AutoAttack subset, and paired bootstrap confidence intervals. The two models
share inputs: do not treat their combined 256 model-input pairs as independent.
This is an exploratory, single-training-seed evaluation, not a full leaderboard
assessment. Failures to find adversarial examples never certify robustness.

Reproduction from this folder (PowerShell):

```
$env:PYTHONPATH = Join-Path (Get-Location) '.deps'
..\.venv\Scripts\python.exe -u heldout.py --stage core
..\.venv\Scripts\python.exe -u heldout.py --stage autoattack
..\.venv\Scripts\python.exe -u heldout.py --stage audit
```

All new code, packages, datasets, checkpoints and outputs stay in this project.
