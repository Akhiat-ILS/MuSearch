# Initial synthetic pilot

Executed 3 October 2026, seed 11. Controller trained in two self-play rounds
using 80 synthetic inputs, then evaluated on 64 separate inputs, 62 of which
were correctly classified. Threat: Linf epsilon 0.1, at most eight blocks of
four gradient steps. This is a pipeline test, not a recognized robustness benchmark.

| Method | Successes / clean-correct | Forwards | Backwards | Attack seconds |
|---|---:|---:|---:|---:|
| Fixed CE/margin alternation | 31/62 | 2202 | 1069 | 0.501 |
| Random action selection | 31/62 | 2212 | 1074 | 0.525 |
| Reactive learned policy | 29/62 | 2426 | 1181 | 0.690 |
| Learned latent planning | 29/62 | 2426 | 1181 | 2.755 |

All 256 saved candidates passed independent model prediction and threat-set
checks. Timings include controller inference and planning, exclude independent
validation and offline training. Training costs are in runs/pilot/results.json.
This single-seed result does not show an advantage for planning. The synthetic
boundary is simple and provides limited need for sequential decisions.

Seven tests passed, covering real success, projection, classifier parameter
preservation, no classifier calls during imagined search and controller training.

## Current limitations

- One-step recurrent training, not the full multi-step MuZero training algorithm.
- Four actions, small observation vector; no gradient or class identity features.
- No predicted terminal head; deeper search can imagine rewards after success.
- No uncertainty calibration for learned dynamics, which may exploit model error.
- CPU, one input at a time; no image benchmark performance demonstrated.
- A subsequent CIFAR-10 development pilot has been executed; its separate report
  is runs/cifar10_pilot/RESULTS.md. Dependencies are isolated in .deps.
- Synthetic controller weights are pipeline artifacts, not trained CIFAR attacks.
- Official AutoAttack option uses its own standard budget; equal-budget claims
  require a separate comparison, not a direct comparison of those runs.

## Research decision

Do not claim state-of-the-art performance. Next, collect trajectories on robust
CIFAR development checkpoints, use richer state observations and recurrent
unrolling with terminal supervision, and compare on separate checkpoints against
official APGD and strong fixed schedules. Freeze model splits and all controller
selection criteria before final evaluation. A failure to find an attack is NOT_FOUND.

## Subsequent real-model pilot

The independent folder now includes isolated RobustBench and official AutoAttack
dependencies, a downloaded robust CIFAR-10 classifier, real-input controller
training, separate evaluation inputs and independent candidate auditing.
See CIFAR_PROTOCOL.md for the frozen development protocol and
runs/cifar10_pilot/RESULTS.md for measured outcomes. This remains a development
pilot, not evidence of state-of-the-art performance.
