# Frozen held-out CIFAR-10 evaluation

Execution status: core comparisons 2/2 models complete; standard AutoAttack 2/2 complete.

128 fresh sampled inputs per model, excluding all 40 prior development inputs.
The same frozen controller was trained only on Wong2020Fast. No retraining
or parameter selection used held-out results. Linf epsilon 8/255.
Official standard AutoAttack runs on the first 32 preselected sampled inputs.
Controller and APGD run on all 128. Compare equal subsets, not raw totals.

| Model | Method | Pool | Success / clean-correct | Observed adversarial accuracy | Forwards | Backwards | Seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| Sehwag2021Proxy_R18 | fixed | 128 | 35/109 | 57.81% | 5124 | 2498 | 162.02 |
| Sehwag2021Proxy_R18 | random | 128 | 34/109 | 58.59% | 5338 | 2605 | 146.16 |
| Sehwag2021Proxy_R18 | reactive | 128 | 35/109 | 57.81% | 5102 | 2487 | 150.83 |
| Sehwag2021Proxy_R18 | planning | 128 | 36/109 | 57.03% | 5052 | 2462 | 164.21 |
| Sehwag2021Proxy_R18 | apgd_ce | 128 | 34/109 | 58.59% | 3725 | 3597 | 100.68 |
| Sehwag2021Proxy_R18 | apgd_dlr | 128 | 33/109 | 59.38% | 3725 | 3597 | 91.29 |
| Sehwag2021Proxy_R18 | autoattack | 32 | 10/27 | 53.12% | 134726 | 33480 | 1988.00 |
| Rade2021Helper_R18_extra | fixed | 128 | 40/117 | 60.16% | 5386 | 2629 | 192.88 |
| Rade2021Helper_R18_extra | random | 128 | 38/117 | 61.72% | 5592 | 2732 | 221.42 |
| Rade2021Helper_R18_extra | reactive | 128 | 40/117 | 60.16% | 5288 | 2580 | 189.81 |
| Rade2021Helper_R18_extra | planning | 128 | 40/117 | 60.16% | 5296 | 2584 | 218.45 |
| Rade2021Helper_R18_extra | apgd_ce | 128 | 35/117 | 64.06% | 3989 | 3861 | 140.98 |
| Rade2021Helper_R18_extra | apgd_dlr | 128 | 40/117 | 60.16% | 3989 | 3861 | 101.10 |
| Rade2021Helper_R18_extra | autoattack | 32 | 11/31 | 62.50% | 158501 | 39412 | 2493.03 |

## Same-subset ensemble comparison

All methods in this table use the same predetermined 32 inputs per model.
Counts exclude initially incorrect predictions from the success denominator.

| Model | Fixed | Reactive | Planning | Standard AutoAttack | Planning only vs AA | AA only |
|---|---:|---:|---:|---:|---:|---:|
| Sehwag2021Proxy_R18 | 10/27 | 10/27 | 10/27 | 10/27 | 0 | 0 |
| Rade2021Helper_R18_extra | 10/31 | 11/31 | 11/31 | 11/31 | 0 | 0 |

## Paired comparisons

Differences are attack-success percentage points among clean-correct inputs.
Positive differences favor planning. Confidence intervals use 10,000 paired
bootstrap replicates; they do not account for controller-training randomness.

| Model | Comparator | Planning only | Comparator only | Difference pp | 95% paired CI pp |
|---|---|---:|---:|---:|---|
| Sehwag2021Proxy_R18 | planning_vs_fixed | 1 | 0 | 0.92 | [0.00, 2.75] |
| Sehwag2021Proxy_R18 | planning_vs_random | 2 | 0 | 1.83 | [0.00, 4.59] |
| Sehwag2021Proxy_R18 | planning_vs_reactive | 1 | 0 | 0.92 | [0.00, 2.75] |
| Sehwag2021Proxy_R18 | planning_vs_apgd_union | 1 | 0 | 0.92 | [0.00, 2.75] |
| Sehwag2021Proxy_R18 | planning_vs_autoattack_subset | 0 | 0 | 0.00 | [0.00, 0.00] |
| Rade2021Helper_R18_extra | planning_vs_fixed | 1 | 1 | 0.00 | [-2.56, 2.56] |
| Rade2021Helper_R18_extra | planning_vs_random | 2 | 0 | 1.71 | [0.00, 4.27] |
| Rade2021Helper_R18_extra | planning_vs_reactive | 0 | 0 | 0.00 | [0.00, 0.00] |
| Rade2021Helper_R18_extra | planning_vs_apgd_union | 0 | 0 | 0.00 | [0.00, 0.00] |
| Rade2021Helper_R18_extra | planning_vs_autoattack_subset | 0 | 0 | 0.00 | [0.00, 0.00] |

## Interpretation

The core evaluation shows one extra planning success over the APGD union
on Sehwag and identical success sets on Rade. Paired intervals and exact
tests do not establish a reliable improvement. Tree search also does not
consistently improve on the fixed schedule or reactive controller.
The ensemble comparison applies only to its preselected 32-input subsets.
These results do not establish state-of-the-art superiority or equivalence.

## Limits and cost interpretation

Controller/APGD iteration caps match at 32; measured total work differs.
APGD references use batches of 16; controller episodes execute individually.
Forward/backward counts are example-level model traversals, not batches.
Independent candidate auditing and model loading are excluded from timings.
Controller timers also exclude the initial clean prediction, while its
forward is counted. Reference timers include their internal clean checks.
Controller/tree-search inference is included in controller attack timings.
The APGD union is the union of two attacks and pays both attacks costs.
Full standard AutoAttack uses a larger budget than the controller.

Models share the same sampled inputs: do not pool them as 256 independent
observations. One controller training seed and two model checkpoints do not
establish superiority over all current white-box attacks or all defenses.
A zero-width bootstrap interval when outcomes coincide is not proof of
population equivalence. Failure to find an attack does not certify robustness.

Controller training cost from the development pilot: 25.702 seconds,
882 classifier forwards and 429 backwards; no retraining in this evaluation.

Frozen manifest: manifest.json. Per-model independent audits: audit.json.
Model/controller provenance and exact candidate arrays are saved beside them.
