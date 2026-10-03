# CIFAR-10 development pilot results

Model: RobustBench Wong2020Fast. Raw Linf epsilon 8/255. Seed 42.
24 controller-training inputs; 16 disjoint evaluation inputs.
Training and evaluation use the same classifier. No held-out-model claim.

| Method | Successes / clean-correct | Forwards | Backwards | Attack seconds |
|---|---:|---:|---:|---:|
| fixed | 9/13 | 312 | 148 | 8.217 |
| random | 9/13 | 316 | 150 | 8.353 |
| reactive | 9/13 | 310 | 147 | 8.273 |
| planning | 9/13 | 310 | 147 | 8.668 |
| apgd_ce | 9/13 | not instrumented | not instrumented | 8.472 |
| apgd_dlr | 9/13 | not instrumented | not instrumented | 8.282 |
| autoattack | 9/13 | not instrumented | not instrumented | 515.830 |

All saved candidates passed independent constraint and prediction auditing.
Controller methods use at most 32 gradient steps per input; standalone APGD
uses one restart and 32 iterations. Full AutoAttack uses its official standard
configuration, with a larger budget. Its runtime is not an equal-budget comparison.
Baselines use batched execution; the controller currently executes inputs individually.
Controller timings include tree search; independent audit and training are excluded.

Controller training: 25.702 seconds, 882 forwards, 429 backwards.

## Paired outcomes

```json
{
  "paired_comparison": {
    "apgd_union_successes": 9,
    "planning_only": 0,
    "apgd_union_only": 0
  },
  "autoattack_comparison": {
    "planning_only": 0,
    "autoattack_only": 0
  }
}
```

## Interpretation

This small, single-model, single-seed development pilot cannot establish
state-of-the-art performance. Compare paired outcomes and total compute,
not attack-success counts alone. Failures to find attacks are NOT_FOUND.

Next: improve recurrent/terminal modeling on development data, then freeze
selection and run multiple seeds on larger pools and held-out robust models.
Do not reuse these evaluation inputs as an untouched test after tuning.
