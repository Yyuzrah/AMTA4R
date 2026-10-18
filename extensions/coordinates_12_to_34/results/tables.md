# From 12 to 34 coordinates on Countdown: result tables

Rounded from the CSV files of this folder (written by `scripts/export_tables.py`). f1 = Small Deep MIXED RoPE; f2 = Large Deep BFS APE.

## Out-of-fold ROC-AUC and McFadden pseudo-R² of the App. J classifier on Phi_d (10 repetitions)

Mean ± s.d. over repetitions; Delta = value(Phi_d) - value(Phi_12), paired by repetition.

| model | n | d | ROC-AUC | Delta AUC | Delta AUC, min to max over repetitions | McFadden | Delta McFadden |
|---|---|---|---|---|---|---|---|
| f1 | 50000 | 12 | 0.9269 ± 0.0002 | - | - | 0.5073 ± 0.0004 | - |
| f1 | 50000 | 16 | 0.9271 ± 0.0001 | 0.0002 | 0.0001 to 0.0004 | 0.5078 ± 0.0005 | 0.0005 |
| f1 | 50000 | 24 | 0.9271 ± 0.0002 | 0.0002 | 0.0001 to 0.0005 | 0.5076 ± 0.0006 | 0.0002 |
| f1 | 50000 | 34 | 0.9289 ± 0.0003 | 0.0020 | 0.0016 to 0.0025 | 0.5133 ± 0.0008 | 0.0059 |
| f2 | 3000 | 12 | 0.9290 ± 0.0013 | - | - | 0.4714 ± 0.0076 | - |
| f2 | 3000 | 16 | 0.9286 ± 0.0009 | -0.0004 | -0.0016 to 0.0005 | 0.4660 ± 0.0051 | -0.0054 |
| f2 | 3000 | 24 | 0.9283 ± 0.0014 | -0.0006 | -0.0019 to 0.0019 | 0.4648 ± 0.0087 | -0.0066 |
| f2 | 3000 | 34 | 0.9266 ± 0.0016 | -0.0024 | -0.0044 to 0.0007 | 0.4558 ± 0.0094 | -0.0155 |

## Exact W1 on the pools of the three training sets (App. G), N = 2,000 per side (10 seeds)

Mean ± s.d. over seeds; within = two disjoint samples of one measure.

| pair | d = 12 | d = 16 | d = 24 | d = 34 | ratio d34/d12 |
|---|---|---|---|---|---|
| BFS-BFS (within) | 0.189 ± 0.065 | 0.233 ± 0.066 | 0.348 ± 0.097 | 0.443 ± 0.112 | 2.35 |
| DFS-DFS (within) | 0.283 ± 0.049 | 0.357 ± 0.055 | 0.399 ± 0.061 | 0.460 ± 0.071 | 1.63 |
| MIXED-MIXED (within) | 0.309 ± 0.035 | 0.390 ± 0.039 | 0.489 ± 0.049 | 0.590 ± 0.059 | 1.91 |
| BFS-MIXED | 1.378 ± 0.053 | 1.815 ± 0.069 | 2.258 ± 0.086 | 2.778 ± 0.110 | 2.02 |
| BFS-DFS | 2.972 ± 0.046 | 3.770 ± 0.050 | 4.695 ± 0.052 | 5.652 ± 0.069 | 1.90 |
| DFS-MIXED | 1.757 ± 0.057 | 2.160 ± 0.065 | 2.740 ± 0.071 | 3.204 ± 0.085 | 1.82 |

## Exact W1 on the reference trajectories of App. J, N = 2,000 per side (10 seeds)

Mean ± s.d. over seeds; within = two disjoint samples of one measure.

| pair | d = 12 | d = 16 | d = 24 | d = 34 | ratio d34/d12 |
|---|---|---|---|---|---|
| BFS-BFS (within) | 0.204 ± 0.041 | 0.248 ± 0.039 | 0.372 ± 0.059 | 0.470 ± 0.073 | 2.31 |
| DFS-DFS (within) | 0.291 ± 0.037 | 0.369 ± 0.044 | 0.410 ± 0.049 | 0.474 ± 0.060 | 1.63 |
| MIXED-MIXED (within) | 0.333 ± 0.038 | 0.414 ± 0.037 | 0.518 ± 0.040 | 0.620 ± 0.050 | 1.86 |
| BFS-MIXED | 1.433 ± 0.027 | 1.880 ± 0.025 | 2.311 ± 0.024 | 2.856 ± 0.035 | 1.99 |
| BFS-DFS | 2.966 ± 0.043 | 3.761 ± 0.045 | 4.680 ± 0.049 | 5.642 ± 0.062 | 1.90 |
| DFS-MIXED | 1.717 ± 0.069 | 2.103 ± 0.080 | 2.692 ± 0.083 | 3.139 ± 0.097 | 1.83 |

## Exact W1 on the evaluation sets, N = 1,000 per side (10 seeds)

Mean ± s.d. over seeds; within = two disjoint samples of one measure.

| pair | d = 12 | d = 16 | d = 24 | d = 34 | ratio d34/d12 |
|---|---|---|---|---|---|
| BFS-BFS (within) | 0.240 ± 0.024 | 0.295 ± 0.029 | 0.429 ± 0.035 | 0.538 ± 0.045 | 2.24 |
| DFS-DFS (within) | 0.408 ± 0.049 | 0.505 ± 0.056 | 0.564 ± 0.067 | 0.651 ± 0.075 | 1.60 |
| MIXED-MIXED (within) | 0.396 ± 0.053 | 0.494 ± 0.059 | 0.620 ± 0.072 | 0.750 ± 0.084 | 1.90 |
| BFS-MIXED | 1.565 ± 0.052 | 1.988 ± 0.065 | 2.498 ± 0.083 | 3.009 ± 0.091 | 1.92 |
| BFS-DFS | 3.049 ± 0.053 | 3.844 ± 0.055 | 4.781 ± 0.041 | 5.749 ± 0.045 | 1.89 |
| DFS-MIXED | 1.559 ± 0.060 | 1.947 ± 0.064 | 2.412 ± 0.066 | 2.892 ± 0.071 | 1.86 |

All 2,000 vs all 2,000 trajectories (one value): BFS-MIXED: d = 12 1.519, d = 16 1.928, d = 24 2.406, d = 34 2.897; BFS-DFS: d = 12 3.028, d = 16 3.821, d = 24 4.749, d = 34 5.715; DFS-MIXED: d = 12 1.515, d = 16 1.898, d = 24 2.352, d = 34 2.827.

## W1 ratios

| block | d | mean within / mean between | BFS-MIXED / BFS-BFS | BFS-DFS / BFS-BFS | DFS-MIXED / DFS-DFS |
|---|---|---|---|---|---|
| pool | 12 | 0.128 | 7.30 | 15.75 | 6.22 |
| pool | 16 | 0.126 | 7.80 | 16.21 | 6.06 |
| pool | 24 | 0.128 | 6.49 | 13.49 | 6.87 |
| pool | 34 | 0.128 | 6.27 | 12.76 | 6.97 |
| fresh | 12 | 0.135 | 7.02 | 14.53 | 5.90 |
| fresh | 16 | 0.133 | 7.57 | 15.14 | 5.70 |
| fresh | 24 | 0.134 | 6.22 | 12.59 | 6.57 |
| fresh | 34 | 0.134 | 6.07 | 11.99 | 6.62 |
| eval | 12 | 0.169 | 6.52 | 12.71 | 3.82 |
| eval | 16 | 0.166 | 6.74 | 13.03 | 3.85 |
| eval | 24 | 0.166 | 5.82 | 11.14 | 4.27 |
| eval | 34 | 0.166 | 5.60 | 10.69 | 4.44 |

## Within-measure W1 on the pools against N

Mean over seeds; alpha = -(least-squares slope of log mean W1 on log N).

| pair | d | N = 250 | N = 500 | N = 1,000 | N = 2,000 | N = 4,000 | alpha |
|---|---|---|---|---|---|---|---|
| BFS-BFS | 12 | 0.523 | 0.382 | 0.247 | 0.189 | 0.143 | 0.475 |
| BFS-BFS | 16 | 0.641 | 0.462 | 0.303 | 0.233 | 0.171 | 0.479 |
| BFS-BFS | 24 | 0.900 | 0.649 | 0.445 | 0.348 | 0.262 | 0.446 |
| BFS-BFS | 34 | 1.116 | 0.802 | 0.568 | 0.443 | 0.338 | 0.431 |
| DFS-DFS | 12 | 0.618 | 0.543 | 0.384 | 0.283 | 0.192 | 0.431 |
| DFS-DFS | 16 | 0.759 | 0.664 | 0.480 | 0.357 | 0.252 | 0.408 |
| DFS-DFS | 24 | 0.844 | 0.748 | 0.542 | 0.399 | 0.281 | 0.408 |
| DFS-DFS | 34 | 0.966 | 0.858 | 0.620 | 0.460 | 0.326 | 0.403 |
| MIXED-MIXED | 12 | 0.727 | 0.575 | 0.389 | 0.309 | 0.224 | 0.429 |
| MIXED-MIXED | 16 | 0.900 | 0.719 | 0.493 | 0.390 | 0.287 | 0.417 |
| MIXED-MIXED | 24 | 1.114 | 0.897 | 0.621 | 0.489 | 0.363 | 0.411 |
| MIXED-MIXED | 34 | 1.326 | 1.077 | 0.742 | 0.590 | 0.441 | 0.404 |
