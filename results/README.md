# Result files

Result files behind the paper's tables, figures and quoted numbers (see also the main [README](../README.md)).

| File | Content | Produced by |
|---|---|---|
| `final_results_54.json` | Accuracy of each of the 54 trained configurations on the three evaluation sets. Keys are `<scale>_<shape>_<training measure>_<pe>` (e.g. `large_wide_mixed_ape`); values are `{"bfs": ..., "dfs": ..., "mixed": ...}`, the fraction of the 2,000 problems of `val_bfs.json`, `val_dfs.json` and `val_mixed.json` that the model solves (greedy decoding, at most 512 new tokens, `metric_fn` score > 0). | `src/eval_flexible.py` (one JSON per model, same keys); this file is the union of the 54 per-model outputs. |
| `w1_robust.json` | For the pairs BFS-BFS and DFS-DFS (two disjoint samples of one measure), BFS-MIXED, BFS-DFS and DFS-MIXED: `[mean, s.d.]` over 10 subsamples (seeds 0-9, N = 2,000 per side, pool of 20,000 per training measure) of `exact_W1_zscored` (the W1 reported in the paper), `exact_W1_maxnormalized`, `sinkhorn_cost_reg0.1` (entropic transport cost, lambda = 0.1, cost divided by its maximum) and `sinkhorn_divergence_debiased` (the Sinkhorn cross-check of App. G). | `src/w1_robust.py` |
| `oracle_analysis.json` | For the three evaluation sets and 20,000-trajectory samples of each training set: success rate of the reference trajectories themselves under `metric_fn` (uncapped, within the prompt plus 512 new tokens, and within the 1,024-token training context), token-length percentiles, share of trajectories longer than 1,024 tokens, distribution of the maximum depth chi_1 and mean backtracking rate chi_6. | `src/oracle_analysis.py` |

Accuracies come from a single training run per configuration (seed 42).
