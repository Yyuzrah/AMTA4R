# Results: from 12 to 34 coordinates on Countdown (Experiment 1, App. M.1.1)

The nested projections Phi_12 ⊂ Phi_16 ⊂ Phi_24 ⊂ Phi_34 take the first d coordinates of the list below. Two
quantities are measured for every d, on CPU only:

* **Prediction of success (App. J).** A `HistGradientBoostingClassifier` (scikit-learn 1.5.2, default settings)
  predicts from Phi_d of a newly generated reference trajectory whether a trained model solves the problem. Metrics
  are out of fold with stratified 5-fold cross-validation, averaged over the 5 folds. f1 = Small Deep MIXED RoPE
  (50,000 problems), f2 = Large Deep BFS APE (3,000 problems); the labels are those of `gbc/labels/`. There are 10
  repetitions: repetition s uses split seed s and classifier `random_state` s (s = 0 is the split of App. J), and
  the folds of a repetition are the same for every d.
* **Exact W1 (App. G).** `ot.emd2` with Euclidean cost and uniform weights between two samples of N z-scored Phi_d
  vectors, 10 seeds. Within pairs (BFS-BFS, DFS-DFS, MIXED-MIXED) compare two disjoint samples of one measure and give
  the baseline due to sampling; between pairs compare two measures. For a given seed, pair and N the same
  trajectories are used for every d. Coordinates are z-scored with the mean and s.d. over the three measures of a
  block (s.d. 0 replaced by 1). Blocks:
  * `pool`: the pools of App. G, 20,000 trajectories per training set (`random.seed(42)`, `random.sample` from
    `data/experiment_500k/train_{bfs,dfs,mixed}.json`), N = 2,000;
  * `fresh`: the reference trajectories of App. J (first 16,667/16,667/16,666 of `gbc/data/fresh_*.json`), N = 2,000;
  * `eval`: the evaluation sets (`data/experiment_500k/val_*.json`, 2,000 each), N = 1,000, plus one value with all
    2,000 against all 2,000 trajectories for the between pairs;
  * `nsweep`: the pools again, within pairs only, N = 250, 500, 1,000 and 4,000.

The numbers quoted in App. M.1.1 are `auc_mean` in `auc_summary.csv` (0.9266 to 0.9290), `delta_auc` and
`delta_mcfadden` at d = 34 in the same file (largest absolute values 0.0024 and 0.0155, both f2) and `ratio_to_d12` at d = 34 for block `pool` in `w1_summary.csv`
(within pairs 1.63 to 2.35, between pairs 1.82 to 2.02).

## Coordinates

Conventions (as in `scripts/ontology_countdown.py`): a node reference is a line that starts with `Generated Node #`
or `Moving to Node #` followed by a node id, a comma-separated path such as `0,1,2`; its depth is the number of
entries (3 here) and the parent of a node is its id without the last entry. A step is a pair of consecutive node
references. Tokens are those of the GPT-Neo tokenizer (`EleutherAI/gpt-neo-1.3B`, no special tokens); the token
position of a line is the index of the token that contains its first character. A ratio with a zero denominator, and
a mean or maximum over an empty set, is 0. All coordinates are computed on the reference trajectory.

Coordinates 1-12 are chi_1..chi_12 of App. A.1, computed by `src/distribution_metrics.py` (which counts every
occurrence of `Node #<id>` as a node reference):

| # | name | class | definition |
|---|---|---|---|
| 1 | chi1_max_depth | NEST | largest depth of a node reference |
| 2 | chi2_mean_depth | TOP | mean depth of the node references |
| 3 | chi3_std_depth | TOP | standard deviation of that depth |
| 4 | chi4_volume | TOP | number of node references |
| 5 | chi5_deepen | CTRL | fraction of steps that increase the depth |
| 6 | chi6_backtrack | CTRL | fraction of steps that decrease the depth |
| 7 | chi7_lateral | CTRL | fraction of steps that keep the depth |
| 8 | chi8_prunes | TERM | number of occurrences of `unequal: No Solution` |
| 9 | chi9_prune_rate | TERM | chi_8 divided by the number of occurrences of `Generated Node #` |
| 10 | chi10_goal | TERM | 1 if `Goal Reached` occurs, else 0 |
| 11 | chi11_mult | INTER | number of `*` and `/` characters |
| 12 | chi12_add | INTER | number of `+` and `-` characters |

Coordinates 13-34 are the 22 further coordinates, in the class order of `ontology_countdown.CLASSES` (TOP topology,
CTRL control flow, TERM recovery and termination, MEM memory, NEST nesting, INTER interaction, SURF surface).
Phi_16 adds 13-16, Phi_24 adds 13-24, Phi_34 adds all 22. The four marked with * are counted in tokens.

| # | name | class | definition |
|---|---|---|---|
| 13 | unique_nodes | TOP | number of distinct node ids among the node references |
| 14 | revisit_ratio | TOP | 1 - unique_nodes / number of node references |
| 15 | branch_mean | TOP | mean, over nodes that are the parent of a referenced node, of the number of distinct referenced children |
| 16 | branch_max | TOP | largest number of distinct referenced children of one node |
| 17 | pop_mean | CTRL | mean depth decrease over the steps that decrease the depth |
| 18 | cross_subtree_rate | CTRL | steps that keep the depth, start at depth 2 or more and change the parent, divided by the number of steps |
| 19 | backtrack_episodes | CTRL | number of maximal runs of consecutive steps that decrease the depth |
| 20 | fail_streak_max | TERM | largest number of lines containing `unequal: No Solution` without a `Generated Node` line between them |
| 21 | goal_pos_frac | TERM | token position of the first line containing `equal: Goal Reached` divided by the number of tokens; 1 if there is none |
| 22 | post_goal_tokens* | TERM | number of tokens from the first line containing `equal: Goal Reached` to the end; 0 if there is none |
| 23 | goal_count | TERM | number of lines containing `equal: Goal Reached` |
| 24 | return_dist_mean* | MEM | mean, over `Moving to Node` lines to a node referenced before (except right after a reference to the same node), of the tokens between the previous reference to that node and this line |
| 25 | return_dist_max* | MEM | largest of these token distances |
| 26 | live_ctx_mean | MEM | mean, over node references, of the number of nodes referenced both before and after it |
| 27 | pop_max | NEST | largest depth decrease of one step |
| 28 | live_ctx_max | NEST | largest number of nodes referenced both before and after one node reference |
| 29 | restate_ratio | NEST | number of `Current State` lines divided by the number of node references |
| 30 | n_actions | INTER | number of `Exploring Operation` lines |
| 31 | actions_per_state | INTER | n_actions divided by the number of `Current State` lines |
| 32 | feedback_ratio | INTER | chi_8 divided by n_actions |
| 33 | len_tokens* | SURF | number of tokens of the trajectory |
| 34 | n_lines | SURF | number of non-empty lines |

## Tables

All CSV files are written by `scripts/export_tables.py` from the files in `runs/`, at full precision; `tables.md`
shows the same tables rounded. "s.d. over repetitions" and "s.d. over seeds" use ddof 1, "s.d. over folds" ddof 0.

`auc_by_repetition.csv`, one row per model, repetition and d:
* `model`: f1 or f2; `model_name`: prefix of its label files in `gbc/labels/`; `n`: labelled problems;
* `repetition`: s (split seed and classifier seed); `d`: number of coordinates;
* `auc`, `mcfadden`, `brier`: mean over the 5 test folds of ROC-AUC, McFadden pseudo-R² (1 - log-likelihood of the
  classifier / log-likelihood of a constant prediction equal to the share solved in the training folds) and Brier
  score; `auc_fold_sd`, `mcfadden_fold_sd`, `brier_fold_sd`: their s.d. over the folds.

`auc_summary.csv`, one row per model and d, over the repetitions:
* `model`, `model_name`, `n` as above; `accuracy`: share of the n problems that the model solves; `n_repetitions`: 10;
* `auc_mean`, `auc_sd`, `mcfadden_mean`, `mcfadden_sd`: mean and s.d. over repetitions of `auc` and `mcfadden`;
* `delta_auc`: mean over repetitions of auc(Phi_d) - auc(Phi_12) on the same folds; `delta_auc_min`,
  `delta_auc_max`: smallest and largest of the 10 differences; `share_abs_delta_auc_below_0.01`: fraction of
  repetitions with an absolute difference below 0.01; `delta_mcfadden`: as `delta_auc` for McFadden. Empty for d = 12.

`w1_by_seed.csv`, one row per W1 value:
* `block`: pool, fresh, eval or nsweep; `pair`: the two measures; `within`: True for two disjoint samples of one
  measure, False for two measures;
* `sample`: `sub` for random samples of N per side drawn with `seed`, `full` for all 2,000 against all 2,000
  trajectories of two evaluation sets (seed 0 only); `N`: trajectories per side; `d`; `seed`;
* `w1`: exact W1; `ot_result_code`: POT result code, 1 = optimal (all values); `recomputed`: True if the value comes
  from `w1_rescue.py` (none).

`w1_summary.csv`, one row per block, pair, sample, N and d (blocks pool, fresh and eval; the N sweep is in
`w1_n_dependence.csv`):
* `block`, `pair`, `within`, `sample`, `N`, `d` as above;
* `mean`, `sd`: mean and s.d. over seeds of `w1`; `n_seeds`: number of values (1 for `sample` = full, whose `sd` is
  empty); `ratio_to_d12`: `mean` divided by the `mean` at d = 12 of the same block and pair.

`w1_ratios.csv`, one row per block and d:
* `mean_within_over_mean_between`: mean of the three within-pair means divided by the mean of the three between-pair
  means;
* `bfs_mixed_over_bfs_bfs`, `bfs_dfs_over_bfs_bfs`: BFS-MIXED and BFS-DFS mean divided by the BFS-BFS mean;
  `dfs_mixed_over_dfs_dfs`: DFS-MIXED mean divided by the DFS-DFS mean;
* `bfs_shifts_at_least_7x_within_bfs`: True if both BFS ratios are at least 7 (for d = 12 on the pools this is the
  "at least sevenfold" of Section 5).

`w1_n_dependence.csv`, within pairs on the pools, one row per pair, d and N:
* `pair`, `d`, `N`; `block`: nsweep, or pool for N = 2,000; `mean`, `sd`, `n_seeds` as in `w1_summary.csv`.

`w1_n_dependence_alpha.csv`, one row per within pair and d:
* `alpha`: minus the least-squares slope of log `mean` on log N over the five N; `n_points`: number of N used.

## Per-run files (`runs/`)

`runs/auc/<model>_s<repetition>.json` (written by `auc_task.py`): `model`, `model_name`, `seed` (the repetition),
`n`, `accuracy`, `label_coding` (the label is 1 when the model solves the problem); `dims[d]`: `folds` (per test
fold: `auc`, `mcfadden`, `brier`, `n_test`, `base_rate_train` = share solved in the training folds, `n_iter` =
boosting iterations) and `auc`, `mcfadden`, `brier` as [mean, s.d.] over the folds; `smoke`: false (true marks a
reduced-scale test run, `COORD_QUICK=1`); `seconds`, `time`: run time and end time. Repetition 0 also holds
`secondary`: `coding_loss[d]` (the label coded as 1 when the model fails), `grouped[d]` (folds grouped by problem,
target and sorted operands, with `StratifiedGroupKFold`, d = 12 and 34), `within_ops[k]` (only problems with k = 3 or
4 operands: `n`, `accuracy` and the metrics per d), `ungrouped_test_rows_with_problem_in_train_folds` (test rows whose
problem also occurs in the training folds of the ungrouped split) and `n_distinct_problems`.

`runs/w1/<block>_s<seed>.json` (written by `w1_task.py`): `block`, `seed`, `n_pool` (trajectories per measure in the
block), `zstats` (mean and s.d. of the 34 coordinates used for z-scoring), `entries` (one per pair, N, sample and d:
`pair`, `a`, `b`, `same` = within, `N`, `tag` = sample, `d`, `w1`, `code` = ot_result_code, `seconds`), `num_iter`
(the POT `numItermax`), `seconds`, `time`, `smoke`.

`runs/w1_rescue.json` (written by `w1_rescue.py`): `n_files` read, `n_rescued` values recomputed with
`numItermax` = 100,000,000 (0, since every value was optimal), `n_still_not_optimal`, `num_iter`, `entries`,
`seconds`, `time`.

The full recompute also writes the subsample indices (`runs/w1/*.npz`), the out-of-fold predictions
(`runs/auc/*.npz`), the per-trajectory coordinates (`data/features/*.npz`), the checks of `gate.py` and `analyze.py`
(`analysis/gate.json`, `analysis/verify.json`) and `analysis/summary.json`, which also holds a paired bootstrap of
the pooled out-of-fold difference in ROC-AUC for repetition 0; these files are not included.
