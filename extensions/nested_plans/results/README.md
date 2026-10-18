# Results: splitting deep nested plans into separate model calls (Experiment 7, App. M.4.1)

120 Transformers were trained from scratch on nested plans: two position encodings (RoPE, GPT-NeoX; APE, GPT-Neo
with learned absolute position embeddings and global attention in every layer) x five models (L2, L4, L8, L12: 2, 4,
8 or 12 layers of width 256 with 4 heads; W624: 2 layers of width 624 with 12 heads, about as many non-embedding
parameters as L12) x training depths d_train in {2, 4, 6, 8} x seeds {0, 1, 2}. All values are written by
`code/export_tables.py`, copied unchanged from the output of `code/analyze.py` (`analysis/summary.json`), of
`code/gatecheck.py` (`runs/gatecheck.json`) and of `code/train.py` (`results.json` of each run).

## Terms

- **Plan**: 48 subgoals of 8 types whose maximum nesting depth is exactly d, written as a word of push and pop events
  (194 tokens for every d). At every pop the model predicts the type of the matching push (teacher forcing, most
  likely token). A plan is **solved** if all 48 predictions are right. Training plans are drawn fresh at every step
  (256 per step) with d uniform in 1..d_train.
- **Test and calibration plans**: 500 test plans and 200 other calibration plans at every depth d = 1..d_test, with
  d_test = 4 d_train + 8.
- **Single-call success** at depth d: fraction of the plans of depth d that one call solves.
- **d_ext**: the single-call success per depth is fitted by a function that may not increase with depth (pool
  adjacent violators, equal weights); d_ext is the largest depth at which the fit is at least 0.5 (0 if there is
  none). `d_ext_calibration` uses the calibration plans and sets the split; `d_ext_test` uses the test plans and is
  reported only.
- **Split**: a test plan deeper than d_ext_calibration is split into calls of nesting depth at most
  D = min(d_ext_calibration, d_train) (`code/handoff.py`). In a call, a subgoal at depth D (counted inside the call)
  that has children is written empty, as `( g ) g`, and its prediction is scored; its subtree becomes a new call in
  a fresh context and is split again in the same way. Plans no deeper than d_ext_calibration are solved in one call.
  A split plan is solved only if every prediction of every call is right. The split reads only the push/pop events and
  the pushed types, never the scored tokens. If D < 2 no split is possible and every plan is solved in one call.
- **Re-training**: after the 120 runs of 30,000 steps, `code/gatecheck.py` checks every run for single-call success of
  at least 0.9 at every depth 1..d_train on the test plans. Every configuration (position encoding, d_train, model)
  with at least one run below 0.9 was trained again, all three seeds, with 60,000 steps; these runs replace the
  30,000-step runs in `runs.csv`, `per_depth.csv` and `runs/`. This selected five APE configurations, W624 at
  d_train = 2, 4, 6, 8 and L2 at d_train = 4 (13 runs below 0.9, see `selection_30000_steps.csv`). After re-training,
  the six runs of W624 APE at d_train = 2 and 4 are still below 0.9 at some depth up to d_train.

The numbers quoted in App. M.4.1 come from `runs.csv`. In the 114 rows with `single_call_ge_0.9_up_to_d_train` True
(38 configurations), `split_success_beyond_d_ext_mean` is at least 0.9871, `split_success_beyond_d_ext_min` at least
0.912 and `single_call_success_beyond_d_ext` at most 0.0402 (quoted as 0.04; run ape_dt4_L2_s0), and
`d_ext_calibration` is 2, 4-5, 6-9 and 9-12 for d_train = 2, 4, 6, 8. Of the other six runs, five have
d_ext_calibration = 1 (no split possible) and ape_dt2_W624_s2 has d_ext_calibration = 2, single-call success 0.0 and
split success 0.9982857 beyond it. Figure 6a shows `d_ext_calibration` of every run; Figure 6b shows the means over
the three seeds of `single_call_success` and `split_success` in `per_depth.csv` for the L12 RoPE runs.

## runs.csv (one row per analysed run, 120 rows)

| column | meaning |
|---|---|
| run | run name `<pe>_dt<d_train>_<model>_s<seed>` |
| position_encoding | `RoPE` or `APE` |
| d_train | deepest training plans |
| model | `L2`, `L4`, `L8`, `L12` or `W624` (see above) |
| layers, width | number of layers and hidden size |
| seed | seed of the initialisation and of the training plans |
| non_embedding_parameters | number of parameters without the token and position embeddings and the output layer |
| training_steps | 30000, or 60000 for the re-trained configurations |
| d_test | deepest test depth, 4 d_train + 8 |
| d_ext_calibration | d_ext on the calibration plans; sets the split |
| d_ext_test | d_ext on the test plans (not used by the split) |
| call_depth | D = min(d_ext_calibration, d_train), the largest nesting depth of a call after the split |
| split_possible | True if call_depth is at least 2 |
| single_call_success_beyond_d_ext | mean over the depths d_ext_calibration + 1 .. d_test of the single-call success on the test plans |
| split_success_beyond_d_ext_mean | mean over the same depths of the fraction of test plans solved with the split |
| split_success_beyond_d_ext_min | smallest value of that fraction over the same depths |
| calls_per_plan_beyond_d_ext | mean number of calls per test plan over the plans deeper than d_ext_calibration (1.0 if no split is possible) |
| single_call_ge_0.9_up_to_d_train | True if the single-call success on the test plans is at least 0.9 at every depth 1..d_train |

If `split_possible` is False, the split columns equal the single-call values (every plan is solved in one call).

## per_depth.csv (one row per run and test depth, 3,360 rows)

| column | meaning |
|---|---|
| run | as in runs.csv |
| depth | nesting depth d of the plans, 1..d_test |
| single_call_success | fraction of the 500 test plans of depth d solved in one call |
| split_success | fraction of the same plans solved with the split of runs.csv (equal to single_call_success at depths up to d_ext_calibration) |
| single_call_success_calibration | fraction of the 200 calibration plans of depth d solved in one call (the values from which d_ext_calibration is computed) |
| calls_per_plan | mean number of calls per test plan of depth d with the split (rounded to 2 decimals by analyze.py) |

## selection_30000_steps.csv (one row per run of 30,000 steps, 120 rows)

| column | meaning |
|---|---|
| run, position_encoding, d_train, model | as in runs.csv |
| training_steps | 30000 |
| single_call_success_d1 .. single_call_success_d8 | single-call success on the test plans of depth 1..d_train of the 30,000-step run (empty beyond d_train) |
| single_call_ge_0.9_up_to_d_train | True if all these values are at least 0.9 |
| cell_retrained_60000_steps | True if the configuration of this run was trained again with 60,000 steps |

## runs/<run>.json (one file per analysed run)

The `results.json` that `code/train.py` wrote for the run, unchanged except `config.root`, which gives the folder of
the run (`runs` or `runs_ext`, for the re-trained configurations) instead of the absolute path.

| key | meaning |
|---|---|
| name | run name |
| config | arguments of train.py: `pe`, `L` (layers), `d` (width), `h` (heads), `seed`, `steps`, `bs` (plans per step), `lr`, `warmup` (steps of linear warm-up before the cosine decay to 0.1 lr), `wd` (weight decay), `N` (subgoals per plan), `k` (subgoal types), `dtrain`, `dtest`, `n_test`, `n_cal`, `n_val` (plans per depth of the test, calibration and validation sets), `val_every`, `ckpt_every`, `workers` (data-loader workers), `time_budget_s`, `eval_bs` (evaluation batch size), `grid_idx` (row of `code/grid.tsv`), `root` |
| params_total, params_nonembed | number of parameters, with and without embeddings and output layer |
| gpu | GPU of the run |
| train_seconds, eval_seconds | training and evaluation time |
| n_resumes | number of restarts from a checkpoint (0 for every run) |
| split_ds | the budgets D with which every test plan was also split (2..d_train) |
| depths | the test depths 1..d_test |
| cal_single, test_single | one call per plan on the calibration and test plans. `per_depth[d]`: `n` (plans), `seq` (fraction solved), `pop_acc` (fraction of the pop predictions that are right), `nontriv_pop_acc` and `nontriv_pop_n` (the same over the pops of subgoals that have children, so that the matching push is not the previous event, and their number), `recency_seq` and `recency_pop_acc` (fraction solved and fraction of right pops for the rule that predicts the type of the most recent push; test plans only, null for calibration). `per_pop_depth[j]`: `n`, `acc`, `nontriv_n`, `nontriv_acc`, the same pop accuracies by the depth j of the popped subgoal |
| test_split<D> | every test plan split with budget D (a plan no deeper than D gives one call, identical to the plan). `per_depth[d]`: `n`, `seq_all` (fraction of plans with every prediction of every call right), `seq_own` (fraction with every prediction right except those of the empty subgoals `( g ) g`), `calls` (mean calls per plan), `tokens` (mean number of tokens over all calls of a plan); `call_success` (fraction of all calls with every prediction right), `n_calls_total`, `max_call_depth`, `call_success_by_call_depth` (call_success by the nesting depth of the call) |
| oracle_composition_ok | True if the true tokens, put into the calls and composed, solve every plan |

`split_success` in per_depth.csv is `test_split<D>.per_depth[d].seq_all` with D = call_depth at depths beyond
d_ext_calibration, and `test_single.per_depth[d].seq` elsewhere.
