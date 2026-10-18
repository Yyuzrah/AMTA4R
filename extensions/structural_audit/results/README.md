# Results: structural shift between groups of trajectories (Experiment 2, App. M.2.1)

`scripts/w1.py` computes, for every pair of groups of one family, the exact W1 between the two groups (`ot.emd2`,
Euclidean cost, uniform weights) and the within-group baseline of each group (W1 between two disjoint samples of the
group), with 10 seeds and N = min(2000, n_a/2, n_b/2) units per sample. A unit is a trajectory, or in family C an
issue, which contributes one of its resolved runs per sample. The coordinates are z-scored with the mean and s.d. over
all groups of the family. A pair is separated if R, the between-group W1 divided by the larger of the two baselines,
is at least 3 and the coefficient of variation of the between-group W1 over the seeds is at most 0.10.
`scripts/report.py` summarizes the runs and draws `../fig_ratios.pdf` (Figure 4).

Families: A, the six ALFWorld task types; B, the three ETO environments; B2, human and GPT-4 demonstrations in
WebShop; C_H1, SWE-agent and OpenHands runs on the issues resolved by both; C_H2, the same on all resolved issues; D,
the seven Bridge skills; E, the five BabyAI instruction kinds. Variants: `main`; `noS`, `main` without the surface
coordinates (`S_CLASS` in `scripts/w1.py`); `pairz`, z-scores computed on the two groups of each pair.

The numbers quoted in App. M.2.1 are in `results.json`: under `main`, 33 of the 52 pairs separated, the largest `cv`
(0.095), and `R`, `B_a` and `B_b` of each pair; under `noS`, 34 pairs separated.

## Files

- `tasks_<variant>.jsonl` (`scripts/w1.py`), one line per W1 value: `kind` (`between`: one sample of each group;
  `within`: two disjoint samples of group `a`), `family`, `a` and `b` (the two groups of the pair; for `within`, `b`
  is the partner of `a`, which matters only in `pairz`; in `main` and `noS` the baseline of a group at one N serves
  every pair with that N), `N` (units per sample), `seed` (of `np.random.default_rng`), `w1` (exact W1), `warning`
  (warning of POT; empty on every line), `sec` (seconds of the transport computation), `key` (task name; `w1.py`
  skips the tasks already in the file).
- `design_<variant>.json` (`scripts/w1.py`), one entry per family: `desc` (description), `cols` (coordinates used),
  `groups`, `excluded` (groups with fewer than 200 units; none), `nunits` (units per group), `nrows` (rows in the pool
  of each group: all rows, or a random 20,000 drawn with seed 12345), `pairs`.
- `results.json` (`scripts/report.py`), for each of `main`, `noS` and `pairz`:
  - `separation`: `n_pairs`, `n_separated`, `majority_separated` (more than half of the pairs separated) and
    `per_family` (`separated`, `pairs`, `majority`);
  - `pairs`, one entry per pair: `family`, `a`, `b`, `N`; `complete` (all 10 seeds of the between-group W1 and of both
    baselines present); `W` and `W_sd` (mean and s.d. over the seeds of the between-group W1); `cv` = `W_sd` / `W`;
    `B_a`, `B_a_sd`, `B_b`, `B_b_sd` (mean and s.d. over the seeds of the baselines of `a` and `b`); `R` = `W` /
    max(`B_a`, `B_b`); `R_meanbase` = `W` / mean(`B_a`, `B_b`); `separated`; `warnings` (POT warnings; 0);
  - `design` (as in `design_<variant>.json`), `n_tasks` (lines of `tasks_<variant>.jsonl`) and `n_warnings` (0).

  `attribution_main` maps `family|a|b` to the coordinates of the family sorted by the one-dimensional W1 between the
  two whole pools (z-scored as in `main`), largest first.
- `table_pairs.md` (`scripts/report.py`), one row per pair of `main`; its last lines give the number of separated
  pairs of each variant and, for `main`, of each family:

  | column | meaning |
  |---|---|
  | Family | A, B, B2, C_H1, C_H2, D or E |
  | Pair | the two groups (simple = pick and place, look = look at an object in light, two = two objects, SciWorld = ScienceWorld, seq = a pick-up followed by a go-to) |
  | N | units per sample |
  | W1 between (sd) | `W` (`W_sd`) |
  | baseline a / b | `B_a` / `B_b` |
  | R = W1/max | `R` |
  | R (noS) | `R` in `noS` (equal to `R` in family C, which has no surface coordinate) |
  | R (pair z) | `R` in `pairz` |
  | top coordinates (1-D W1) | the three coordinates with the largest one-dimensional W1, with that W1 |
  | separated | yes or no |
- `table_families.tex` (`scripts/report.py`): LaTeX table, one row per family: number of groups, number of
  coordinates d, N (range over the pairs), separated pairs, median `R` in `main` and in `noS`.
- `paper_table_pairs.tex` (`extensions/paper_tables.py`): LaTeX rows of the 52 pairs, grouped by family: pair, N,
  `W` (`W_sd`), `B_a` / `B_b`, `R` in `main` (bold if separated) and `R` in `noS`.
- `babyai_drop.json` (`scripts/w1_babyai_drop.py`): family E with the estimator of `main`, without `n_objs`
  (`E_noT9`) and without `n_objs` and `chi11_manip` (`E_noT9_noManip`); each has `cols`, `pairs` (`a`, `b`, `N`,
  `n_seeds`, `W`, `W_sd`, `B_a`, `B_b`, `R`, `separated`, as in `results.json`), `n_separated` (9 in both) and
  `warnings` (0).

## Per-trajectory coordinates (`../features/`)

The inputs of `w1.py`. The coordinate columns of each family are the lists `ALF` (A), `ETO` (B, B2), `HAR` (C), `VLA`
(D) and `BAB` (E) in `scripts/w1.py`; Table 8.1 of `docs/APPENDIX_M_DETAILS.md` defines them. The other columns:

- `eto_features.csv` (`scripts/feat_eto.py`): `env` (alfworld, sciworld or webshop), `uid` (ETO trajectory id),
  `group_alf` (ALFWorld task type) and `alf_target` (its target object), `group_src` (WebShop demonstrator: human or
  gpt), `sci_task` (ScienceWorld task), `n_parsed_actions` and `n_empty_actions` (actions read from the trajectory,
  and those with empty text).
- `harness_features.csv` (`scripts/feat_harness.py`): `group` (`SWEagent_llama70b` or `OpenHands_qwen3coder480b`),
  `instance_id` (SWE-rebench issue), `model_name`.
- `vla_features.csv` (`scripts/feat_vla.py`): `uid` (ECoT episode key, see `../SOURCES.md`), `skill`, `n_steps_meta`
  (number of steps in the ECoT metadata).
- `babyai_<kind>.csv` (`scripts/gen_babyai.py`): `kind`, `seed` (level seed), `mission` (instruction), `bot_error`
  (empty when the bot ran without error), `chi10_oracle_success` (1 when the bot solved the level; `w1.py` uses these
  rows only), `chi7_lateral` and `ref_interact` (share of steps that keep the subgoal depth, and number of
  interaction actions; computed but outside `BAB`).
