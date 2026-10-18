# A Measure-Theoretic Analysis of Reasoning in Transformers: Structural Generalization and Approximation Limits

Yuyang Zhang, Yifu Zhang, Xuehai Zhou, Xiaoyin Chen. NeurIPS 2026. [Paper](paper/paper.pdf)

We map search trajectories to 12 structural features, measure the shift between trajectory distributions with
the Wasserstein-1 distance, and bound out-of-distribution risk in terms of the positional encoding (APE or RoPE),
depth and width of a Transformer. The experiments train 54 Transformer configurations (9 architectures x 3 training
distributions x 2 positional encodings) on Countdown search trajectories in the Stream of Search format and
evaluate each one on BFS, DFS and MIXED trajectories.

## Setup

Python 3.10, PyTorch 2.5.1 (CUDA 12.4) and `transformers` 4.57.1, with all versions pinned in
`requirements.txt`. The classifier analysis in `gbc/` uses a second environment with scikit-learn 1.5.2.

```bash
python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt
python3.10 -m venv .venv-analysis && .venv-analysis/bin/pip install -r requirements-analysis.txt
source .venv/bin/activate
export PYA=$PWD/.venv-analysis/bin/python
```

Each training or evaluation run uses one GPU with bfloat16 support (we used an A100 40 GB) and less than 24 GB
of RAM. Training takes 3-5 h for a Small model and 18-59 h for a Large one. The data take about 8.4 GB of disk
and the 54 trained models about 170 GB (up to 8 GB per model, including its last checkpoint). The GPT-Neo
tokenizer is downloaded from the Hugging Face Hub on first use.

All commands are run from the repository root.

## 1. Data

```bash
python src/generate_dataset_SOS.py --num_samples 500000 --data_dir data/experiment_500k --seed 42
python scripts/make_val_mixed.py --data_dir data/experiment_500k
```

This writes the training sets `train_{bfs,dfs,mixed}.json` (500,000 trajectories each) and the evaluation sets
`val_{bfs,dfs,mixed}.json` (2,000 each) to `data/experiment_500k/`. It runs on one CPU core in about 2 hours and
needs about 20 GB of RAM. The `distribution_metrics.json` written at the end is not used in the paper.

## 2. Training

```bash
EXPS=$(for s in small medium large; do for sh in deep balanced wide; do for d in bfs dfs mixed; do
       for pe in ape rope; do echo ${s}_${sh}_${d}_${pe}; done; done; done; done)
python scripts/submit_experiments.py --exp_ids $EXPS --lr 8e-5 --batch_mode repo --tag full
```

For each configuration this writes `configs/full/<id>.json` and a training and an evaluation job in
`scripts/sbatch_full/`, and submits both to Slurm (the evaluation starts after training). The architectures are
`MODEL_CONFIGS` in `scripts/submit_train_jobs.py`. `--lr 8e-5` is the learning rate used in the paper and
overrides the value in `BASE_TEMPLATE`. `--batch_mode repo` gives the batch sizes used in the paper: 32 for Small
models, and 4 with 8 accumulation steps and gradient checkpointing for Medium and Large models. The jobs request
one A100 on the Slurm partition given by `--partition` (default `gpu`); adjust it and the `#SBATCH` lines to your
cluster.

To train one model without Slurm:

```bash
python scripts/submit_experiments.py --exp_ids small_deep_bfs_ape --lr 8e-5 --batch_mode repo --tag full --dry_run
accelerate launch --num_processes 1 src/train_flexible.py --config configs/full/small_deep_bfs_ape.json
```

The model after 3 epochs is saved to `checkpoints/full/<id>/final_model`. If a run is interrupted, delete
`checkpoints/full/<id>/checkpoint-*` before restarting it (with torch 2.5.1, `transformers` 4.57.1 does not load the
saved optimizer state).

## 3. Evaluation

```bash
python src/eval_flexible.py --model_id small_deep_bfs_ape --checkpoints_dir checkpoints/full \
    --data_dir data/experiment_500k --output_dir analysis/full
```

This writes the accuracy on the three evaluation sets to `analysis/full/<id>.json` (greedy decoding, at most 512
new tokens). The jobs from step 2 run it automatically. APE models are decoded one problem at a time and can
take more than a day; RoPE models are decoded in batches of 32. `results/final_results_54.json` contains the
accuracies of our 54 models. To collect your own in the same format:

```bash
python -c "import glob,json,os; print(json.dumps({os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob('analysis/full/*.json')) if not p.endswith('.partial.json')}, indent=4))" > my_results.json
```

## 4. Structural features and W1 distances

```bash
cd src
python w1_robust.py                                            # W1 estimates (App. G)
python oracle_analysis.py --data_dir ../data/experiment_500k   # statistics of the reference trajectories
cd ..
```

The 12 structural features (App. A.1) are computed by `extract_structural_features` in
`src/distribution_metrics.py`. `w1_robust.py` computes the exact W1 (`ot.emd2`) between samples of z-scored
features (N = 2,000 per side, 10 seeds) and the debiased Sinkhorn divergence used as a cross-check. Both scripts
write to `data/experiment_500k/` and take a few minutes on CPU; our outputs are in `results/`.

## 5. Structural sufficiency check (App. J)

For two of the trained models, Small Deep MIXED RoPE and Large Deep BFS APE, a gradient-boosted classifier
(scikit-learn `HistGradientBoostingClassifier`, default settings) predicts from the 12 structural features of a
newly generated reference trajectory whether the model solves its problem, i.e. the model's zero-one loss on
that problem. The metrics are computed out of fold with stratified 5-fold cross-validation, on 50,000 problems
for the Small model and 3,000 for the Large model.

| Model | ROC-AUC | McFadden pseudo-R² |
|---|---|---|
| Small Deep MIXED RoPE | 0.927 | 0.51 |
| Large Deep BFS APE | 0.929 | 0.48 |

McFadden's pseudo-R² is 1 - log L_model / log L_null, where the null model always predicts the base rate of the
training folds; it is the fraction by which the classifier reduces the log-loss of this null model. App. J compares
the values with simpler inputs such as the token length of the trajectory and the problem alone.

The labels and features are included, so the metrics can be recomputed on CPU in one or two minutes:

```bash
(cd gbc/labels && gunzip -k *.jsonl.gz)
$PYA gbc/analyze.py       # writes gbc/results_f1.json (Small) and gbc/results_f2.json (Large)
```

In both files, `pooled_F12` gives `auc` and `pseudo_r2` as [mean, s.d.] over the five folds. To regenerate the
labels and features (this needs the two models from step 2; labeling takes about 17 GPU-hours):

```bash
python gbc/gen_fresh.py   # gbc/data/fresh_{bfs,dfs,mixed}.json (seed 2026)
$PYA gbc/overlap.py       # marks problems that also occur in a training set
for m in bfs dfs mixed; do
  n=$([ $m = mixed ] && echo 16666 || echo 16667)
  python gbc/label.py --ckpt checkpoints/full/small_deep_mixed_rope/final_model --measure $m \
      --start 0 --end $n --out gbc/labels/f1_small_deep_mixed_rope_$m.jsonl
  for s in 0 500; do
    python gbc/label.py --ckpt checkpoints/full/large_deep_bfs_ape/final_model --measure $m \
        --start $s --end $((s+500)) --out gbc/labels/f2_large_deep_bfs_ape_${m}_$s.jsonl
  done
done
python gbc/featurize.py   # gbc/features.npz
```

`label.py` appends to `--out` and skips problems that are already there, so delete any unpacked
`gbc/labels/*.jsonl` first.

The other numbers in App. J come from `gbc/subpopulations.py`, `gbc/operand_stats.py`, `gbc/table2_folds.py` and
`gbc/analyze_loss.py` (label coded as the zero-one loss); run them with `$PYA` after unpacking the labels
(`operand_stats.py` also needs the trajectories from `gen_fresh.py`).

## 6. Figures and statistics

```bash
python figures/make_figures.py --results results/final_results_54.json --outdir figures/out
python analysis/results_stats.py
```

`make_figures.py` writes Figures 1-3 as PDF files (use `--results my_results.json` for your own models).
`results_stats.py` prints the statistics quoted in the paper, such as the RoPE vs. APE counts and sign test, the
Deep vs. Wide gaps, the width comparisons and the averages per training distribution.

## 7. Empirical checks of the terms of the main bound (App. M)

App. M, "Empirical Checks of the Terms of the Main Bound", reports eight experiments, and `extensions/` holds the code
and results of each in its own folder. The `results/README.md` of each experiment folder explains its files and
columns; the figures of App. M are listed at the end of this section. Table 3 (App. M.5) summarizes the eight
experiments with numbers from their result folders. [`docs/APPENDIX_M_DETAILS.md`](docs/APPENDIX_M_DETAILS.md)
describes the designs and the catalogue of candidate coordinates. The scripts skip outputs that already exist; move the
released files aside to recompute them. Model weights are not included; the scripts download the public checkpoints
from the Hugging Face Hub or use the models of step 2.

**Experiment 1: more coordinates on Countdown (App. M.1.1)**, `extensions/coordinates_12_to_34/`: the classifier of
App. J and the W1 estimates of App. G on nested projections with 12, 16, 24 and 34 coordinates, in the analysis
environment (POT 0.9.5). The tables are rebuilt from the per-run files in `results/runs/` in seconds:

```bash
X=extensions/coordinates_12_to_34/scripts
$PYA $X/export_tables.py   # results/*.csv, results/tables.md
```

The full recomputation runs on CPU. It needs the data of step 1 and the unpacked labels and fresh trajectories of
step 5, and writes the per-trajectory coordinates and the runs to `outputs/coordinates_12_to_34/` (or `$COORD_BASE`):

```bash
for m in bfs dfs mixed; do $PYA $X/prep_pools.py --measure $m; done
for t in $(seq 0 8); do python $X/featurize.py $t; done   # main environment (GPT-Neo tokenizer)
$PYA $X/gate.py                                            # checks against App. G and App. J
for t in $(seq 0 19); do $PYA $X/auc_task.py $t; done
for t in $(seq 0 39); do $PYA $X/w1_task.py $t; done
$PYA $X/w1_rescue.py && $PYA $X/analyze.py
$PYA $X/export_tables.py --base outputs/coordinates_12_to_34 --out outputs/coordinates_12_to_34/results
```

**Experiment 2: structural shift between groups of agent, embodied and robot trajectories (App. M.2.1, Figure 4)**,
`extensions/structural_audit/`: per-trajectory coordinates (`features/`) of ETO, SWE-agent and OpenHands, Bridge/ECoT
and BabyAI trajectories, and the exact W1 between groups and the within-group baselines for the 52 pairs
(`results/`). From the included features, in about two minutes on 16 CPU cores:

```bash
S=extensions/structural_audit/scripts
rm extensions/structural_audit/results/tasks_*.jsonl   # w1.py skips the tasks listed there
for v in main noS pairz; do python $S/w1.py $v 16; done
python $S/report.py && python $S/w1_babyai_drop.py 8   # results/*.json, *.md, *.tex; fig_ratios.pdf (Figure 4)
python extensions/paper_tables.py   # LaTeX rows of all pairs and of the ranges per model of Experiment 4
```

From the source data (about 4.6 GB, downloaded to `extensions/structural_audit/data/`); the BabyAI levels are
generated in a Python 3.11 environment:

```bash
python $S/download_data.py
python $S/extract_phi_swe.py 0 8 && python $S/extract_phi_openhands.py 6
python $S/feat_eto.py && python $S/feat_harness.py && python $S/feat_vla.py
python3.11 -m venv .venv-babyai && .venv-babyai/bin/pip install -r extensions/structural_audit/requirements-babyai.txt
.venv-babyai/bin/python $S/gen_babyai.py all 4000 40000
```

`extensions/structural_audit/SOURCES.md` lists the sources, their licenses and what the released files contain.

**Experiment 3: one model in two agent harnesses (App. M.2.2)**, `extensions/harness_switch/`: Qwen3-8B runs 300
SWE-bench Verified instances (`data/instances_main.txt`) in mini-SWE-agent 2.4.6 and in OpenHands (software-agent-sdk
1.16.1 with the OpenHands benchmarks of 3 June 2026); `results/` holds the 16 coordinates of the 600 episodes, the W1
between the two harnesses and the resolve rates. The tables are rebuilt from the included coordinates on CPU in about
15 seconds:

```bash
H=extensions/harness_switch; W=outputs/harness_switch/w1; mkdir -p $W
python $H/code/harness/w1.py $H/results/trajectory_coordinates.csv main $W
python $H/code/harness/w1.py $H/results/trajectory_coordinates.csv noLEN $W
python $H/code/harness/w1.py $H/results/trajectory_coordinates_family_c_rule.csv family_c_file_rule $W
python $H/code/make_tables.py --w1-dir $W --out $H/results   # reads results/resolve_by_instance.csv
```

Running the 600 episodes again needs one GPU (vLLM 0.31.0), Apptainer 1.5.4 and swebench 5.0.2;
`extensions/harness_switch/RERUN.md` lists the environments, input files and commands.

**Experiment 4: attention scores of pretrained models (App. M.3.1)**, `extensions/lemma2_pretrained/`: four
learned-APE models (GPT-2 124M and 355M, GPT-Neo 125M and 1.3B) and four RoPE models (Pythia 160M, 410M and 1.4B,
SmolLM2 360M), 512 sequences of 512 tokens per model, every position index shifted by k = 16, 64 or 256.
`results/raw/` holds the per-sequence statistics, so the analysis runs on CPU in seconds:

```bash
L=extensions/lemma2_pretrained/scripts
python $L/analyze.py && python $L/make_fig.py   # results/summary.json, table_*.{csv,tex}; fig_lemma2.pdf
```

Recomputing `results/raw/` needs the data of step 1, Python 3.10.19 (its standard library provides the `code` inputs)
and about 22 minutes on one H100 GPU:

```bash
M="gpt2 gpt2-medium gpt-neo-125m gpt-neo-1.3B pythia-160m pythia-410m pythia-1.4b smollm2-360m"
python $L/download_models.py && python $L/prep_inputs.py   # results/inputs.json
python $L/run_offsets.py --precision fp32 --models $M && python $L/run_offsets.py --precision fp64diag --models $M
python $L/verify_layer1.py   # CPU: independent float64 layer-1 scores
```

`download_models.py` downloads the checkpoints from the Hugging Face Hub (GPT-2: modified MIT, GPT-Neo: MIT, Pythia
and SmolLM2: Apache-2.0).

**Experiment 5: answers under shifted position indices (App. M.3.2, Figure 5)**,
`extensions/shifted_positions_serving/`: 33 public models from the Hugging Face Hub answer 3,000 items each under the
conditions (i)-(vi) of App. M.3.2, run as the five scenarios offset, leftpad, cache, window and offset_bf16; `results/` holds the tables and one card per model.
Recomputing the tables needs the answers (about 500 MB) of a full run in the main environment, split into 165 units
(model x scenario) of one GPU each. `pc_run.py` stops after 5 hours with exit code 3; the same command continues.

```bash
V=extensions/shifted_positions_serving; C=$V/code
python $C/make_fewshot_items.py                            # fewshot/data/items_*.json (needs step 1)
for i in $(seq 0 32); do python $C/fetch.py $i $V; done    # meta/fetch/<model>.json
python $C/make_items.py $V main                            # data/items_main.json, data/plan_main.json
mkdir -p $V/results/markers && python $C/test_engine.py $V/results/markers/unittest.json   # CPU unit tests
for u in $(seq 0 164); do python $C/pc_run.py --base $V --unit $u; done
python $C/gatecheck.py $V    # units listed under "rescue" in results/gatecheck.json: rerun them with --strict
python $C/analyze.py $V main && python $C/export_results.py $V   # results/table_*.csv, results/cards/
```

One check of `analyze.py` compares the answers with a second run of the five few-shot tasks in
`<FEWSHOT_DIR>/results/raw/` (`$FEWSHOT_DIR`, by default `$V/fewshot`) and writes `fewshot_run_agreement`, which stays
empty when that folder has no such run.

**Experiment 6: two Countdown models of Section 5 (App. M.3.3)**, `extensions/shifted_positions_checkpoints/`: the
routing tokens of Small Deep MIXED RoPE and Large Deep BFS APE of step 2 with all position indices shifted, on
reference trajectories (teacher-forced) and in the models' own generations (free-running). The tables are rebuilt from
the included per-item and per-problem files on CPU in seconds:

```bash
P=extensions/shifted_positions_checkpoints/scripts
python $P/global_shift_tables.py   # results/teacher_forced*.csv, free_running*.csv, tables.md
```

To recompute them on one GPU (outputs in `outputs/shifted_positions_checkpoints/results/`):

```bash
mkdir -p outputs/shifted_positions_checkpoints/results && python $P/make_items.py
for m in f1 f2; do
  python $P/tf_run.py --model $m
  python $P/fr_run.py --model $m --stage base && python $P/fr_run.py --model $m --stage cond
done
python $P/global_shift_tables.py --raw outputs/shifted_positions_checkpoints/results --out outputs/shifted_positions_checkpoints/results
```

The run scripts also compute conditions that insert tokens or leave a gap in the position indices, and
`scripts/common.py` also lists public GPT-Neo and Pythia models; the tables use the two models of step 2 and the
conditions that shift all indices.

**Experiment 7: nested plans (App. M.4.1, Figure 6a-b)**, `extensions/nested_plans/`: 120 small Transformers trained
from scratch on nested plans and scored with each plan in one call and split into separate model calls; `results/runs/`
holds the `results.json` of every run. Recomputing the tables needs the checkpoints and per-plan scores of a full run
(main environment, one GPU per run) of `train.py`, `gatecheck.py`, `verify.py`, `analyze.py` and `export_tables.py`,
in this order:

```bash
N=extensions/nested_plans; mkdir -p $N/runs/markers
python $N/code/test_handoff.py && echo '{"pass": true}' > $N/runs/markers/unittest.json   # unit tests (CPU)
tail -n +2 $N/code/grid.tsv | while IFS=$'\t' read idx name pe dt model L d h seed steps; do
  python $N/code/train.py --workers 5 --name $name --root $N/runs --pe $pe --L $L --d $d --h $h --seed $seed \
      --dtrain $dt --steps $steps --grid_idx $idx      # exit code 3: rerun to continue
done
python $N/code/gatecheck.py $N $N/code/grid.tsv   # rows of runs/gatecheck.json "rescue_idx": train again with
                                                  # --root $N/runs_ext --steps 60000
python $N/code/verify.py $N/runs/* $N/runs_ext/*
python $N/code/analyze.py $N $N/code/grid.tsv && python $N/code/export_tables.py
```

**Experiment 8: nested tool calls (App. M.4.2, Figure 6c-d)**, `extensions/nested_tool_calls/`: eleven open-weight
instruction-tuned models, served by vLLM 0.31.0 in bf16, answer chains of nested Python function calls in one call and
split into separate model calls (models of 24B and more need one 80 GB GPU). The CSV tables are rebuilt on CPU from
`results/metrics.json` and `results/per_model/`:

```bash
(cd extensions/nested_tool_calls && python code/export_tables.py)   # results/*.csv
```

To rerun the models (from `extensions/nested_tool_calls/`, in an environment with vLLM 0.31.0):

```bash
python code/test_tasks2.py && mkdir -p meta && python code/fetch_models.py
M="qwen2.5-7b qwen2.5-14b olmo2-7b olmo2-13b qwen3-8b qwen3-14b phi4-14b qwen2.5-32b olmo2-32b qwen3-32b mistral-small-24b"
for m in $M; do python code/run2.py --model $m --task T --unit cal; done
python code/analyze2.py gate                      # analysis/d_star.json, from the calibration split only
for m in $M; do for u in I E A H rerun; do python code/run2.py --model $m --task T --unit $u; done; done
python code/analyze2.py metrics && python code/verify2.py && python code/analyze2.py final
python code/export_tables.py --metrics analysis/metrics.json --per-model results --out results
```

**Figures 4, 5 and 6.** `report.py` of Experiment 2 draws Figure 4 (`extensions/structural_audit/fig_ratios.pdf`). The
two scripts in `extensions/figures_appendix_m/` draw Figures 5 and 6 on CPU in the main environment, from the result
tables of Experiments 5, 7 and 8:

```bash
F=extensions/figures_appendix_m; mkdir -p outputs/figures
python $F/make_fig_shift.py outputs/figures   # Figure 5, from shifted_positions_serving/results/table_pooled.csv
python $F/make_fig_split.py outputs/figures   # Figure 6, from nested_plans/results and nested_tool_calls/results
```

`extensions/figures_appendix_m/fig_split.pdf` is Figure 6, and `make_fig_shift.py` writes Figure 5 as
`fig_shift.pdf`.

## Paper items

| Paper | Code | Output |
|---|---|---|
| Table 1 | `MODEL_CONFIGS` in `scripts/submit_train_jobs.py` | |
| Figure 1, Figure 2 (App. H.1), Figure 3 (App. H.2) | `figures/make_figures.py` | `figures/out/*.pdf` |
| Accuracy statistics in Sec. 5 and App. G.3, H, I, K | `analysis/results_stats.py` | printed |
| W1 estimates (Sec. 5, App. G) | `src/w1_robust.py` | `results/w1_robust.json` |
| Reference trajectories (Sec. 5, App. G.3, App. K) | `src/oracle_analysis.py` | `results/oracle_analysis.json` |
| App. J | `gbc/*.py` | `gbc/results_*.json` |
| App. K (data, training, evaluation) | `src/generate_dataset_SOS.py`, `scripts/submit_experiments.py`, `src/train_flexible.py`, `src/eval_flexible.py` | `results/final_results_54.json` |
| App. M.1 (how the coordinates of Φ are chosen; catalogue of candidate coordinates) | | `docs/APPENDIX_M_DETAILS.md` |
| App. M.1.1 (Experiment 1) | `extensions/coordinates_12_to_34/scripts/*.py` | `extensions/coordinates_12_to_34/results/` |
| App. M.2.1 and Figure 4 (Experiment 2) | `extensions/structural_audit/scripts/*.py` | `extensions/structural_audit/results/`, `extensions/structural_audit/fig_ratios.pdf` |
| App. M.2.2 (Experiment 3) | `extensions/harness_switch/code/` | `extensions/harness_switch/results/` |
| App. M.3.1 (Experiment 4) | `extensions/lemma2_pretrained/scripts/*.py` | `extensions/lemma2_pretrained/results/` |
| App. M.3.2 and Figure 5 (Experiment 5) | `extensions/shifted_positions_serving/code/`, `extensions/figures_appendix_m/make_fig_shift.py` | `extensions/shifted_positions_serving/results/`, `outputs/figures/fig_shift.pdf` |
| App. M.3.3 (Experiment 6) | `extensions/shifted_positions_checkpoints/scripts/` | `extensions/shifted_positions_checkpoints/results/` |
| App. M.4.1 and Figure 6a-b (Experiment 7) | `extensions/nested_plans/code/`, `extensions/figures_appendix_m/make_fig_split.py` | `extensions/nested_plans/results/`, `extensions/figures_appendix_m/fig_split.pdf` |
| App. M.4.2 and Figure 6c-d (Experiment 8) | `extensions/nested_tool_calls/code/`, `extensions/figures_appendix_m/make_fig_split.py` | `extensions/nested_tool_calls/results/`, `extensions/figures_appendix_m/fig_split.pdf` |
| Table 3 (App. M.5, summary of the eight experiments) | the code of Experiments 1-8 above | the result folders of Experiments 1-8 above |

The Countdown generator, search procedures and scorer in `src/countdown*.py` are from
[Stream of Search](https://github.com/kanishkg/stream-of-search).

## Citation

```bibtex
@inproceedings{zhang2026measure,
  title     = {A Measure-Theoretic Analysis of Reasoning in Transformers: Structural Generalization and Approximation Limits},
  author    = {Zhang, Yuyang and Zhang, Yifu and Zhou, Xuehai and Chen, Xiaoyin},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

Apache License 2.0 (see `LICENSE`).
