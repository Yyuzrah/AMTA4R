# Results: attention scores of pretrained models (Experiment 4, App. M.3.1)

Eight public checkpoints take part: four with learned absolute position embeddings (APE: `gpt2`, `gpt2-medium`,
`gpt-neo-125m`, `gpt-neo-1.3B`) and four with rotary position embeddings (RoPE: `pythia-160m`, `pythia-410m`,
`pythia-1.4b`, `smollm2-360m`). Each reads 512 sequences of 512 tokens, 128 per input family: `countdown_dfs` and
`countdown_bfs` (reference trajectories of `data/experiment_500k/val_{dfs,bfs}.json`), `dyck` (Dyck-4 strings nested
at most 12 deep) and `code` (files of the Python 3.10 standard library). Each sequence runs in one batch of five rows
with the same tokens: position indices 0..511 (row 0), the same indices again (row 1, the replicate) and every index
shifted by k = 16, 64 and 256 (rows 2-4). `fp32` is the main run (TF32 off, eager attention); `fp64diag` runs every
sequence again in float64 and records the distance of the fp32 scores from the float64 scores.

A denotes the pre-softmax attention scores of row 0 and ΔA their change in a shifted row, taken over heads, attended
pairs (causal; within 256 tokens in the local layers of GPT-Neo) and sequences. The metrics are m1, the change of A
relative to its scale; m2, the fraction of positions whose top-1 next-token prediction changes; and m3, the change of
the log-probability of the next token (shifted minus unshifted, in nats per token).

The numbers quoted in App. M.3.1 come from the rows of `table_cells.csv` with `family` other than `pooled` and k = 16,
64 or 256. APE in fp32: `m1_first_max` 0.42 to 1.19, `m2` 0.140 to 0.874, `m3_signed` -6.77 to -0.20. RoPE in fp32:
`m1_first_max` at most 6.2e-6, `m1_all_max` at most 3.2e-3, `m2` at most 0.00038. RoPE in `fp64diag`: `m1_all_max`
at most 4.4e-12, and `F_base_all_max`, the fp32 rounding without a shift, up to 1.5e-3.

## table_cells.csv (written by `scripts/analyze.py`; one row per model, precision, family and k)

| column | meaning |
|---|---|
| model, encoding | checkpoint and its positional encoding (APE or RoPE) |
| license | license of the checkpoint as listed in `scripts/analyze.py` (`MIT` for GPT-2 stands for its modified MIT license) |
| precision | `fp32`, or `fp64diag` for the float64 run |
| family | input family, or `pooled` over the four families |
| k | shift of every position index; `rep0` compares the replicate row with row 0 |
| m1_first_max, m1_first_mean | first layer: max abs(ΔA) / max abs(A), and sum abs(ΔA) / sum abs(A) |
| m1_all_max | largest, over layers, of the per-layer max abs(ΔA) / max abs(A) |
| m1_all_mean | sum abs(ΔA) / sum abs(A) over all layers |
| m1_first_max_abs, m1_all_max_abs | max abs(ΔA) in the first layer and in all layers |
| m1_first_mean_abs, m1_all_mean_abs | mean abs(ΔA) per attended pair and head, in the first layer and in all layers |
| m2, m2_count, n_positions | fraction and number of positions whose top-1 prediction changes, and number of positions |
| m3_signed, m3_abs | mean change and mean absolute change of the next-token log-probability, in nats (negative: the shift lowers the likelihood) |
| base_nll | mean negative log-likelihood per token without the shift, in nats |
| n_seq | number of sequences |
| F_base_all_max, F_shift_all_max | `fp64diag` rows only: largest, over layers, of max abs(A_fp32 - A_float64) / max abs(A), without and with the shift |
| F_base_all_mean, F_shift_all_mean | `fp64diag` rows only: sum abs(A_fp32 - A_float64) / sum abs(A) over all layers, without and with the shift |

## Other files

- `table_pooled.tex` (`scripts/analyze.py`): LaTeX table, one row per model and k, from the `pooled` rows: m1 in the
  first layer (max, mean) and over all layers (max, mean) in fp32, m1 over all layers (max) in float64, m2 in percent,
  and m3 signed and absolute.
- `paper_table_models.tex` (`extensions/paper_tables.py`): LaTeX rows, one per model; each cell is the range, over the
  12 combinations of family and k, of fp32 `m1_first_max`, fp32 `m1_all_max`, float64 `m1_all_max`, float64
  `F_base_all_max`, fp32 `m2` in percent and fp32 `m3_signed`.
- `summary.json` (`scripts/analyze.py`):
  - `cells`: every row of `table_cells.csv`, keyed `model|precision|family|k`, with further fields:
    `score_scale_first_max` and `score_scale_first_mean` (max abs(A) and mean abs(A) in the first layer),
    `m1_layer_max` and `m1_layer_mean` (m1 in each layer), `argmax_layer` (the layer with the largest
    `m1_layer_max`), `D2_flip_by_pos64` and `D2_absdlp_by_pos64` (m2 and the mean absolute log-probability change in
    eight bins of 64 positions), `D3_flip_margin_max` (largest gap between the top-1 and top-2 logits without the
    shift at a position whose prediction changes; null if none), and in `fp64diag` also `F_base_first_max`,
    `F_shift_first_max` and `logit_fp32_vs_fp64_maxabs_base` (largest logit difference between fp32 and float64
    without the shift).
  - The checks R1-R4 and the controls V1-V3, defined in the docstring of `scripts/analyze.py`: `failing_cells` (the
    cells that fail each check), `rules_all_cells_pass`, `R1_all_layers_holds_all_cells` (whether the fp32 change of
    every RoPE cell over all layers is at most 1e-5; false), `rope_fp64_cells_above_1e-6` (empty), `V1` and
    `V1_pass`, `V2_replicate_values`, `V2_replicate_exact_zero` and `V2_pass`, `V3` and `V3_pass`, and
    `all_checks_pass` (true: R1-R4 hold in every cell and V1-V3 pass).
  - `licenses` and `devices` (the GPU of each run).
- `verify_layer1.json` (`scripts/verify_layer1.py`, control V3): for `gpt2`, `gpt-neo-125m`, `pythia-160m` and
  `smollm2-360m`, the first-layer m1 computed again in float64 on CPU directly from the weights, on the first 2
  sequences of each family (`n_seq` = 8): `independent_fp64_m1_first_max` and `independent_fp64_m1_first_mean` per
  k, the same values from the fp32 run (`captured_fp32_m1_first_max`, `captured_fp32_m1_first_mean`), and the
  relative differences of max abs(A) and sum abs(A) in the first layer between the two (`scale_maxA_rel_diff`,
  `scale_sumA_rel_diff`).
- `raw/<model>_<precision>.json` and `raw/<model>_<precision>.npz` (`scripts/run_offsets.py`): the per-sequence
  statistics from which `analyze.py` computes all of the above.
  - JSON: `tag`, `hub` (Hugging Face model id), `encoding`, `tokenizer`, `n_layers`, `context` (trained context
    length), `T` (512), `KS` (shift of each row: 0, 0, 16, 64, 256), `precision`,
    `S1_maxabs_logit_diff_default_vs_arange` (control V1: largest logit difference between the default forward pass
    and explicit position indices 0..T-1), `device`, `torch` (version), `seconds`, and per family `n_seq`, `item_ids`
    (source of each sequence: evaluation file and index, Dyck seed or file name), `n_pos` (positions with a top-1
    prediction, T) and `n_lp` (positions with a next-token log-probability, T - 1), with one entry per sequence of
    `flips` (per row, positions whose top-1 prediction differs from row 0), `sum_dlp` and `sum_abs_dlp` (per row, sum
    of the change and of the absolute change of the next-token log-probability), `base_nll_sum` (summed negative
    log-likelihood of row 0), `flip_margin_max` (per row, largest top-1 minus top-2 logit gap of row 0 at a changed
    position) and, in `fp64diag`, `logit_fp32_vs_fp64_maxabs` (per row, largest difference between the fp32 and
    float64 logits).
  - NPZ, arrays named `<family>/<name>`: `maxA`, `sumA`, `rmsA` [sequence, layer]: largest, summed and
    root-mean-square abs(A) of row 0; `n` [sequence, layer]: attended pairs times heads; `maxD`, `sumD` [sequence,
    layer, row]: largest and summed abs(A_row - A_row0); `flip_pos` [sequence, row, position]: 1 where the top-1
    prediction differs from row 0; `dlp_pos` [sequence, row, position]: change of the next-token log-probability. In
    `fp64diag` these come from the float64 pass, and `maxF`, `sumF` [sequence, layer, row] are the largest and summed
    abs(A_fp32 - A_float64).

`scripts/make_fig.py` draws `../fig_lemma2.pdf` from `summary.json` (and a PNG preview in this folder): for each model
and k, `m1_first_max`, `m1_all_max` (filled: fp32, hollow: float64), `m2` in percent and `m3_abs`, pooled over the
families, with a bar over the four per-family values. Recomputing `raw/` uses `inputs.json`, the token ids of the
inputs under each tokenizer, which `scripts/prep_inputs.py` writes to this folder.
