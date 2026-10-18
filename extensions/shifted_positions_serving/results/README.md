# Results: answers under shifted position indices (Experiment 5, App. M.3.2)

33 public base models (12 with learned absolute position embeddings, APE; 5 with ALiBi; 16 with RoPE) answered
3,000 exact-match items each with greedy decoding: 500 items of each of five few-shot tasks (the same items for every
model) and 500 key-value retrieval items of the model's context tier. Each item was run once with the unshifted
position indices (the reference row `ref`) and once per shift k in {16, 64, 256, 1024}, in five scenarios; a shift k
is used for a task only where the longest prompt of the task plus its generation budget plus k fits the model's
trained context. All values are written as `code/analyze.py` produced them and copied unchanged by
`code/export_results.py`, which only renames some columns and removes two lines of each card.

## Scenarios (column `scenario`) and rows (column `row`)

| scenario | rows | what the row does | paper |
|---|---|---|---|
| `offset` | `k16` ... `k1024` | all position indices of prompt and generation shifted by k (fp32) | (i) |
| `leftpad` | `p16` ... `p1024` | every row has max(k) masked pad tokens in front; the reference row takes its positions from the attention mask (real tokens at 0..T-1); row `p<k>` counts positions as if k pads were real, so the real tokens get k..k+T-1 | (ii) |
| `cache` | `block16` ... `block1024` | the prompt is prefilled in two steps (a prefix into the KV cache, then the rest); `block<k>` places both parts k positions later | (iii) |
| `cache` | `gap16` ... `gap1024` | the cached prefix keeps positions 0..m-1 and the rest of the prompt starts at m+k, so the distances between the two parts change (control) | (v) |
| `window` | `k16` ... `k1024` | sliding-window attention over the 128 most recent cache slots and the first 4 slots; the reference row is the same window without the shift; the five few-shot tasks only (2,500 items) | (iv) |
| `offset_bf16` | `k16` ... `k1024` | as `offset`, with the model in bf16; the reference row is the bf16 reference | (vi) |

Every scenario also has the row `rep`: the reference row computed a second time in the same call and placed last, so
that when the rows of an item are split into chunks it measures the numerical noise between chunks. The
GPT-2 models (trained context 1024) have no k = 1024 rows; the models with a trained context of 2048 have k = 1024
rows without the retrieval task (2,500 items).

Tasks (column `task` and `tasks`): `copy`, `reverse`, `brackets` (close all open brackets), `arith` (sum of two
numbers), `countdown` (complete a line of a Countdown search trajectory) and `retrieval` (the word paired with a
code in a list of code-word records; prompts of at most 700 tokens for a trained context of 1024, 1,700 for 2048 and
3,000 for 4096 or more). An answer is the generated text up to the first newline or end-of-sequence token, stripped,
and it is correct if it equals the target exactly.

## table_cells.csv (one row per model, scenario, task and row)

| column | meaning |
|---|---|
| model | model name (Hugging Face Hub id in the cards and in `code/registry.py`) |
| position_encoding | `APE`, `ALiBi` or `RoPE` |
| scenario, row, task | see above |
| n | number of items |
| acc_ref | accuracy of the reference row on these items |
| acc_row | accuracy of this row on the same items |
| delta | acc_row - acc_ref |
| ci_lo, ci_hi | 95% paired percentile bootstrap interval of delta (10,000 resamples of the items) |
| answer_flip_rate | fraction of items whose answer differs from the answer of the reference row (whether or not correctness changes) |
| n_correct_to_wrong | items answered correctly by the reference row and wrongly by this row |
| n_wrong_to_correct | items answered wrongly by the reference row and correctly by this row |
| first_token_flip_rate | fraction of items whose first generated token differs from that of the reference row |
| median_max_abs_logit_diff | median over the items of the largest absolute difference, over the vocabulary, between the logits of this row and of the reference row at the first generated token |
| max_max_abs_logit_diff | the same quantity, largest over the items |
| median_first_token_kl | median over the items of KL(reference row distribution, this row distribution) of the first generated token, in nats |

## table_pooled.csv (one row per model, scenario and row; all tasks of the row together)

| column | meaning |
|---|---|
| model, position_encoding, scenario, row | as above |
| n | number of items over all tasks of the row |
| tasks | the tasks included (space separated) |
| acc_ref, acc_row, delta | as above, over all these items |
| ci_lo, ci_hi | 95% paired percentile bootstrap interval of delta, with the items of each task resampled separately (10,000 resamples) |
| abs_delta_ge_0.02_and_ci_excludes_0 | 1 if abs(delta) >= 0.02 and the interval excludes 0, else 0 |
| answer_flip_rate, first_token_flip_rate | as above, over all these items |
| frac_logit_changed_gt_1e-3 | fraction of the items whose first-step logits differ from the reference row by more than 1e-3 (largest absolute difference) |

Accuracies, deltas, intervals and rates have four decimals; the logit columns have four significant digits and the
KL column three. When delta is 0 for every item of a row, the interval is [0, 0].

## cards/<model>.md (one file per model)

The Hub id, license (the license tag of the Hub; for the four GPT-2 models it is `MIT`, and their license text is a
modified MIT license), position encoding, trained context and number of parameters, then one table row per scenario
and shifted row (the `rep` row is left out): `verdict` is `unchanged` if the 95% interval of delta contains 0,
`degrades` if it lies below 0 and `improves` if it lies above 0; `Delta acc [95% CI]` and `acc ref -> row` come from
table_pooled.csv (accuracies to three decimals); `answer flips` and `first-token flips` are the two flip rates; `median
max abs logit diff` is the median over the tasks of median_max_abs_logit_diff (three significant digits). The
recommendations list, per scenario, whether every shifted row is `unchanged`. A scenario would be marked `(invalid)`
if its unit failed the checks in `code/analyze.py` (`unit_gate`); no unit did.

## Numbers quoted in App. M.3.2

- 40 of 948,000 answers: the sum of answer_flip_rate x n in table_cells.csv (rates per task, n = 500) over the 21
  ALiBi and RoPE models and the rows `k*` of `offset` and `window`, `p*` of `leftpad` and `block*` of `cache`
  (table_pooled.csv rounds the rates to four decimals, so the same sum over it gives 37.1).
- APE at k = 256: delta of `offset/k256`, `leftpad/p256` and `cache/block256` (-0.517 to -0.131) and of `window/k256`
  (-0.5008 to -0.0412); answer_flip_rate of `offset/k256` 0.7053 to 1.0.
- Control (v): `cache/gap64`, largest drop -0.4397 (pythia-1b); the interval of delta excludes 0 in 63 of the 64
  `gap*` rows of the RoPE models and 19 of the 20 of the ALiBi models.
- bf16: `offset_bf16`; pythia-160m has acc_ref 0.0123 (0.2087 in `offset`); the other RoPE models stay within 0.0223
  of the bf16 reference, and the interval excludes 0 only for qwen2.5-1.5b at `k256` and qwen2.5-7b at `k64` and
  `k256`; ALiBi at k = 256 -0.0333 to -0.0067 and at k = 1024 -0.3608 to -0.2792.
