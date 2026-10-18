# Results: nested tool calls with open-weight language models (Experiment 8, App. M.4.2)

Eleven instruction-tuned models (table below) answered what a call returns in a registry of one-line Python
functions. The CSV files are written by `code/export_tables.py` from `metrics.json` (the output of
`code/analyze2.py metrics`) and from the files of `code/run2.py` in `per_model/`; the values are copied unchanged.

## Terms

- **Item**: a prompt with an instruction, four fixed worked examples (function names outside the pool of the items)
  and a registry of 26 lines: the d functions of a chain (`def f(x): return g(x) + c` or, for the innermost one,
  `def f(x): return x + c`, with c in {1, 2, 3}), 2 other functions that never call a chain function, and 24 - d note
  lines (`# note:` and nine words), so that the registry has the same length at every d. Every function is listed
  after the function it calls. The question is `What does f1(x0) return?` with x0 in {0, ..., 9}; the answer is x0
  plus the constants of the chain. The model continues after the prefill `Answer:` (greedy decoding, at most 8 new
  tokens, bf16); the first integer of its output is the answer. Qwen3 models run with thinking disabled.
- **Depth d**: number of functions in the chain, on the grid 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24.
- **Calibration and test items**: 100 calibration items and 200 test items per depth, generated from fixed seeds by
  `code/tasks2.py`; `code/verify2.py` checks that no calibration item equals a test item.
- **Conditions** (test items):
  - `I`, one call: the item as above.
  - `E`, one call with the call chain listed: the prompt also contains the line
    `Call chain of the question (each call waits for the next one): f1(x0) -> f2(x0) -> ... -> fd(x0) [base].`
  - `A`, one level per call: the chain is answered in d calls, from the innermost function outward. Each call is a
    fresh prompt with the same registry, in which the definition of the function computed by the previous call is
    replaced by `def g(x): return v  # result for x = x0, returned by a helper call`, where v is the answer that the
    previous call returned; the call asks for the next function outward.
  - `H`, calls of D levels with D = max(1, d-hat) from the calibration items: the chain is cut from the innermost
    function outward into segments of D functions (the outermost segment can be shorter), one call per segment, in
    the same way as in `A`. Chains with d <= D are answered in one call, as in `I`. H was run for both thresholds
    (0.8 and 0.9). For a model with D = 1, H uses the same prompts as A but is a separate run, so its values can
    differ from those of A by a few items (at most 0.015 here), because the outputs of greedy decoding in vLLM
    depend slightly on how the prompts are batched.
  The split uses only which function calls which, never the constants or the answers.
- **d-hat (dhat)**: the calibration accuracy of `I` per depth is fitted by the weighted least-squares fit that may not
  increase with depth (pool adjacent violators, weight = number of items at the depth). d-hat at threshold tau is the
  largest depth of the grid at which the fit is at least tau, 0 if there is none. The paper's d-hat uses tau = 0.8;
  tau = 0.9 is the second threshold. The 95% interval is the 2.5 and 97.5 percentiles of d-hat over 2000 bootstrap
  resamples of the calibration items within each depth.
- **Tokens**: counted with each model's tokenizer; prompt tokens include the chat template and the prefill.

## Models

| model | Hugging Face id | licence |
|---|---|---|
| qwen2.5-7b, qwen2.5-14b, qwen2.5-32b | Qwen/Qwen2.5-{7B,14B,32B}-Instruct | Apache-2.0 |
| olmo2-7b, olmo2-13b | allenai/OLMo-2-1124-{7B,13B}-Instruct | Apache-2.0 |
| olmo2-32b | allenai/OLMo-2-0325-32B-Instruct | Apache-2.0 |
| qwen3-8b, qwen3-14b, qwen3-32b | Qwen/Qwen3-{8B,14B,32B} | Apache-2.0 |
| phi4-14b | microsoft/phi-4 | MIT |
| mistral-small-24b | mistralai/Mistral-Small-24B-Instruct-2501 | Apache-2.0 |

## Which values the paper uses

App. M.4.2 and Figure 6c-d use the column `accuracy` of `test_by_depth.csv` (conditions `I`, `E`, `A`, and `H`
with threshold 0.8), `calls_per_item` of the same file, and `dhat_0.8` of `dhat.csv`. The text gives these values
to two decimals as Python formats the stored values (`'%.2f'`), so 0.875, 0.025, 0.035 and 0.515 appear as 0.88,
0.03, 0.04 and 0.52, and 0.155, 0.245, 0.435, 0.595 and 0.705 as 0.15, 0.24, 0.43, 0.59 and 0.70: `I` at depth 1 is 0.875-1.0, at depth 4 at most 0.515 (qwen2.5-32b) and at
depth 8 at most 0.155; the mean over depths of `E` minus `I` is at most 0.0937 in absolute value (olmo2-32b); for
the eight Qwen, Phi and Mistral models at depth 8, `I` is 0.015-0.155 and `A` 0.435-0.99, and `A` of
mistral-small-24b at depth 24 is 0.705; for the three OLMo-2 models, `A` is 0.035-0.06 and `I` 0.0-0.025 at depth
8, and `A` is 0.37-0.595 and `I` 0.435-0.72 at depth 2; `dhat_0.8` is 1-3, and is 2 or 3 for qwen3-14b, phi4-14b,
qwen2.5-32b and mistral-small-24b, whose `H` (threshold 0.8) at depth 8 uses 4, 4, 3 and 4 calls and reaches 0.44,
0.245, 0.51 and 0.36. Figure 6c shows the mean of `accuracy` over the eight Qwen, Phi and Mistral models (and over the three OLMo-2
models) for `I` and `A`; Figure 6d the mean over those four models for `I`, `A` and `H` (threshold 0.8).

## calibration_by_depth.csv (one row per model and depth, 132 rows)

| column | meaning |
|---|---|
| model | model tag (see the table above) |
| depth | depth d of the chain |
| n_items | number of calibration items at this depth (100) |
| accuracy | fraction of these items answered correctly by one call (`I`) |
| fit | value at this depth of the fit that may not increase with depth (the same fit for both thresholds) |

## dhat.csv (one row per model, 11 rows)

| column | meaning |
|---|---|
| model | model tag |
| hub_id | Hugging Face id of the model |
| family | model family (qwen2.5, olmo2, qwen3, phi4, mistral) |
| parameters_b | nominal number of parameters, in billions |
| dhat_0.8, dhat_0.9 | d-hat at threshold 0.8 and 0.9 on the calibration items |
| dhat_ci95_low_0.8, dhat_ci95_high_0.8, dhat_ci95_low_0.9, dhat_ci95_high_0.9 | bounds of the 95% bootstrap interval of d-hat |
| levels_per_call_0.8, levels_per_call_0.9 | D = max(1, d-hat), the number of functions per call in condition `H` |
| dhat_test_0.8, dhat_test_0.9 | d-hat computed in the same way from the test items of `I` (reported only; the split uses the calibration value) |
| cal_accuracy_d1_d2_ge_0.8 | True if the calibration accuracy of `I` is at least 0.8 at depth 1 and at depth 2 |

## test_by_depth.csv (one row per model, condition and depth, 660 rows)

| column | meaning |
|---|---|
| model | model tag |
| condition | `I`, `E`, `A` or `H` (see Terms) |
| threshold | for `H`: the threshold (0.8 or 0.9) of the d-hat that sets the number of functions per call; empty otherwise |
| levels_per_call | functions per call: 1 for `A`, D for `H`; empty for `I` and `E` (one call for the whole chain) |
| depth | depth d of the chain |
| n_items | number of test items at this depth (200) |
| accuracy | fraction of the items whose final answer (the first integer of the output of the last call) is right |
| accuracy_exact_output | fraction of the items whose last output, without surrounding spaces, is exactly the right integer |
| all_calls_correct | fraction of the items for which every call returned the right value (equal to `accuracy` for `I` and `E`); `accuracy` can be higher, because a last call can be right after a wrong earlier call |
| calls_per_item | mean number of model calls per item |
| tokens_per_item | mean number of tokens per item, prompt and generated tokens summed over all calls |
| prompt_tokens_per_item | mean number of prompt tokens per item, summed over all calls |

## rerun.csv (one row per model, 11 rows)

A random tenth of the test items of `I` (240 items, the same items for both runs) was answered again in a new vLLM
process, with the items in another order, at most 16 prompts per batch and no prefix caching.

| column | meaning |
|---|---|
| model | model tag |
| n_items | number of items answered again |
| accuracy_stored | accuracy of these items in the stored run |
| accuracy_rerun | accuracy of these items in the new run |
| abs_difference | absolute difference of the two accuracies |
| identical_outputs | fraction of these items with exactly the same output text in both runs |
| difference_le_0.03 | True if `abs_difference` is at most 0.03 |

## metrics.json

Output of `code/analyze2.py metrics`, unchanged; the CSV files above are taken from it. `results.<model>.T` holds:
`cal` (calibration: `acc` and `n` per depth; for `tau0.8` and `tau0.9`: `dhat`, `D`, `ci95`, `fit`, `censored` =
d-hat at the deepest depth of the grid; `included` = `cal_accuracy_d1_d2_ge_0.8`), `budgets` (copy of
`budgets_T.json`), `curves` (per condition key `I`, `E`, `A`, `H0.8`, `H0.9`: `accuracy` per depth; keys with
`_strict`: `accuracy_exact_output`; keys with `_allcalls`: `all_calls_correct`), `costs` (per condition: `calls`,
`tokens`, `prompt_tokens` per depth, and `percall_acc_by_nest`: for each number of functions resolved in one call,
the fraction of calls that returned the right value and the number of calls), `n_test` (items per depth),
`dhat_test_0.8` and `dhat_test_0.9` (d-hat and fit from the test items of `I`), and `summary_0.8` and
`summary_0.9`. A summary block gives, for the depths deeper than d-hat (`B`), the mean accuracy (`acc_<c>`), calls
(`calls_<c>`) and tokens (`tokens_<c>`) of each condition c, the same means over all depths (`full_...`), the
differences `H_minus_A` and `H_minus_I` with 95% paired bootstrap intervals (2000 resamples of the items within each
depth), the ratios of calls and tokens of H to A, and the checks a, b, c and noharm:
`a` (the fit from the test items of `I` is below 0.8 at depth 24 and d-hat is below 24), `b_acc` (H on `B` at most
0.02 below A), `b` (`b_acc` and H needs fewer calls than A on `B`), `c` (H at least 0.10 above I on `B` and the
lower bound of the interval of `H_minus_I` above 0), `noharm` (H over all depths at least A - 0.05). `a` is given
only for the threshold 0.8. `generated` is the time of the analysis; `smoke` is false (true marks a run in quick test mode with reduced
sample sizes).

## verify.json

Output of `code/verify2.py`, unchanged: `pass` is true and `failed` is empty if every check passed; `checks` gives
for each check (see the docstring of `code/verify2.py`) whether it passed, the number of cases checked and up to
five failing examples. All checks passed, among them the re-scoring of all 730,600 stored outputs, the
reconstruction of all 664,600 prompts of `A` and `H` from the prompt of `I`, and the recomputation of every
accuracy, call and token mean in `metrics.json`.

## per_model/<model>/

Files written by `code/run2.py` for each model, unchanged.

| file | content |
|---|---|
| dhat_T.json | calibration: `depths`, `acc` (accuracy of `I` per depth), `n` (items per depth); for `tau0.8` and `tau0.9`: `dhat`, `D` = max(1, dhat), `fit` (fit per depth) and `ci95` (bootstrap interval of dhat); `included` (as `cal_accuracy_d1_d2_ge_0.8`) |
| budgets_T.json | for each threshold: `dhat`, `D` and the raw output file of condition `H` with this D |
| T_rerun.json | the row of `rerun.csv` (`n`, `acc_stored`, `acc_rerun`, `abs_diff`, `identical_outputs`, `pass`) with `model`, `task` and the `time` of the re-run |

The raw outputs of the runs (`T_I_cal.jsonl`, `T_I_test.jsonl`, `T_E_test.jsonl`, `T_A_test.jsonl`,
`T_H_D<D>_test.jsonl`; about 10 MB per model, 115 MB in all) are not included; see the README of the repository for
how to produce them.
