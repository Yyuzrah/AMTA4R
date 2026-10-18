# Results: all position indices shifted on two checkpoints of Section 5 (Experiment 6, App. M.3.3)

Two of the trained models are tested: f1 = Small Deep MIXED RoPE (GPT-NeoX, 12 layers, width 192, rotary position
encoding, trained on MIXED) and f2 = Large Deep BFS APE (GPT-Neo, 48 layers, width 768, learned absolute position
embeddings, trained on BFS). Under the conditions A16, A64 and Akc every position index of the input is shifted by
the same k (16, 64 or k_c); the tokens are unchanged. All runs use fp32, eager attention and no TF32.

The numbers quoted in App. M.3.3 are, in `teacher_forced.csv` (site type `all`), the 0 flips of f1 out of 94,278
routing tokens under A16 and A64 and the flip rates 0.2571 (A16) and 0.2830 (A64) of f2; and, in `free_running.csv`
(site type `all`), the 2,128 (f1) and 1,143 (f2) problems with a site, lost = gained = 0 for f1 under every shift,
and for f2 the success 0.5267 under `base_resume` and the changes -0.4042 (A16) and -0.3928 (A64).

`tables.md` shows the three summary tables with rounded numbers; the CSV files give every number at full precision,
as `scripts/global_shift_tables.py` computes it. The tables cover the three
conditions that shift every position index (A16, A64, Akc). The run scripts also compute conditions that insert
tokens or shift part of the indices. The two checkpoints are trained with steps 1 and 2 of the main README.

## Terms

- **Reference traces**: the 6,000 trajectories of `data/experiment_500k/val_{bfs,dfs,mixed}.json`, each given to the
  model as BOS + trajectory, cut to 896 tokens so that every shifted input stays below 1,024 positions.
- **Site**: the start of a leaf-level `Exploring Operation` line (the current node has two numbers, the result one
  number) before the first `Goal Reached` or `No solution found` line, whose previous line is either a sibling's
  `No Solution` line at the same node (**bfs type**) or a repeated `Current State` line that follows
  `Moving to Node #X` (**dfs type**), and where an inert operation exists: a valid operation on the node's two numbers
  whose result is not the target and whose (operator, result) is not explored at that node anywhere in the text.
- **k_c**: the number of tokens of the smallest well-formed dead-end node that could be inserted at the site (one
  extra `Exploring Operation` line and its `No Solution` line, plus for the dfs type the `Moving to Node` and
  `Current State` lines of the format): 26-29 tokens for the bfs type, 53-65 for the dfs type. Here it only sets the
  size of the shift in condition Akc; nothing is inserted.
- **Items**: the first site of each reference trace whose token index lies between 16 and 880 and that has at least
  one routing token after it in the window: 2,863 items, 949 of the bfs type and 1,914 of the dfs type (2,909 traces
  have no site and 228 have one only outside the window). Both models use the same items (same tokenizer).
- **Routing tokens**: the tokens after the site, within the 896-token window, that belong to the node id after
  `Moving to Node #` or to `a op b` in `Exploring Operation: a op b=`. The model's prediction of a routing token is
  the most likely next token (top-1) at the position before it, with the reference trace as input.
- **Conditions**: `base` (positions 0, 1, ..., T-1), `A16`, `A64` and `Akc` (every position plus 16, 64 or k_c).
- **Free running**: the first 1,000 problems of each evaluation set (3,000 per model). The unedited run starts from
  BOS + `Current State: {target}:{nums}, Operations: []` and a newline, decodes greedily for at most 512 new tokens,
  and is scored with `metric_fn` of `src/countdown_utils.py`. The site is then searched in the model's own output
  (strict rule: the inert operation must not be explored at that node anywhere in the model's own unedited run; the
  site must lie in the generated part and leave at least 64 of the 512 new tokens). `base_resume` keeps the model's own
  tokens up to the site and continues greedily for the remaining G = 512 - (tokens generated before the site) tokens;
  `A16`, `A64` and `Akc` do the same with every position index, of the kept tokens and of the new ones, shifted by k.
  Success is scored on the prompt, the kept tokens and the continuation.

f2 was trained on BFS trajectories only, so its dfs-type items are traces of a format it was not trained on.

## teacher_forced.csv (one row per model, site type and condition)

| column | meaning |
|---|---|
| model | `f1` or `f2` (the `--model` name of the scripts) |
| checkpoint | configuration name of the trained model |
| encoding | RoPE (rotary) or APE (learned absolute position embeddings) |
| site_type | `all` items, or only the `bfs`-type or `dfs`-type items |
| condition | `A16`, `A64` or `Akc` |
| shift | the shift k of every position index (`k_c`: the item's own k_c) |
| items | number of reference traces with a site (one item each) |
| routing_tokens | number of routing tokens over these items |
| k_c_median | median of k_c over the items |
| base_accuracy | fraction of routing tokens whose top-1 prediction without the shift equals the reference token |
| flips | number of routing tokens whose top-1 prediction changes under the shift |
| flip_rate | flips / routing_tokens |
| accuracy | fraction of routing tokens whose top-1 prediction under the shift equals the reference token |
| accuracy_change | accuracy - base_accuracy |
| max_abs_dlogit | largest absolute change of a next-token logit (any vocabulary entry, any routing position, any item) |

## teacher_forced_per_item.csv (one row per model and item)

| column | meaning |
|---|---|
| model | `f1` or `f2` |
| eval_set, trace_index | evaluation set (`bfs`, `dfs`, `mixed`) and index of the trace in `val_<set>.json` |
| site_type | `bfs` or `dfs` |
| k_c | tokens of the dead-end node at the site (sets the shift of Akc) |
| routing_tokens | routing tokens of the item |
| base_correct | routing tokens predicted correctly without the shift |
| `<cond>_flips`, `<cond>_correct` | for cond = A16, A64, Akc: routing tokens whose prediction changes, and that are predicted correctly, under the shift |
| `<cond>_max_abs_dlogit` | largest absolute logit change of the item under the shift |

## free_running_unedited.csv (one row per model)

| column | meaning |
|---|---|
| model, checkpoint, encoding | as above |
| problems | problems decoded (3,000) |
| unedited_success | fraction of these problems solved by the unedited run |
| problems_with_site | problems whose own output has a site under the strict rule |

## free_running.csv (one row per model, site type and condition)

| column | meaning |
|---|---|
| model, checkpoint, encoding | as above |
| site_type | `all` problems with a site, or those whose site is of the `bfs` or `dfs` type (the type is that of the model's own output) |
| condition | `base_resume` (no shift), `A16`, `A64` or `Akc` |
| shift | 0, 16, 64 or `k_c` (the problem's own k_c) |
| problems | problems in the row |
| k_c_median | median of k_c over these problems |
| unedited_success | fraction of these problems solved by the unedited run |
| resume_token_identical | fraction of these problems whose `base_resume` continuation is token for token the unedited run's own continuation |
| success | fraction of these problems solved under the condition |
| change | success - success of `base_resume` on the same problems |
| lost | problems solved under `base_resume` but not under the condition |
| gained | problems solved under the condition but not under `base_resume` |

## free_running_per_problem.csv (one row per model and problem)

| column | meaning |
|---|---|
| model | `f1` or `f2` |
| problem | `<set>:<index>`: evaluation set and index of the problem in `val_<set>.json` (only target and numbers are used) |
| unedited_success | 1 if the unedited run solves the problem, else 0 |
| site | 1 if the model's own output has a site under the strict rule, else 0 (the remaining columns are then empty) |
| site_type | `bfs` or `dfs` |
| k_c | tokens of the dead-end node at the site |
| budget | G, the new tokens allowed after the site |
| resume_token_identical | 1 if the `base_resume` continuation equals the unedited run's own continuation |
| `<cond>_success` | for cond = base_resume, A16, A64, Akc: 1 if the problem is solved under the condition, else 0 |
