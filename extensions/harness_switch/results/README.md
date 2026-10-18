# Results: one model in two harnesses (Experiment 3, App. M.2.2)

Qwen3-8B ran once in mini-SWE-agent and once in OpenHands on each of 300 SWE-bench Verified instances
(`../data/instances_main.txt`). All 300 episodes finished in both harnesses. "mini" is mini-SWE-agent, "openhands"
(or "oh" in the trajectory files) is OpenHands. Numbers are written at full precision, as the scripts produced them.

Variants (used in the first three files):

- `all_coordinates`: the 16 coordinates of the trajectory files below (the paper's numbers).
- `without_tool_calls`: the same without `T4_tool_calls` (15 coordinates).
- `family_c_file_rule`: the 16 coordinates computed with the file rule of family C of the structural audit, where only
  editor views and edits set the file focus; bash commands that read or write one file do not
  (`trajectory_coordinates_family_c_rule.csv`). mini-SWE-agent has only a bash tool, so its file focus is always empty here.

## w1_summary.csv (one row per variant)

| column | meaning |
|---|---|
| variant | see above |
| coordinates | number of coordinates used |
| n_instances | instances finished in both harnesses (pool size per harness) |
| N_per_sample | trajectories per sample in every W1 estimate, floor(n_instances / 2) |
| w1_between | exact W1 between N mini and N OpenHands trajectories (drawn without replacement), mean over 10 seeds |
| w1_between_sd | standard deviation of that W1 over the 10 seeds |
| baseline_mini | W1 between two disjoint halves of the mini pool (first N against next N of one random order), mean over 10 seeds |
| baseline_mini_sd | standard deviation of baseline_mini over the seeds |
| baseline_openhands, baseline_openhands_sd | the same for the OpenHands pool |
| baseline_max | the larger of the two baselines |
| R | w1_between / baseline_max |
| CV | w1_between_sd / w1_between |

Coordinates are z-scored over the union of both pools (mean and standard deviation of all 600 trajectories; a
standard deviation of 0 is replaced by 1); the cost is Euclidean and all trajectories have equal weight.

## w1_per_seed.csv (one row per variant, seed and estimate)

| column | meaning |
|---|---|
| variant | see above |
| kind | `between` (mini against OpenHands) or `within` (two halves of one harness's pool) |
| pools | `mini_vs_openhands`, `mini` or `openhands` |
| seed | seed of `numpy.random.default_rng` (0 to 9) |
| n_instances, N_per_sample | as in w1_summary.csv |
| w1 | the exact W1 of this estimate |

## coordinate_attribution.csv (one row per variant and coordinate, largest w1_1d first)

| column | meaning |
|---|---|
| variant | see above |
| coordinate | coordinate name (see the trajectory files) |
| w1_1d | one-dimensional W1 between all 300 mini and all 300 OpenHands values of this coordinate (z-scored as above) |
| baseline_1d | one-dimensional W1 between two halves of one harness's values (N against N, mean over 10 seeds), the larger of the two harnesses |
| ratio_1d | w1_1d / baseline_1d |
| smd_openhands_minus_mini | standardized mean difference: (mean under OpenHands - mean under mini) / sqrt((variance under mini + variance under OpenHands) / 2), on the raw values |
| mean_mini | mean of the raw coordinate under mini-SWE-agent |
| mean_openhands | mean of the raw coordinate under OpenHands |
| share | w1_1d divided by the sum of w1_1d over the coordinates of the variant |

## resolve_by_instance.csv (one row per instance)

| column | meaning |
|---|---|
| instance_id | SWE-bench Verified instance |
| gold_patch_valid | 1 if the dataset's reference patch resolves the instance on our grading setup, else 0 (only `psf__requests-5414` fails) |
| resolved_mini_native | 1 if the patch that mini-SWE-agent submitted resolves the instance |
| resolved_mini_common | 1 if the final working tree of the mini episode (all changes, new files included, against the base commit) resolves the instance |
| resolved_openhands | 1 if the OpenHands patch resolves the instance (OpenHands always takes the final working tree, so its native and common patch are the same) |
| reason_gold, reason_mini_native, reason_mini_common, reason_openhands | how the grading ended: `graded` (tests ran), `empty_patch` (nothing to apply; counts as not resolved), `patch_apply_failed` (the patch did not apply; not resolved) |

## resolve_rates.csv (one row per patch rule)

Rates are computed on the 299 instances with `gold_patch_valid` = 1.

| column | meaning |
|---|---|
| patch_rule | `native`: each harness's own patch (the paper's numbers); `common`: the final working tree in both harnesses |
| n_instances | number of instances used |
| rate_mini, rate_mini_ci_low, rate_mini_ci_high | resolve rate of mini-SWE-agent and its 95% percentile bootstrap interval (10,000 resamples of instances, seed 0) |
| rate_openhands, rate_openhands_ci_low, rate_openhands_ci_high | the same for OpenHands |
| diff_openhands_minus_mini, diff_ci_low, diff_ci_high | difference of the two rates and its 95% paired bootstrap interval (same resamples) |
| resolved_only_mini | instances resolved by mini-SWE-agent only |
| resolved_only_openhands | instances resolved by OpenHands only |
| resolved_both | instances resolved by both |
| mcnemar_p | exact two-sided McNemar p-value (binomial test on the two discordant counts) |

## trajectory_coordinates.csv and trajectory_coordinates_family_c_rule.csv (one row per harness and instance)

A step is one tool call among those issued by the first 100 LLM calls of the episode. The file focus is the file a
step views or edits (an editor view or edit; in trajectory_coordinates.csv also a bash command that reads or writes
exactly one file). The focus stack holds the files in focus: a file not on the stack is pushed, a file already on it
pops the stack back to that file.

| column | meaning |
|---|---|
| harness | `mini` (mini-SWE-agent) or `oh` (OpenHands) |
| instance_id | SWE-bench Verified instance |
| exit_status | how the episode ended. mini: `Submitted`, `LimitsExceeded` (100 LLM calls), `ContextWindowExceededError` (the request no longer fit the model's 32,768-token context), `RepeatedFormatError`. No episode reached the 3000 s wall-time limit. OpenHands: `finished`, `stuck` (OpenHands' loop detector stopped the agent), `error` (the agent ended in OpenHands' error state), `LimitsExceeded` (100 LLM calls) |
| T1_max_depth | largest depth of the focus stack |
| T2_mean_depth | mean depth of the focus stack over the steps |
| T3_sd_depth | standard deviation of that depth over the steps |
| T4_tool_calls | number of steps |
| T5_distinct_files | number of distinct files viewed |
| T6_file_reopen | share of file views that view a file viewed before |
| T6a_repeat_action | share of steps that repeat an earlier identical action |
| C1_push_rate | pushes onto the focus stack divided by (steps - 1) |
| C2_pop_rate | pops back to a file on the stack divided by (steps - 1) |
| C7_failed_edits | number of failed edits: an editor edit that returns an error, or (single-file rule only) a bash write to one file with nonzero exit code |
| C9_failed_edit_cascade | longest run of failed edits not broken by a successful file edit |
| C14_run_calls | number of steps whose command starts with python, pytest, tox, bash, ./, sh, make, pip, nosetests, timeout or PYTHONPATH |
| C15_first_edit_pos | index of the first file edit (editor edit, or single-file bash write) divided by the number of steps (1 if there is none) |
| I1_read_calls | number of view steps (editor view of a file; cat, head, tail, less, nl, awk, sed -n) and search steps (editor view of a directory; grep, find, ls, rg, tree, git, pwd, which) |
| I1_write_calls | number of edit steps (editor create, str_replace, insert, undo_edit; bash commands starting with sed, rm, mv, cp, mkdir, touch, or echo with a redirection; single-file bash writes) |
| I3_error_rate | share of steps whose output shows an error (nonzero exit code, editor error, Python traceback or a known error message in the first 400 characters) |
| n_llm_calls | LLM calls counted for the steps (at most 100) |
| n_tool_calls | tool calls issued by those LLM calls (equals T4_tool_calls) |
| n_known | tool calls whose tool the extraction recognises (bash; terminal, file_editor, finish, task_tracker, think) |
| len_chars | characters of model output and tool output over the steps (not a coordinate) |
| n_llm_calls_total | OpenHands only: all LLM responses of the episode, also those after the 100th (empty for mini) |
