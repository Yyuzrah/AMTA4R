"""Family C of App. M.2: harness groups on code-agent reference runs (nebius/SWE-agent-trajectories and
nebius/SWE-rebench-openhands-trajectories, both CC-BY-4.0).

Reference runs = resolved runs of swe-agent-llama-70b (SWE-agent harness) and qwen3-coder-480b (OpenHands v0.54
harness).  Per-step parsing and the common step schema are those of extract_phi_swe.py and extract_phi_openhands.py;
this script reads their per-trajectory output (data/harness/phi_swe_full.csv, data/harness/phi_oh_full.csv) and
selects 16 coordinates that are defined in both harnesses.  Excluded as outcome-revealing or harness-only:
target/resolved, exit_status, submitted_in_trace, recovery_rate, task_tracker/think shares (OpenHands only),
token/char counts, issue length.
Phi_H (16): T1 max focus-stack depth | T2 mean | T3 s.d. | T4 tool calls (= S1 events) | T5 distinct files opened |
T6 file reopen ratio | T6a repeated identical actions share | C1 push (new file) rate | C2 pop (return to stacked
file) rate | C7 failed edits | C9 longest failed-edit cascade | C14 run/test commands | C15 position of the first edit
(share of calls before it) | I1 read calls (view+search) | I1 write calls (edit) | I3 error observations per call.
Output: features/harness_features.csv (resolved runs only). Idempotent.
"""
import os
import numpy as np, pandas as pd

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "harness")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "features")
COLS = ["T1_max_depth", "T2_mean_depth", "T3_sd_depth", "T4_tool_calls", "T5_distinct_files", "T6_file_reopen",
        "T6a_repeat_action", "C1_push_rate", "C2_pop_rate", "C7_failed_edits", "C9_failed_edit_cascade",
        "C14_run_calls", "C15_first_edit_pos", "I1_read_calls", "I1_write_calls", "I3_error_rate"]


def conv(df, group):
    N = df["len_turns"].astype(float)
    out = pd.DataFrame(dict(
        group=group, instance_id=df["instance_id"].values, model_name=df["model_name"].values,
        T1_max_depth=df["chi1_max_depth"], T2_mean_depth=df["chi2_mean_depth"], T3_sd_depth=df["chi3_std_depth"],
        T4_tool_calls=df["chi4_volume"], T5_distinct_files=df["distinct_files_opened"],
        T6_file_reopen=df["open_revisit_rate"], T6a_repeat_action=df["repeat_action_rate"],
        C1_push_rate=df["chi5_down_rate"], C2_pop_rate=df["chi6_up_rate"], C7_failed_edits=df["chi8_failed_edits"],
        C9_failed_edit_cascade=df["max_failed_edit_cascade"], C14_run_calls=df["n_test_runs"],
        C15_first_edit_pos=df["first_edit_pos"], I1_read_calls=((df["frac_view"] + df["frac_search"]) * N).round(),
        I1_write_calls=(df["frac_edit"] * N).round(), I3_error_rate=df["err_obs_rate"]))
    return out.reset_index(drop=True)


def main():
    path = os.path.join(OUT, "harness_features.csv")
    if os.path.exists(path):
        print("exists", path); return
    a = pd.read_csv(os.path.join(SRC, "phi_swe_full.csv"))
    b = pd.read_csv(os.path.join(SRC, "phi_oh_full.csv"))
    a = a[(a.model_name == "swe-agent-llama-70b") & (a.target == 1)]
    b = b[b.target == 1]
    df = pd.concat([conv(a, "SWEagent_llama70b"), conv(b, "OpenHands_qwen3coder480b")], ignore_index=True)
    assert df[COLS].notna().all().all()
    df.to_csv(path + ".tmp", index=False); os.replace(path + ".tmp", path)
    print("wrote", path, df.groupby("group").agg(n=("instance_id", "size"), issues=("instance_id", "nunique")).to_dict())


if __name__ == "__main__":
    main()
