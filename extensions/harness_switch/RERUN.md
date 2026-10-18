# Rerunning the agents of Experiment 3 (App. M.2.2)

These commands rerun the 600 episodes (Qwen3-8B on the 300 instances of `data/instances_main.txt`, once in
mini-SWE-agent 2.4.6 and once in OpenHands) and grade them. The tables in `results/` are rebuilt from the included
coordinates without them (section 7 of the repository README).

## Requirements

- One GPU for vLLM 0.31.0, which serves Qwen/Qwen3-8B in bf16.
- Apptainer 1.5.4. `code/harness/build_sqfs.py` builds the image of each instance from the OpenHands agent-server
  image of the instance on GHCR.
- Three environments in this folder:
  - `envs/mini2`: `pip install mini-swe-agent==2.4.6 httpx`;
  - `envs/oh2`: `uv sync --frozen` in a checkout of the OpenHands benchmarks (3 June 2026, software-agent-sdk 1.16.1)
    at `code/benchmarks`, Python 3.12;
  - `envs/tools`: `pip install swebench==5.0.2`.
- Three files in `data/`:
  - `swebench_verified.json`: the 500 instances of SWE-bench Verified keyed by instance id, each with its swebench
    evaluation script;
  - `image_map.json`: written by `python3 code/select_instances.py <tags_json> <tag_prefix>` (arguments in its
    docstring; it also writes `instances_main.txt` and stops if that file exists, so move the released one aside first);
  - `singularity_d_template.tar`: the `/.singularity.d` folder of an image built by Apptainer 1.5.4.
- A Hugging Face cache that holds Qwen/Qwen3-8B. `code/harness_env.sh` keeps all other caches in this folder and sets
  `HF_HUB_OFFLINE=1`; source it before every command.

## Episodes (GPU)

```bash
cd extensions/harness_switch && source code/harness_env.sh && export HF_HOME=/path/to/hf_cache
for i in $(seq 0 299); do python3 code/harness/build_sqfs.py data/instances_main.txt $i; done   # sif/<id>.sqfs
export HARNESS_LLM_KEY=$(python3 -c "import secrets; print(secrets.token_hex(16))"); mkdir -p results/proxy
vllm serve Qwen/Qwen3-8B --host 127.0.0.1 --port 8000 --api-key $HARNESS_LLM_KEY --served-model-name Qwen/Qwen3-8B \
    --max-model-len 32768 --gpu-memory-utilization 0.90 --enable-prefix-caching --enable-auto-tool-choice \
    --tool-call-parser hermes &
envs/oh2/bin/python code/harness/proxy.py 8001 mini http://127.0.0.1:8000 results/proxy/mini.jsonl &
envs/oh2/bin/python code/harness/proxy.py 8002 oh http://127.0.0.1:8000 results/proxy/oh.jsonl &
python3 code/harness/scheduler.py mini data/instances_main.txt 4 http://127.0.0.1:8001/v1 runs/main & S=$!
python3 code/harness/scheduler.py oh data/instances_main.txt 4 http://127.0.0.1:8002/v1 runs/main; wait $S
python3 code/harness/requeue_infra.py main    # then run the two schedulers again
```

The proxy gives every request of both harnesses the same sampling settings and logs it to `results/proxy/`. Episodes
are written to `runs/main/<harness>/<instance>/`.

## Grading, coordinates, W1 and tables (CPU)

```bash
mkdir -p work && for c in gold mini_native mini_common oh_native; do sed "s/$/ $c/" data/instances_main.txt; done > work/eval_tasks.txt
envs/tools/bin/python code/harness/eval_patches.py main work/eval_tasks.txt 0 1200   # eval/main/
export PYTHONPATH=$HARNESS_DIR/code/harness/pyshim; O=work/main; mkdir -p $O/w1
envs/oh2/bin/python code/harness/phi_features.py main $O/features_main.csv 1
envs/oh2/bin/python code/harness/phi_features.py main $O/features_family_c_rule.csv 0
python code/harness/w1.py $O/features_main.csv main $O/w1 && python code/harness/w1.py $O/features_main.csv noLEN $O/w1
python code/harness/w1.py $O/features_family_c_rule.csv family_c_file_rule $O/w1
python code/harness/report.py main $O        # work/main/report.json
python code/make_tables.py --w1-dir $O/w1 --out $O --eval-dir eval/main --instances data/instances_main.txt
```

`make_tables.py` writes the five CSV tables of `results/` to `work/main/`. The two coordinate files written by
`phi_features.py` have the format of `results/trajectory_coordinates.csv` (`features_main.csv`) and
`results/trajectory_coordinates_family_c_rule.csv` (`features_family_c_rule.csv`).
