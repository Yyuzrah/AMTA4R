# Writes the training config and the training/evaluation Slurm scripts of each configuration id and submits
# them (evaluation after training). Each config is MODEL_CONFIGS + BASE_TEMPLATE from submit_train_jobs.py,
# with the learning rate --lr and the batch layout --batch_mode. The paper's models use --lr 8e-5 --batch_mode repo.
# The default resources (--resources default) are those of App. K; --partition names the Slurm partition.
import argparse
import copy
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from submit_train_jobs import MODEL_CONFIGS, DATASETS, BASE_TEMPLATE  # noqa: E402

TRAIN_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/slurm/{job_name}_%j.out
#SBATCH --error=logs/slurm/{job_name}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --gres=gpu:a100:1
#SBATCH --time={time}
#SBATCH --partition={partition}

source .venv/bin/activate

export WANDB_MODE=offline
export PYTHONPATH=$PYTHONPATH:.
# Random port to prevent conflicts during DDP initialization
export MASTER_PORT=$(shuf -i 20000-60000 -n 1)

echo "Starting training for {job_name} on port $MASTER_PORT"
nvidia-smi --query-gpu=name,memory.total --format=csv
accelerate launch --num_processes 1 --main_process_port $MASTER_PORT src/train_flexible.py --config {config_path} --wandb
"""

EVAL_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=eval_{job_name}
#SBATCH --output=logs/eval_slurm/{job_name}_%j.out
#SBATCH --error=logs/eval_slurm/{job_name}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --gres=gpu:a100:1
#SBATCH --time={time}
#SBATCH --partition={partition}

source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
nvidia-smi --query-gpu=name,memory.total --format=csv
python src/eval_flexible.py --model_id {model_id} --checkpoints_dir {ckpt_dir} --data_dir data/experiment_500k --output_dir {out_dir}
test -f {out_dir}/{model_id}.json || {{ echo "EVAL RESULT MISSING for {model_id}"; exit 3; }}
cat {out_dir}/{model_id}.json
"""

RES = {  # resource profile -> (train cpus, mem, time), (eval cpus, mem, time)
    "default": (("16", "128G", "7-00:00:00"), ("24", "128G", "7-00:00:00")),
    "reduced": (("4", "24G", "5-00:00:00"), ("4", "24G", "5-00:00:00")),
}


def build_config(exp_id, lr, batch_mode, tag):
    size_shape, data_name, pe_type = exp_id.rsplit("_", 2)
    arch = MODEL_CONFIGS[size_shape]
    config = copy.deepcopy(BASE_TEMPLATE)
    config.update(arch)
    config["train_file"] = DATASETS[data_name]
    config["pe_type"] = pe_type
    config["arch_type"] = "gptneox" if pe_type == "rope" else "gptneo"
    config["name"] = f"{exp_id}__{tag}"
    config["output_dir"] = f"checkpoints/{tag}/{exp_id}"
    config["lr"] = lr
    heavy = "large" in size_shape or "medium" in size_shape
    if batch_mode == "paper" or heavy:
        # batch 4 x 8 accumulation steps for Medium/Large models (all models with --batch_mode paper)
        config["batch_size"] = 4
        config["gradient_accumulation_steps"] = 8
    else:  # Small models with --batch_mode repo: batch 32 (App. K)
        config["batch_size"] = 32
        config["gradient_accumulation_steps"] = 1
    config["gradient_checkpointing"] = bool(heavy)
    return config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_ids", nargs="+", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--batch_mode", choices=["repo", "paper"], required=True)
    ap.add_argument("--tag", required=True, help="sub-folder for configs/checkpoints/results")
    ap.add_argument("--partition", default="gpu", help="Slurm partition of the training jobs")
    ap.add_argument("--eval_partition", default=None, help="Slurm partition of the evaluation jobs (default: --partition)")
    ap.add_argument("--resources", default="default", choices=list(RES))
    ap.add_argument("--eval_resources", default=None, choices=list(RES))
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    (tc, tm, tt), _ = RES[a.resources]
    _, (ec, em, et) = RES[a.eval_resources or a.resources]
    for d in [f"configs/{a.tag}", f"scripts/sbatch_{a.tag}", "logs/slurm", "logs/eval_slurm", f"analysis/{a.tag}", "wandb_logs"]:
        os.makedirs(d, exist_ok=True)
    for exp_id in a.exp_ids:
        cfg = build_config(exp_id, a.lr, a.batch_mode, a.tag)
        cfg_path = os.path.abspath(f"configs/{a.tag}/{exp_id}.json")
        json.dump(cfg, open(cfg_path, "w"), indent=4)
        job = f"{exp_id}__{a.tag}"
        tr = f"scripts/sbatch_{a.tag}/{exp_id}.sh"
        ev = f"scripts/sbatch_{a.tag}/eval_{exp_id}.sh"
        open(tr, "w").write(TRAIN_TEMPLATE.format(job_name=job, config_path=cfg_path, cpus=tc, mem=tm, time=tt, partition=a.partition))
        open(ev, "w").write(EVAL_TEMPLATE.format(job_name=job, model_id=exp_id, ckpt_dir=f"checkpoints/{a.tag}",
                                                 out_dir=f"analysis/{a.tag}", cpus=ec, mem=em, time=et,
                                                 partition=a.eval_partition or a.partition))
        if a.dry_run:
            print("[dry-run]", tr, ev)
            continue
        jt = subprocess.run(["sbatch", "--parsable", tr], check=True, capture_output=True, text=True).stdout.strip()
        je = subprocess.run(["sbatch", "--parsable", f"--dependency=afterok:{jt}", ev], check=True, capture_output=True, text=True).stdout.strip()
        print(f"{exp_id} [{a.tag}] lr={a.lr} bs_mode={a.batch_mode}: train={jt} eval={je}")


if __name__ == "__main__":
    main()
