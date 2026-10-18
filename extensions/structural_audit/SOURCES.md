# Data sources (App. M.2)

`scripts/download_data.py` downloads the four public datasets below into `data/` (about 4.6 GB, not part of the
repository); `scripts/gen_babyai.py` generates the BabyAI levels. The released `features/` and `results/` contain only
derived quantities: per-trajectory coordinates, group labels, unit identifiers (trajectory ids, issue ids, Bridge
episode keys, BabyAI seeds and missions) and statistics computed from them. No text of the trajectories is included:
the language instruction of each Bridge episode, which `scripts/feat_vla.py` writes to the column `instr` next to the
skill read from it, is removed from the released `features/vla_features.csv` (the analysis uses only the skill).
In the same file, `uid` is the ECoT episode key without its leading folder (an absolute path on the machine where the
annotations were made): the path of the episode file inside the Bridge data and the episode index, for example
`bridge_data_v2/deepthought_folding_table/stack_blocks/19/train/out.npy#43`.

| Family | Source | License | Derived file |
|---|---|---|---|
| A, B, B2 | ETO expert trajectories in ALFWorld, ScienceWorld and WebShop, [agent-eto/eto-sft-trajectory](https://huggingface.co/datasets/agent-eto/eto-sft-trajectory) (Song et al., 2024) | Apache-2.0 | `features/eto_features.csv` |
| C | SWE-agent runs, [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) (Nebius, 2024) | CC-BY-4.0 | `features/harness_features.csv` |
| C | OpenHands runs of Qwen3-Coder-480B-A35B-Instruct on SWE-rebench issues, [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) (Trofimova et al., 2025) | CC-BY-4.0 | `features/harness_features.csv` |
| D | ECoT features of BridgeData V2 demonstrations, [Embodied-CoT/embodied_features_bridge](https://huggingface.co/datasets/Embodied-CoT/embodied_features_bridge) (Zawalski et al., 2025); BridgeData V2 (Walke et al., 2023) | MIT; BridgeData V2: CC-BY-4.0 | `features/vla_features.csv` |
| E | BabyAI-MixedTrainLocal-v0 levels generated with the [Minigrid](https://github.com/Farama-Foundation/Minigrid) fork used by BALROG, [BartekCupial/Minigrid](https://github.com/BartekCupial/Minigrid) (version 2.3.1), and the trajectories of its BabyAI bot, which originates from [mila-iqia/babyai](https://github.com/mila-iqia/babyai) | Apache-2.0; BabyAI: BSD-3-Clause | `features/babyai_*.csv` |

The derived files are computed from these datasets by the scripts in `scripts/`.
