# Answers under shifted position indices: pythia-160m

- Hugging Face Hub model `EleutherAI/pythia-160m`; license: Apache-2.0
- position class **RoPE** (gpt_neox), trained context 2048, 0.162 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.209 -> 0.209 | 0.0003 | 0.0003 | 0.0022 |
| offset | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.209 -> 0.209 | 0.0003 | 0.0003 | 0.0023 |
| offset | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.209 -> 0.209 | 0.0000 | 0.0000 | 0.0022 |
| offset | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.226 -> 0.226 | 0.0000 | 0.0000 | 0.002 |
| leftpad | p16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.209 -> 0.209 | 0.0003 | 0.0003 | 0.0022 |
| leftpad | p64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.209 -> 0.209 | 0.0000 | 0.0000 | 0.0023 |
| leftpad | p256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.209 -> 0.209 | 0.0000 | 0.0000 | 0.0023 |
| leftpad | p1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.226 -> 0.226 | 0.0004 | 0.0004 | 0.002 |
| cache | block16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.209 -> 0.209 | 0.0000 | 0.0000 | 0.0017 |
| cache | block64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.209 -> 0.209 | 0.0000 | 0.0000 | 0.0017 |
| cache | block256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.209 -> 0.209 | 0.0000 | 0.0000 | 0.0017 |
| cache | block1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.226 -> 0.226 | 0.0004 | 0.0004 | 0.0017 |
| cache | gap16 | unchanged | -0.0057 [-0.0167, +0.0057] | 0.209 -> 0.203 | 0.5640 | 0.3370 | 3.24 |
| cache | gap64 | degrades | -0.0193 [-0.0313, -0.0077] | 0.209 -> 0.189 | 0.6493 | 0.4013 | 4.54 |
| cache | gap256 | degrades | -0.0273 [-0.0397, -0.0150] | 0.209 -> 0.181 | 0.6960 | 0.4627 | 4.89 |
| cache | gap1024 | degrades | -0.0320 [-0.0468, -0.0176] | 0.226 -> 0.194 | 0.7140 | 0.5012 | 5.36 |
| window | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.244 -> 0.244 | 0.0004 | 0.0000 | 0.002 |
| window | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.244 -> 0.244 | 0.0004 | 0.0004 | 0.002 |
| window | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.244 -> 0.244 | 0.0004 | 0.0000 | 0.0019 |
| window | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.244 -> 0.244 | 0.0004 | 0.0000 | 0.002 |
| offset_bf16 | k16 | unchanged | -0.0017 [-0.0060, +0.0027] | 0.012 -> 0.011 | 0.6770 | 0.4793 | 4 |
| offset_bf16 | k64 | unchanged | -0.0037 [-0.0080, +0.0003] | 0.012 -> 0.009 | 0.6813 | 0.4653 | 4 |
| offset_bf16 | k256 | unchanged | +0.0000 [-0.0043, +0.0043] | 0.012 -> 0.012 | 0.6803 | 0.4730 | 4 |
| offset_bf16 | k1024 | unchanged | -0.0040 [-0.0092, +0.0012] | 0.014 -> 0.010 | 0.7536 | 0.5060 | 4 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- left-pad with naive absolute position ids: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy unchanged at every tested k (max answer-flip rate 0.000)
- append to a cached prefix with a position gap: accuracy changes at gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy unchanged at every tested k (max answer-flip rate 0.000)
