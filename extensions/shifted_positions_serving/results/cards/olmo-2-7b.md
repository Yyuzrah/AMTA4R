# Answers under shifted position indices: olmo-2-7b

- Hugging Face Hub model `allenai/OLMo-2-1124-7B`; license: Apache-2.0
- position class **RoPE** (olmo2), trained context 4096, 7.3 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.675 -> 0.675 | 0.0000 | 0.0000 | 0 |
| offset | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.675 -> 0.675 | 0.0000 | 0.0000 | 0 |
| offset | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.675 -> 0.675 | 0.0000 | 0.0000 | 0 |
| offset | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.675 -> 0.675 | 0.0000 | 0.0000 | 0.0001 |
| leftpad | p16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.675 -> 0.675 | 0.0000 | 0.0000 | 0 |
| leftpad | p64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.675 -> 0.675 | 0.0000 | 0.0000 | 0 |
| leftpad | p256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.675 -> 0.675 | 0.0000 | 0.0000 | 0 |
| leftpad | p1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.675 -> 0.675 | 0.0000 | 0.0000 | 0.0001 |
| cache | block16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.675 -> 0.675 | 0.0000 | 0.0000 | 0 |
| cache | block64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.675 -> 0.675 | 0.0000 | 0.0000 | 0 |
| cache | block256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.675 -> 0.675 | 0.0000 | 0.0000 | 0 |
| cache | block1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.675 -> 0.675 | 0.0000 | 0.0000 | 0.0001 |
| cache | gap16 | improves | +0.0177 [+0.0053, +0.0300] | 0.675 -> 0.693 | 0.3620 | 0.1847 | 8.1 |
| cache | gap64 | degrades | -0.0317 [-0.0460, -0.0180] | 0.675 -> 0.644 | 0.4247 | 0.1920 | 8.56 |
| cache | gap256 | degrades | -0.1293 [-0.1447, -0.1140] | 0.675 -> 0.546 | 0.4840 | 0.2057 | 8.69 |
| cache | gap1024 | degrades | -0.2360 [-0.2533, -0.2190] | 0.675 -> 0.439 | 0.6127 | 0.2520 | 12.8 |
| window | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.731 -> 0.731 | 0.0000 | 0.0000 | 0 |
| window | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.731 -> 0.731 | 0.0000 | 0.0000 | 0 |
| window | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.731 -> 0.731 | 0.0000 | 0.0000 | 0 |
| window | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.731 -> 0.731 | 0.0000 | 0.0000 | 0.0001 |
| offset_bf16 | k16 | unchanged | -0.0003 [-0.0030, +0.0020] | 0.674 -> 0.674 | 0.0230 | 0.0107 | 0.25 |
| offset_bf16 | k64 | unchanged | +0.0020 [-0.0007, +0.0050] | 0.674 -> 0.676 | 0.0270 | 0.0117 | 0.25 |
| offset_bf16 | k256 | unchanged | +0.0003 [-0.0023, +0.0030] | 0.674 -> 0.674 | 0.0253 | 0.0100 | 0.25 |
| offset_bf16 | k1024 | unchanged | +0.0000 [-0.0027, +0.0030] | 0.674 -> 0.674 | 0.0233 | 0.0110 | 0.25 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- left-pad with naive absolute position ids: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy unchanged at every tested k (max answer-flip rate 0.000)
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy unchanged at every tested k (max answer-flip rate 0.000)
