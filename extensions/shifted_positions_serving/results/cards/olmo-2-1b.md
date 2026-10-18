# Answers under shifted position indices: olmo-2-1b

- Hugging Face Hub model `allenai/OLMo-2-0425-1B`; license: Apache-2.0
- position class **RoPE** (olmo2), trained context 4096, 1.48 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.556 -> 0.556 | 0.0003 | 0.0003 | 0 |
| offset | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.556 -> 0.556 | 0.0000 | 0.0000 | 0 |
| offset | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.556 -> 0.556 | 0.0000 | 0.0000 | 0 |
| offset | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.556 -> 0.556 | 0.0003 | 0.0003 | 0.0001 |
| leftpad | p16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.556 -> 0.556 | 0.0003 | 0.0003 | 0.0001 |
| leftpad | p64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.556 -> 0.556 | 0.0000 | 0.0000 | 0.0001 |
| leftpad | p256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.556 -> 0.556 | 0.0000 | 0.0000 | 0.0001 |
| leftpad | p1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.556 -> 0.556 | 0.0003 | 0.0003 | 0.0001 |
| cache | block16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.556 -> 0.556 | 0.0000 | 0.0000 | 0.0001 |
| cache | block64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.556 -> 0.556 | 0.0003 | 0.0003 | 0.0001 |
| cache | block256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.556 -> 0.556 | 0.0000 | 0.0000 | 0.0001 |
| cache | block1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.556 -> 0.556 | 0.0000 | 0.0000 | 0.0001 |
| cache | gap16 | degrades | -0.0223 [-0.0357, -0.0087] | 0.556 -> 0.534 | 0.3290 | 0.1320 | 5 |
| cache | gap64 | degrades | -0.0517 [-0.0667, -0.0367] | 0.556 -> 0.504 | 0.4213 | 0.1440 | 5.15 |
| cache | gap256 | degrades | -0.1403 [-0.1567, -0.1233] | 0.556 -> 0.416 | 0.5427 | 0.1927 | 8.73 |
| cache | gap1024 | degrades | -0.2747 [-0.2923, -0.2577] | 0.556 -> 0.281 | 0.6933 | 0.3713 | 12 |
| window | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.564 -> 0.564 | 0.0000 | 0.0000 | 0 |
| window | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.564 -> 0.564 | 0.0000 | 0.0000 | 0 |
| window | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.564 -> 0.564 | 0.0000 | 0.0000 | 0 |
| window | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.564 -> 0.564 | 0.0000 | 0.0000 | 0.0001 |
| offset_bf16 | k16 | unchanged | +0.0007 [-0.0033, +0.0047] | 0.558 -> 0.559 | 0.0383 | 0.0133 | 0.25 |
| offset_bf16 | k64 | unchanged | -0.0020 [-0.0060, +0.0020] | 0.558 -> 0.556 | 0.0413 | 0.0140 | 0.262 |
| offset_bf16 | k256 | unchanged | -0.0010 [-0.0050, +0.0030] | 0.558 -> 0.557 | 0.0387 | 0.0117 | 0.297 |
| offset_bf16 | k1024 | unchanged | -0.0030 [-0.0070, +0.0010] | 0.558 -> 0.555 | 0.0430 | 0.0157 | 0.344 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- left-pad with naive absolute position ids: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy unchanged at every tested k (max answer-flip rate 0.000)
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy unchanged at every tested k (max answer-flip rate 0.000)
