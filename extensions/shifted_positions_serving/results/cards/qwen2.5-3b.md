# Answers under shifted position indices: qwen2.5-3b

- Hugging Face Hub model `Qwen/Qwen2.5-3B`; license: Qwen Research License (hub tag "other"; non-commercial research use permitted)
- position class **RoPE** (qwen2), trained context 32768, 3.09 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.791 -> 0.791 | 0.0000 | 0.0000 | 0 |
| offset | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.791 -> 0.791 | 0.0000 | 0.0000 | 0 |
| offset | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.791 -> 0.791 | 0.0000 | 0.0000 | 0 |
| offset | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.791 -> 0.791 | 0.0000 | 0.0000 | 0 |
| leftpad | p16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.791 -> 0.791 | 0.0000 | 0.0000 | 0 |
| leftpad | p64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.791 -> 0.791 | 0.0000 | 0.0000 | 0 |
| leftpad | p256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.791 -> 0.791 | 0.0000 | 0.0000 | 0 |
| leftpad | p1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.791 -> 0.791 | 0.0000 | 0.0000 | 0.0001 |
| cache | block16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.791 -> 0.791 | 0.0000 | 0.0000 | 0 |
| cache | block64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.791 -> 0.791 | 0.0000 | 0.0000 | 0 |
| cache | block256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.791 -> 0.791 | 0.0000 | 0.0000 | 0 |
| cache | block1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.791 -> 0.791 | 0.0000 | 0.0000 | 0.0001 |
| cache | gap16 | degrades | -0.1397 [-0.1497, -0.1297] | 0.791 -> 0.651 | 0.3217 | 0.0743 | 7.15 |
| cache | gap64 | degrades | -0.1563 [-0.1667, -0.1457] | 0.791 -> 0.634 | 0.3603 | 0.0837 | 7.61 |
| cache | gap256 | degrades | -0.1823 [-0.1940, -0.1703] | 0.791 -> 0.608 | 0.4003 | 0.0877 | 8.05 |
| cache | gap1024 | degrades | -0.1937 [-0.2070, -0.1800] | 0.791 -> 0.597 | 0.4247 | 0.1057 | 8.1 |
| window | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.746 -> 0.746 | 0.0000 | 0.0000 | 0 |
| window | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.746 -> 0.746 | 0.0000 | 0.0000 | 0 |
| window | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.746 -> 0.746 | 0.0000 | 0.0000 | 0 |
| window | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.746 -> 0.746 | 0.0000 | 0.0000 | 0.0001 |
| offset_bf16 | k16 | unchanged | +0.0023 [-0.0013, +0.0063] | 0.788 -> 0.790 | 0.0313 | 0.0123 | 0.277 |
| offset_bf16 | k64 | unchanged | -0.0003 [-0.0040, +0.0033] | 0.788 -> 0.787 | 0.0323 | 0.0120 | 0.273 |
| offset_bf16 | k256 | unchanged | +0.0013 [-0.0027, +0.0053] | 0.788 -> 0.789 | 0.0360 | 0.0147 | 0.289 |
| offset_bf16 | k1024 | unchanged | +0.0000 [-0.0037, +0.0037] | 0.788 -> 0.788 | 0.0340 | 0.0127 | 0.312 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- left-pad with naive absolute position ids: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy unchanged at every tested k (max answer-flip rate 0.000)
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy unchanged at every tested k (max answer-flip rate 0.000)
