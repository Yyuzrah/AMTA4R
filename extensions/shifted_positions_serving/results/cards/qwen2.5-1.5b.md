# Answers under shifted position indices: qwen2.5-1.5b

- Hugging Face Hub model `Qwen/Qwen2.5-1.5B`; license: Apache-2.0
- position class **RoPE** (qwen2), trained context 32768, 1.54 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | unchanged | +0.0003 [+0.0000, +0.0010] | 0.683 -> 0.684 | 0.0003 | 0.0003 | 0 |
| offset | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.683 -> 0.683 | 0.0000 | 0.0000 | 0 |
| offset | k256 | unchanged | +0.0003 [+0.0000, +0.0010] | 0.683 -> 0.684 | 0.0003 | 0.0003 | 0 |
| offset | k1024 | unchanged | +0.0003 [+0.0000, +0.0010] | 0.683 -> 0.684 | 0.0003 | 0.0003 | 0.0001 |
| leftpad | p16 | unchanged | +0.0003 [+0.0000, +0.0010] | 0.683 -> 0.684 | 0.0003 | 0.0003 | 0 |
| leftpad | p64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.683 -> 0.683 | 0.0000 | 0.0000 | 0 |
| leftpad | p256 | unchanged | +0.0003 [+0.0000, +0.0010] | 0.683 -> 0.684 | 0.0003 | 0.0003 | 0 |
| leftpad | p1024 | unchanged | +0.0003 [+0.0000, +0.0010] | 0.683 -> 0.684 | 0.0003 | 0.0003 | 0.0001 |
| cache | block16 | unchanged | +0.0003 [+0.0000, +0.0010] | 0.683 -> 0.684 | 0.0003 | 0.0003 | 0 |
| cache | block64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.683 -> 0.683 | 0.0000 | 0.0000 | 0 |
| cache | block256 | unchanged | +0.0003 [+0.0000, +0.0010] | 0.683 -> 0.684 | 0.0003 | 0.0003 | 0 |
| cache | block1024 | unchanged | +0.0003 [+0.0000, +0.0010] | 0.683 -> 0.684 | 0.0003 | 0.0003 | 0.0001 |
| cache | gap16 | degrades | -0.0860 [-0.0987, -0.0733] | 0.683 -> 0.597 | 0.3903 | 0.1393 | 5.02 |
| cache | gap64 | degrades | -0.1320 [-0.1443, -0.1193] | 0.683 -> 0.551 | 0.4510 | 0.1520 | 5.03 |
| cache | gap256 | degrades | -0.1690 [-0.1827, -0.1557] | 0.683 -> 0.514 | 0.4993 | 0.1640 | 5.05 |
| cache | gap1024 | degrades | -0.1693 [-0.1823, -0.1560] | 0.683 -> 0.514 | 0.5077 | 0.1620 | 5.65 |
| window | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.692 -> 0.692 | 0.0000 | 0.0000 | 0 |
| window | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.692 -> 0.692 | 0.0000 | 0.0000 | 0 |
| window | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.692 -> 0.692 | 0.0000 | 0.0000 | 0 |
| window | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.692 -> 0.692 | 0.0000 | 0.0000 | 0.0001 |
| offset_bf16 | k16 | unchanged | -0.0020 [-0.0077, +0.0037] | 0.598 -> 0.596 | 0.0923 | 0.0503 | 0.312 |
| offset_bf16 | k64 | unchanged | +0.0017 [-0.0047, +0.0080] | 0.598 -> 0.600 | 0.1150 | 0.0633 | 0.346 |
| offset_bf16 | k256 | degrades | -0.0103 [-0.0187, -0.0020] | 0.598 -> 0.588 | 0.1733 | 0.0893 | 0.781 |
| offset_bf16 | k1024 | unchanged | +0.0043 [-0.0047, +0.0133] | 0.598 -> 0.602 | 0.1983 | 0.0970 | 0.971 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- left-pad with naive absolute position ids: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy unchanged at every tested k (max answer-flip rate 0.000)
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy unchanged at every tested k (max answer-flip rate 0.000)
