# Answers under shifted position indices: gpt-neo-2.7B

- Hugging Face Hub model `EleutherAI/gpt-neo-2.7B`; license: MIT
- position class **APE** (gpt_neo), trained context 2048, 2.72 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | degrades | -0.0787 [-0.0930, -0.0640] | 0.625 -> 0.546 | 0.4120 | 0.2537 | 4.72 |
| offset | k64 | degrades | -0.2177 [-0.2350, -0.2003] | 0.625 -> 0.407 | 0.5950 | 0.4053 | 7.59 |
| offset | k256 | degrades | -0.4800 [-0.4963, -0.4637] | 0.625 -> 0.145 | 0.8570 | 0.8130 | 17.3 |
| offset | k1024 | degrades | -0.5680 [-0.5860, -0.5504] | 0.586 -> 0.018 | 0.9792 | 0.9620 | 19.2 |
| leftpad | p16 | degrades | -0.0787 [-0.0933, -0.0643] | 0.625 -> 0.546 | 0.4120 | 0.2537 | 4.72 |
| leftpad | p64 | degrades | -0.2177 [-0.2350, -0.2010] | 0.625 -> 0.407 | 0.5950 | 0.4053 | 7.59 |
| leftpad | p256 | degrades | -0.4800 [-0.4967, -0.4640] | 0.625 -> 0.145 | 0.8570 | 0.8130 | 17.3 |
| leftpad | p1024 | degrades | -0.5680 [-0.5860, -0.5504] | 0.586 -> 0.018 | 0.9792 | 0.9620 | 19.2 |
| cache | block16 | degrades | -0.0787 [-0.0933, -0.0643] | 0.625 -> 0.546 | 0.4120 | 0.2537 | 4.72 |
| cache | block64 | degrades | -0.2177 [-0.2353, -0.2003] | 0.625 -> 0.407 | 0.5950 | 0.4053 | 7.59 |
| cache | block256 | degrades | -0.4800 [-0.4967, -0.4637] | 0.625 -> 0.145 | 0.8570 | 0.8130 | 17.3 |
| cache | block1024 | degrades | -0.5680 [-0.5856, -0.5508] | 0.586 -> 0.018 | 0.9792 | 0.9620 | 19.2 |
| cache | gap16 | unchanged | +0.0023 [-0.0023, +0.0070] | 0.625 -> 0.627 | 0.0350 | 0.0113 | 0.197 |
| cache | gap64 | unchanged | -0.0040 [-0.0123, +0.0047] | 0.625 -> 0.621 | 0.1553 | 0.0563 | 0.676 |
| cache | gap256 | degrades | -0.4340 [-0.4507, -0.4170] | 0.625 -> 0.191 | 0.7800 | 0.6050 | 9.56 |
| cache | gap1024 | degrades | -0.5804 [-0.5976, -0.5632] | 0.586 -> 0.006 | 0.9564 | 0.7860 | 12.7 |
| window | k16 | degrades | -0.0892 [-0.1064, -0.0720] | 0.376 -> 0.287 | 0.6040 | 0.2964 | 5.67 |
| window | k64 | degrades | -0.0864 [-0.1064, -0.0664] | 0.376 -> 0.290 | 0.8068 | 0.4404 | 8.51 |
| window | k256 | degrades | -0.2640 [-0.2836, -0.2452] | 0.376 -> 0.112 | 0.9520 | 0.8000 | 14.6 |
| window | k1024 | degrades | -0.3480 [-0.3660, -0.3300] | 0.376 -> 0.028 | 0.9736 | 0.9108 | 15.9 |
| offset_bf16 | k16 | degrades | -0.0353 [-0.0533, -0.0173] | 0.502 -> 0.467 | 0.5560 | 0.3420 | 4.62 |
| offset_bf16 | k64 | degrades | -0.1473 [-0.1667, -0.1287] | 0.502 -> 0.355 | 0.6863 | 0.4673 | 6.75 |
| offset_bf16 | k256 | degrades | -0.3857 [-0.4037, -0.3677] | 0.502 -> 0.117 | 0.9120 | 0.8537 | 16.5 |
| offset_bf16 | k1024 | degrades | -0.4836 [-0.5020, -0.4652] | 0.492 -> 0.008 | 0.9880 | 0.9688 | 17.6 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy changes at k16, k64, k256, k1024 -> recompute/reset positions
- left-pad with naive absolute position ids: accuracy changes at p16, p64, p256, p1024 -> recompute/reset positions
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy changes at block16, block64, block256, block1024 -> recompute/reset positions
- append to a cached prefix with a position gap: accuracy changes at gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy changes at k16, k64, k256, k1024 -> recompute/reset positions
