# Answers under shifted position indices: gpt2-xl

- Hugging Face Hub model `gpt2-xl`; license: MIT
- position class **APE** (gpt2), trained context 1024, 1.56 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | degrades | -0.3140 [-0.3307, -0.2973] | 0.416 -> 0.102 | 0.8870 | 0.6703 | 10.7 |
| offset | k64 | degrades | -0.4157 [-0.4303, -0.4010] | 0.416 -> 0.001 | 0.9983 | 0.9523 | 13.8 |
| offset | k256 | degrades | -0.4163 [-0.4313, -0.4017] | 0.416 -> 0.000 | 1.0000 | 1.0000 | 15.7 |
| leftpad | p16 | degrades | -0.3140 [-0.3307, -0.2977] | 0.416 -> 0.102 | 0.8870 | 0.6703 | 10.7 |
| leftpad | p64 | degrades | -0.4157 [-0.4303, -0.4007] | 0.416 -> 0.001 | 0.9983 | 0.9523 | 13.8 |
| leftpad | p256 | degrades | -0.4163 [-0.4313, -0.4013] | 0.416 -> 0.000 | 1.0000 | 1.0000 | 15.7 |
| cache | block16 | degrades | -0.3140 [-0.3307, -0.2973] | 0.416 -> 0.102 | 0.8870 | 0.6703 | 10.7 |
| cache | block64 | degrades | -0.4157 [-0.4307, -0.4007] | 0.416 -> 0.001 | 0.9983 | 0.9523 | 13.8 |
| cache | block256 | degrades | -0.4163 [-0.4317, -0.4017] | 0.416 -> 0.000 | 1.0000 | 1.0000 | 15.7 |
| cache | gap16 | unchanged | -0.0010 [-0.0033, +0.0013] | 0.416 -> 0.415 | 0.0270 | 0.0143 | 0.104 |
| cache | gap64 | degrades | -0.0147 [-0.0217, -0.0077] | 0.416 -> 0.402 | 0.1807 | 0.0627 | 0.485 |
| cache | gap256 | degrades | -0.0377 [-0.0457, -0.0300] | 0.416 -> 0.379 | 0.1507 | 0.0457 | 0.351 |
| window | k16 | degrades | -0.1852 [-0.2000, -0.1712] | 0.189 -> 0.004 | 0.8768 | 0.7028 | 12.6 |
| window | k64 | degrades | -0.1884 [-0.2032, -0.1740] | 0.189 -> 0.000 | 0.9504 | 0.8760 | 12.1 |
| window | k256 | degrades | -0.1888 [-0.2032, -0.1740] | 0.189 -> 0.000 | 1.0000 | 1.0000 | 14.4 |
| offset_bf16 | k16 | degrades | -0.3130 [-0.3297, -0.2963] | 0.417 -> 0.104 | 0.8870 | 0.6707 | 10.7 |
| offset_bf16 | k64 | degrades | -0.4157 [-0.4307, -0.4010] | 0.417 -> 0.002 | 0.9977 | 0.9520 | 13.8 |
| offset_bf16 | k256 | degrades | -0.4173 [-0.4323, -0.4023] | 0.417 -> 0.000 | 1.0000 | 1.0000 | 15.7 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy changes at k16, k64, k256 -> recompute/reset positions
- left-pad with naive absolute position ids: accuracy changes at p16, p64, p256 -> recompute/reset positions
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy changes at block16, block64, block256 -> recompute/reset positions
- append to a cached prefix with a position gap: accuracy changes at gap64, gap256 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy changes at k16, k64, k256 -> recompute/reset positions
