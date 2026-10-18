# Answers under shifted position indices: opt-2.7b

- Hugging Face Hub model `facebook/opt-2.7b`; license: OPT-175B License Agreement (hub tag "other"; non-commercial research use permitted)
- position class **APE** (opt), trained context 2048, 2.65 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | degrades | -0.0307 [-0.0450, -0.0163] | 0.432 -> 0.401 | 0.4357 | 0.2313 | 2.5 |
| offset | k64 | degrades | -0.3227 [-0.3390, -0.3060] | 0.432 -> 0.109 | 0.8313 | 0.6833 | 5.84 |
| offset | k256 | degrades | -0.4277 [-0.4440, -0.4117] | 0.432 -> 0.004 | 0.9913 | 0.9817 | 10.3 |
| offset | k1024 | degrades | -0.4476 [-0.4660, -0.4300] | 0.452 -> 0.005 | 0.9916 | 0.9708 | 11.1 |
| leftpad | p16 | degrades | -0.0307 [-0.0450, -0.0167] | 0.432 -> 0.401 | 0.4357 | 0.2313 | 2.5 |
| leftpad | p64 | degrades | -0.3227 [-0.3393, -0.3067] | 0.432 -> 0.109 | 0.8313 | 0.6833 | 5.84 |
| leftpad | p256 | degrades | -0.4277 [-0.4437, -0.4113] | 0.432 -> 0.004 | 0.9913 | 0.9817 | 10.3 |
| leftpad | p1024 | degrades | -0.4476 [-0.4652, -0.4300] | 0.452 -> 0.005 | 0.9916 | 0.9708 | 11.1 |
| cache | block16 | degrades | -0.0307 [-0.0450, -0.0163] | 0.432 -> 0.401 | 0.4357 | 0.2313 | 2.5 |
| cache | block64 | degrades | -0.3227 [-0.3390, -0.3067] | 0.432 -> 0.109 | 0.8313 | 0.6833 | 5.84 |
| cache | block256 | degrades | -0.4277 [-0.4440, -0.4113] | 0.432 -> 0.004 | 0.9913 | 0.9817 | 10.3 |
| cache | block1024 | degrades | -0.4476 [-0.4648, -0.4300] | 0.452 -> 0.005 | 0.9916 | 0.9708 | 11.1 |
| cache | gap16 | degrades | -0.0077 [-0.0147, -0.0007] | 0.432 -> 0.424 | 0.0813 | 0.0473 | 0.195 |
| cache | gap64 | degrades | -0.0130 [-0.0203, -0.0057] | 0.432 -> 0.419 | 0.0947 | 0.0507 | 0.237 |
| cache | gap256 | degrades | -0.0153 [-0.0247, -0.0063] | 0.432 -> 0.417 | 0.1670 | 0.0733 | 0.402 |
| cache | gap1024 | unchanged | -0.0096 [-0.0216, +0.0024] | 0.452 -> 0.443 | 0.2828 | 0.1224 | 0.684 |
| window | k16 | degrades | -0.1564 [-0.1732, -0.1400] | 0.336 -> 0.179 | 0.6556 | 0.4224 | 3.47 |
| window | k64 | degrades | -0.2616 [-0.2796, -0.2432] | 0.336 -> 0.074 | 0.8520 | 0.6728 | 5.51 |
| window | k256 | degrades | -0.3304 [-0.3484, -0.3124] | 0.336 -> 0.005 | 0.9856 | 0.9740 | 10.6 |
| window | k1024 | degrades | -0.3308 [-0.3492, -0.3128] | 0.336 -> 0.005 | 0.9912 | 0.9712 | 11.1 |
| offset_bf16 | k16 | degrades | -0.0160 [-0.0303, -0.0013] | 0.421 -> 0.405 | 0.4500 | 0.2423 | 2.5 |
| offset_bf16 | k64 | degrades | -0.3123 [-0.3287, -0.2957] | 0.421 -> 0.108 | 0.8323 | 0.6913 | 6.19 |
| offset_bf16 | k256 | degrades | -0.4160 [-0.4323, -0.3997] | 0.421 -> 0.005 | 0.9907 | 0.9777 | 10.3 |
| offset_bf16 | k1024 | degrades | -0.4344 [-0.4520, -0.4168] | 0.440 -> 0.005 | 0.9900 | 0.9684 | 11.2 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy changes at k16, k64, k256, k1024 -> recompute/reset positions
- left-pad with naive absolute position ids: accuracy changes at p16, p64, p256, p1024 -> recompute/reset positions
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy changes at block16, block64, block256, block1024 -> recompute/reset positions
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy changes at k16, k64, k256, k1024 -> recompute/reset positions
