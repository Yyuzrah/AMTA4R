# Answers under shifted position indices: gpt-neo-125m

- Hugging Face Hub model `EleutherAI/gpt-neo-125m`; license: MIT
- position class **APE** (gpt_neo), trained context 2048, 0.125 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | degrades | -0.0530 [-0.0643, -0.0410] | 0.276 -> 0.223 | 0.5330 | 0.3323 | 6.55 |
| offset | k64 | degrades | -0.0803 [-0.0933, -0.0677] | 0.276 -> 0.195 | 0.6220 | 0.3963 | 7.2 |
| offset | k256 | degrades | -0.1360 [-0.1497, -0.1227] | 0.276 -> 0.140 | 0.7053 | 0.4147 | 7.71 |
| offset | k1024 | degrades | -0.2216 [-0.2380, -0.2052] | 0.311 -> 0.090 | 0.8436 | 0.5356 | 11.9 |
| leftpad | p16 | degrades | -0.0530 [-0.0650, -0.0410] | 0.276 -> 0.223 | 0.5330 | 0.3323 | 6.55 |
| leftpad | p64 | degrades | -0.0803 [-0.0933, -0.0677] | 0.276 -> 0.195 | 0.6220 | 0.3963 | 7.2 |
| leftpad | p256 | degrades | -0.1360 [-0.1497, -0.1227] | 0.276 -> 0.140 | 0.7053 | 0.4147 | 7.71 |
| leftpad | p1024 | degrades | -0.2216 [-0.2384, -0.2048] | 0.311 -> 0.090 | 0.8436 | 0.5356 | 11.9 |
| cache | block16 | degrades | -0.0530 [-0.0647, -0.0413] | 0.276 -> 0.223 | 0.5330 | 0.3323 | 6.55 |
| cache | block64 | degrades | -0.0803 [-0.0933, -0.0673] | 0.276 -> 0.195 | 0.6220 | 0.3963 | 7.2 |
| cache | block256 | degrades | -0.1360 [-0.1497, -0.1223] | 0.276 -> 0.140 | 0.7053 | 0.4147 | 7.71 |
| cache | block1024 | degrades | -0.2216 [-0.2380, -0.2052] | 0.311 -> 0.090 | 0.8436 | 0.5356 | 11.9 |
| cache | gap16 | degrades | -0.0190 [-0.0273, -0.0107] | 0.276 -> 0.257 | 0.1900 | 0.0737 | 0.888 |
| cache | gap64 | degrades | -0.1973 [-0.2107, -0.1840] | 0.276 -> 0.078 | 0.7480 | 0.5083 | 8.82 |
| cache | gap256 | degrades | -0.2633 [-0.2767, -0.2500] | 0.276 -> 0.012 | 0.9083 | 0.7447 | 16.8 |
| cache | gap1024 | degrades | -0.3064 [-0.3216, -0.2908] | 0.311 -> 0.005 | 0.9944 | 0.8236 | 15 |
| window | k16 | degrades | -0.0528 [-0.0668, -0.0392] | 0.190 -> 0.138 | 0.6072 | 0.3064 | 7.2 |
| window | k64 | degrades | -0.0228 [-0.0376, -0.0080] | 0.190 -> 0.168 | 0.7100 | 0.3692 | 7.97 |
| window | k256 | degrades | -0.0412 [-0.0564, -0.0260] | 0.190 -> 0.149 | 0.7468 | 0.3700 | 9.49 |
| window | k1024 | degrades | -0.0520 [-0.0688, -0.0352] | 0.190 -> 0.138 | 0.8472 | 0.5280 | 11.9 |
| offset_bf16 | k16 | degrades | -0.0537 [-0.0660, -0.0413] | 0.261 -> 0.207 | 0.5617 | 0.3373 | 6.7 |
| offset_bf16 | k64 | degrades | -0.0747 [-0.0883, -0.0607] | 0.261 -> 0.186 | 0.6443 | 0.4100 | 7.34 |
| offset_bf16 | k256 | degrades | -0.1410 [-0.1547, -0.1273] | 0.261 -> 0.120 | 0.7313 | 0.4263 | 7.94 |
| offset_bf16 | k1024 | degrades | -0.2188 [-0.2356, -0.2020] | 0.293 -> 0.074 | 0.8548 | 0.5380 | 11.8 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy changes at k16, k64, k256, k1024 -> recompute/reset positions
- left-pad with naive absolute position ids: accuracy changes at p16, p64, p256, p1024 -> recompute/reset positions
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy changes at block16, block64, block256, block1024 -> recompute/reset positions
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy changes at k16, k64, k256, k1024 -> recompute/reset positions
