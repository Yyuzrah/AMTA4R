# Answers under shifted position indices: gpt2-medium

- Hugging Face Hub model `gpt2-medium`; license: MIT
- position class **APE** (gpt2), trained context 1024, 0.355 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | degrades | -0.3300 [-0.3433, -0.3170] | 0.336 -> 0.006 | 0.9820 | 0.8740 | 237 |
| offset | k64 | degrades | -0.3350 [-0.3483, -0.3220] | 0.336 -> 0.001 | 0.9940 | 0.9700 | 248 |
| offset | k256 | degrades | -0.3360 [-0.3493, -0.3227] | 0.336 -> 0.000 | 0.9920 | 0.9827 | 229 |
| leftpad | p16 | degrades | -0.3300 [-0.3433, -0.3170] | 0.336 -> 0.006 | 0.9820 | 0.8740 | 237 |
| leftpad | p64 | degrades | -0.3350 [-0.3483, -0.3220] | 0.336 -> 0.001 | 0.9940 | 0.9700 | 248 |
| leftpad | p256 | degrades | -0.3360 [-0.3493, -0.3223] | 0.336 -> 0.000 | 0.9920 | 0.9827 | 229 |
| cache | block16 | degrades | -0.3300 [-0.3430, -0.3167] | 0.336 -> 0.006 | 0.9820 | 0.8740 | 237 |
| cache | block64 | degrades | -0.3350 [-0.3480, -0.3223] | 0.336 -> 0.001 | 0.9940 | 0.9700 | 248 |
| cache | block256 | degrades | -0.3360 [-0.3493, -0.3230] | 0.336 -> 0.000 | 0.9920 | 0.9827 | 229 |
| cache | gap16 | unchanged | +0.0003 [-0.0033, +0.0037] | 0.336 -> 0.337 | 0.0370 | 0.0127 | 0.709 |
| cache | gap64 | degrades | -0.0290 [-0.0377, -0.0200] | 0.336 -> 0.307 | 0.2153 | 0.0627 | 3.44 |
| cache | gap256 | degrades | -0.2747 [-0.2867, -0.2627] | 0.336 -> 0.062 | 0.7170 | 0.5627 | 45.7 |
| window | k16 | degrades | -0.1260 [-0.1384, -0.1136] | 0.132 -> 0.006 | 0.9592 | 0.8620 | 292 |
| window | k64 | degrades | -0.1308 [-0.1432, -0.1184] | 0.132 -> 0.002 | 0.8912 | 0.8632 | 270 |
| window | k256 | degrades | -0.1320 [-0.1444, -0.1196] | 0.132 -> 0.000 | 0.9900 | 0.9800 | 222 |
| offset_bf16 | k16 | degrades | -0.3247 [-0.3377, -0.3120] | 0.331 -> 0.006 | 0.9850 | 0.8837 | 237 |
| offset_bf16 | k64 | degrades | -0.3297 [-0.3427, -0.3167] | 0.331 -> 0.001 | 0.9947 | 0.9740 | 250 |
| offset_bf16 | k256 | degrades | -0.3300 [-0.3430, -0.3170] | 0.331 -> 0.001 | 0.9883 | 0.9833 | 233 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy changes at k16, k64, k256 -> recompute/reset positions
- left-pad with naive absolute position ids: accuracy changes at p16, p64, p256 -> recompute/reset positions
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy changes at block16, block64, block256 -> recompute/reset positions
- append to a cached prefix with a position gap: accuracy changes at gap64, gap256 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy changes at k16, k64, k256 -> recompute/reset positions
