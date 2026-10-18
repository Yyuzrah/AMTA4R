# Answers under shifted position indices: gpt2-large

- Hugging Face Hub model `gpt2-large`; license: MIT
- position class **APE** (gpt2), trained context 1024, 0.774 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | degrades | -0.3353 [-0.3497, -0.3217] | 0.359 -> 0.024 | 0.9667 | 0.8247 | 11.5 |
| offset | k64 | degrades | -0.3537 [-0.3677, -0.3400] | 0.359 -> 0.005 | 0.9887 | 0.9207 | 13.3 |
| offset | k256 | degrades | -0.3590 [-0.3727, -0.3453] | 0.359 -> 0.000 | 1.0000 | 0.9983 | 13.2 |
| leftpad | p16 | degrades | -0.3353 [-0.3493, -0.3217] | 0.359 -> 0.024 | 0.9667 | 0.8247 | 11.5 |
| leftpad | p64 | degrades | -0.3537 [-0.3673, -0.3397] | 0.359 -> 0.005 | 0.9887 | 0.9207 | 13.3 |
| leftpad | p256 | degrades | -0.3590 [-0.3733, -0.3457] | 0.359 -> 0.000 | 1.0000 | 0.9983 | 13.2 |
| cache | block16 | degrades | -0.3353 [-0.3493, -0.3220] | 0.359 -> 0.024 | 0.9667 | 0.8247 | 11.5 |
| cache | block64 | degrades | -0.3537 [-0.3677, -0.3400] | 0.359 -> 0.005 | 0.9887 | 0.9207 | 13.3 |
| cache | block256 | degrades | -0.3590 [-0.3727, -0.3453] | 0.359 -> 0.000 | 1.0000 | 0.9983 | 13.2 |
| cache | gap16 | unchanged | -0.0003 [-0.0033, +0.0027] | 0.359 -> 0.359 | 0.0247 | 0.0127 | 0.0766 |
| cache | gap64 | unchanged | -0.0013 [-0.0063, +0.0037] | 0.359 -> 0.358 | 0.0800 | 0.0357 | 0.271 |
| cache | gap256 | degrades | -0.0757 [-0.0853, -0.0663] | 0.359 -> 0.283 | 0.2127 | 0.0567 | 0.515 |
| window | k16 | degrades | -0.1492 [-0.1624, -0.1360] | 0.156 -> 0.007 | 0.9772 | 0.7940 | 12.6 |
| window | k64 | degrades | -0.1508 [-0.1644, -0.1372] | 0.156 -> 0.006 | 0.9244 | 0.8480 | 14 |
| window | k256 | degrades | -0.1564 [-0.1700, -0.1432] | 0.156 -> 0.000 | 0.9228 | 0.9204 | 13.8 |
| offset_bf16 | k16 | degrades | -0.3347 [-0.3483, -0.3210] | 0.358 -> 0.024 | 0.9690 | 0.8333 | 11.6 |
| offset_bf16 | k64 | degrades | -0.3527 [-0.3663, -0.3390] | 0.358 -> 0.006 | 0.9887 | 0.9243 | 13.3 |
| offset_bf16 | k256 | degrades | -0.3583 [-0.3720, -0.3450] | 0.358 -> 0.000 | 1.0000 | 0.9967 | 13.4 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy changes at k16, k64, k256 -> recompute/reset positions
- left-pad with naive absolute position ids: accuracy changes at p16, p64, p256 -> recompute/reset positions
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy changes at block16, block64, block256 -> recompute/reset positions
- append to a cached prefix with a position gap: accuracy changes at gap256 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy changes at k16, k64, k256 -> recompute/reset positions
