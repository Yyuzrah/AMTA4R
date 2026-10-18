# Answers under shifted position indices: opt-1.3b

- Hugging Face Hub model `facebook/opt-1.3b`; license: OPT-175B License Agreement (hub tag "other"; non-commercial research use permitted)
- position class **APE** (opt), trained context 2048, 1.32 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | degrades | -0.0713 [-0.0850, -0.0577] | 0.348 -> 0.277 | 0.5373 | 0.3270 | 3.58 |
| offset | k64 | degrades | -0.3480 [-0.3633, -0.3330] | 0.348 -> 0.000 | 0.9970 | 0.9863 | 9.58 |
| offset | k256 | degrades | -0.3423 [-0.3570, -0.3273] | 0.348 -> 0.006 | 0.9887 | 0.9507 | 12.3 |
| offset | k1024 | degrades | -0.3600 [-0.3756, -0.3444] | 0.360 -> 0.000 | 0.9964 | 0.9988 | 14.6 |
| leftpad | p16 | degrades | -0.0713 [-0.0850, -0.0577] | 0.348 -> 0.277 | 0.5373 | 0.3270 | 3.58 |
| leftpad | p64 | degrades | -0.3480 [-0.3630, -0.3333] | 0.348 -> 0.000 | 0.9970 | 0.9863 | 9.58 |
| leftpad | p256 | degrades | -0.3423 [-0.3567, -0.3277] | 0.348 -> 0.006 | 0.9887 | 0.9507 | 12.3 |
| leftpad | p1024 | degrades | -0.3600 [-0.3760, -0.3444] | 0.360 -> 0.000 | 0.9964 | 0.9988 | 14.6 |
| cache | block16 | degrades | -0.0713 [-0.0853, -0.0577] | 0.348 -> 0.277 | 0.5373 | 0.3270 | 3.58 |
| cache | block64 | degrades | -0.3480 [-0.3627, -0.3333] | 0.348 -> 0.000 | 0.9970 | 0.9863 | 9.58 |
| cache | block256 | degrades | -0.3423 [-0.3570, -0.3277] | 0.348 -> 0.006 | 0.9887 | 0.9507 | 12.3 |
| cache | block1024 | degrades | -0.3600 [-0.3756, -0.3440] | 0.360 -> 0.000 | 0.9964 | 0.9988 | 14.6 |
| cache | gap16 | unchanged | +0.0060 [+0.0000, +0.0120] | 0.348 -> 0.354 | 0.0840 | 0.0500 | 0.255 |
| cache | gap64 | unchanged | -0.0017 [-0.0100, +0.0063] | 0.348 -> 0.347 | 0.1513 | 0.0913 | 0.403 |
| cache | gap256 | degrades | -0.0253 [-0.0357, -0.0150] | 0.348 -> 0.323 | 0.2943 | 0.1560 | 0.88 |
| cache | gap1024 | degrades | -0.0744 [-0.0876, -0.0608] | 0.360 -> 0.286 | 0.3828 | 0.1732 | 1.05 |
| window | k16 | degrades | -0.0952 [-0.1104, -0.0808] | 0.206 -> 0.111 | 0.7116 | 0.4108 | 3.65 |
| window | k64 | degrades | -0.2060 [-0.2224, -0.1904] | 0.206 -> 0.000 | 0.9920 | 0.9780 | 9.46 |
| window | k256 | degrades | -0.1996 [-0.2152, -0.1836] | 0.206 -> 0.007 | 0.9864 | 0.9412 | 11.2 |
| window | k1024 | degrades | -0.2064 [-0.2220, -0.1916] | 0.206 -> 0.000 | 0.9844 | 0.9972 | 14.6 |
| offset_bf16 | k16 | degrades | -0.0710 [-0.0850, -0.0577] | 0.345 -> 0.274 | 0.5533 | 0.3380 | 3.7 |
| offset_bf16 | k64 | degrades | -0.3450 [-0.3597, -0.3307] | 0.345 -> 0.000 | 0.9967 | 0.9853 | 9.54 |
| offset_bf16 | k256 | degrades | -0.3387 [-0.3533, -0.3243] | 0.345 -> 0.007 | 0.9887 | 0.9493 | 12.3 |
| offset_bf16 | k1024 | degrades | -0.3580 [-0.3740, -0.3424] | 0.358 -> 0.000 | 0.9960 | 0.9988 | 14.5 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy changes at k16, k64, k256, k1024 -> recompute/reset positions
- left-pad with naive absolute position ids: accuracy changes at p16, p64, p256, p1024 -> recompute/reset positions
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy changes at block16, block64, block256, block1024 -> recompute/reset positions
- append to a cached prefix with a position gap: accuracy changes at gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy changes at k16, k64, k256, k1024 -> recompute/reset positions
