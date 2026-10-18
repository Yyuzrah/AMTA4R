# Answers under shifted position indices: gpt-neo-1.3B

- Hugging Face Hub model `EleutherAI/gpt-neo-1.3B`; license: MIT
- position class **APE** (gpt_neo), trained context 2048, 1.37 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | degrades | -0.0580 [-0.0717, -0.0447] | 0.502 -> 0.444 | 0.3957 | 0.2213 | 5.29 |
| offset | k64 | degrades | -0.1300 [-0.1457, -0.1143] | 0.502 -> 0.372 | 0.5123 | 0.2910 | 6.28 |
| offset | k256 | degrades | -0.2817 [-0.2967, -0.2667] | 0.502 -> 0.220 | 0.7423 | 0.5907 | 9.33 |
| offset | k1024 | degrades | -0.3872 [-0.4040, -0.3712] | 0.471 -> 0.084 | 0.8996 | 0.8060 | 14 |
| leftpad | p16 | degrades | -0.0580 [-0.0713, -0.0447] | 0.502 -> 0.444 | 0.3957 | 0.2213 | 5.29 |
| leftpad | p64 | degrades | -0.1300 [-0.1457, -0.1143] | 0.502 -> 0.372 | 0.5123 | 0.2910 | 6.28 |
| leftpad | p256 | degrades | -0.2817 [-0.2967, -0.2663] | 0.502 -> 0.220 | 0.7423 | 0.5907 | 9.33 |
| leftpad | p1024 | degrades | -0.3872 [-0.4040, -0.3704] | 0.471 -> 0.084 | 0.8996 | 0.8060 | 14 |
| cache | block16 | degrades | -0.0580 [-0.0713, -0.0447] | 0.502 -> 0.444 | 0.3957 | 0.2213 | 5.29 |
| cache | block64 | degrades | -0.1300 [-0.1457, -0.1143] | 0.502 -> 0.372 | 0.5123 | 0.2910 | 6.28 |
| cache | block256 | degrades | -0.2817 [-0.2967, -0.2667] | 0.502 -> 0.220 | 0.7423 | 0.5907 | 9.33 |
| cache | block1024 | degrades | -0.3872 [-0.4040, -0.3708] | 0.471 -> 0.084 | 0.8996 | 0.8060 | 14 |
| cache | gap16 | unchanged | -0.0033 [-0.0087, +0.0020] | 0.502 -> 0.498 | 0.0607 | 0.0293 | 0.391 |
| cache | gap64 | degrades | -0.0303 [-0.0403, -0.0203] | 0.502 -> 0.471 | 0.2560 | 0.1257 | 1.17 |
| cache | gap256 | degrades | -0.4173 [-0.4330, -0.4020] | 0.502 -> 0.084 | 0.8870 | 0.6723 | 12.7 |
| cache | gap1024 | degrades | -0.4668 [-0.4836, -0.4500] | 0.471 -> 0.004 | 0.9904 | 0.9264 | 17.2 |
| window | k16 | degrades | -0.0660 [-0.0816, -0.0504] | 0.281 -> 0.215 | 0.5712 | 0.2560 | 6.17 |
| window | k64 | degrades | -0.0532 [-0.0704, -0.0360] | 0.281 -> 0.228 | 0.6704 | 0.3348 | 6.81 |
| window | k256 | degrades | -0.1616 [-0.1788, -0.1444] | 0.281 -> 0.120 | 0.8276 | 0.6532 | 10.9 |
| window | k1024 | degrades | -0.1932 [-0.2104, -0.1760] | 0.281 -> 0.088 | 0.8860 | 0.7876 | 14 |
| offset_bf16 | k16 | degrades | -0.0583 [-0.0727, -0.0443] | 0.498 -> 0.439 | 0.4130 | 0.2317 | 5.38 |
| offset_bf16 | k64 | degrades | -0.1347 [-0.1507, -0.1187] | 0.498 -> 0.363 | 0.5307 | 0.2983 | 6.06 |
| offset_bf16 | k256 | degrades | -0.2903 [-0.3063, -0.2750] | 0.498 -> 0.207 | 0.7603 | 0.6117 | 9.5 |
| offset_bf16 | k1024 | degrades | -0.3916 [-0.4080, -0.3752] | 0.468 -> 0.077 | 0.9028 | 0.8008 | 14 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy changes at k16, k64, k256, k1024 -> recompute/reset positions
- left-pad with naive absolute position ids: accuracy changes at p16, p64, p256, p1024 -> recompute/reset positions
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy changes at block16, block64, block256, block1024 -> recompute/reset positions
- append to a cached prefix with a position gap: accuracy changes at gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy changes at k16, k64, k256, k1024 -> recompute/reset positions
