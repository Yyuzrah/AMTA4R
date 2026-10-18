# Answers under shifted position indices: opt-6.7b

- Hugging Face Hub model `facebook/opt-6.7b`; license: OPT-175B License Agreement (hub tag "other"; non-commercial research use permitted)
- position class **APE** (opt), trained context 2048, 6.66 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | degrades | -0.0907 [-0.1060, -0.0757] | 0.521 -> 0.430 | 0.4790 | 0.2820 | 4.52 |
| offset | k64 | degrades | -0.4110 [-0.4263, -0.3953] | 0.521 -> 0.110 | 0.8383 | 0.6250 | 6.44 |
| offset | k256 | degrades | -0.5170 [-0.5323, -0.5017] | 0.521 -> 0.004 | 0.9937 | 0.9620 | 12.5 |
| offset | k1024 | degrades | -0.5552 [-0.5716, -0.5384] | 0.555 -> 0.000 | 0.9992 | 0.9948 | 15.8 |
| leftpad | p16 | degrades | -0.0907 [-0.1057, -0.0757] | 0.521 -> 0.430 | 0.4790 | 0.2820 | 4.52 |
| leftpad | p64 | degrades | -0.4110 [-0.4270, -0.3953] | 0.521 -> 0.110 | 0.8383 | 0.6250 | 6.44 |
| leftpad | p256 | degrades | -0.5170 [-0.5330, -0.5013] | 0.521 -> 0.004 | 0.9937 | 0.9620 | 12.5 |
| leftpad | p1024 | degrades | -0.5552 [-0.5716, -0.5384] | 0.555 -> 0.000 | 0.9992 | 0.9948 | 15.8 |
| cache | block16 | degrades | -0.0907 [-0.1060, -0.0757] | 0.521 -> 0.430 | 0.4790 | 0.2820 | 4.52 |
| cache | block64 | degrades | -0.4110 [-0.4267, -0.3950] | 0.521 -> 0.110 | 0.8383 | 0.6250 | 6.44 |
| cache | block256 | degrades | -0.5170 [-0.5327, -0.5013] | 0.521 -> 0.004 | 0.9937 | 0.9620 | 12.5 |
| cache | block1024 | degrades | -0.5552 [-0.5720, -0.5384] | 0.555 -> 0.000 | 0.9992 | 0.9948 | 15.8 |
| cache | gap16 | unchanged | -0.0010 [-0.0063, +0.0040] | 0.521 -> 0.520 | 0.0460 | 0.0227 | 0.159 |
| cache | gap64 | unchanged | +0.0043 [-0.0013, +0.0100] | 0.521 -> 0.525 | 0.0560 | 0.0310 | 0.192 |
| cache | gap256 | unchanged | +0.0060 [-0.0013, +0.0133] | 0.521 -> 0.527 | 0.1013 | 0.0497 | 0.377 |
| cache | gap1024 | unchanged | +0.0012 [-0.0096, +0.0116] | 0.555 -> 0.556 | 0.1760 | 0.0696 | 0.781 |
| window | k16 | degrades | -0.2984 [-0.3152, -0.2816] | 0.505 -> 0.207 | 0.6896 | 0.4416 | 5.87 |
| window | k64 | degrades | -0.4028 [-0.4204, -0.3852] | 0.505 -> 0.102 | 0.8412 | 0.5724 | 6.36 |
| window | k256 | degrades | -0.5008 [-0.5180, -0.4836] | 0.505 -> 0.004 | 0.9884 | 0.9512 | 11.9 |
| window | k1024 | degrades | -0.5052 [-0.5220, -0.4884] | 0.505 -> 0.000 | 0.9992 | 0.9952 | 15.8 |
| offset_bf16 | k16 | degrades | -0.0863 [-0.1017, -0.0707] | 0.514 -> 0.428 | 0.4943 | 0.2940 | 4.58 |
| offset_bf16 | k64 | degrades | -0.4093 [-0.4253, -0.3933] | 0.514 -> 0.105 | 0.8470 | 0.6310 | 6.46 |
| offset_bf16 | k256 | degrades | -0.5093 [-0.5247, -0.4940] | 0.514 -> 0.005 | 0.9943 | 0.9600 | 12.5 |
| offset_bf16 | k1024 | degrades | -0.5492 [-0.5660, -0.5328] | 0.549 -> 0.000 | 0.9988 | 0.9948 | 15.8 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy changes at k16, k64, k256, k1024 -> recompute/reset positions
- left-pad with naive absolute position ids: accuracy changes at p16, p64, p256, p1024 -> recompute/reset positions
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy changes at block16, block64, block256, block1024 -> recompute/reset positions
- append to a cached prefix with a position gap: accuracy unchanged at every tested k (max answer-flip rate 0.176)
- start a 128-token sliding window at a non-zero position: accuracy changes at k16, k64, k256, k1024 -> recompute/reset positions
