# Answers under shifted position indices: bloom-7b1

- Hugging Face Hub model `bigscience/bloom-7b1`; license: BigScience BLOOM RAIL License v1.0 (open RAIL; research use permitted with use restrictions)
- position class **ALiBi** (bloom), trained context 2048, 7.07 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.591 -> 0.591 | 0.0000 | 0.0000 | 0 |
| offset | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.591 -> 0.591 | 0.0000 | 0.0000 | 0 |
| offset | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.591 -> 0.591 | 0.0000 | 0.0000 | 0 |
| offset | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.573 -> 0.573 | 0.0000 | 0.0000 | 0 |
| leftpad | p16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.591 -> 0.591 | 0.0000 | 0.0000 | 0 |
| leftpad | p64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.591 -> 0.591 | 0.0000 | 0.0000 | 0 |
| leftpad | p256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.591 -> 0.591 | 0.0000 | 0.0000 | 0 |
| leftpad | p1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.573 -> 0.573 | 0.0000 | 0.0000 | 0 |
| cache | block16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.591 -> 0.591 | 0.0000 | 0.0000 | 0 |
| cache | block64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.591 -> 0.591 | 0.0000 | 0.0000 | 0 |
| cache | block256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.591 -> 0.591 | 0.0000 | 0.0000 | 0 |
| cache | block1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.573 -> 0.573 | 0.0000 | 0.0000 | 0 |
| cache | gap16 | degrades | -0.1463 [-0.1607, -0.1317] | 0.591 -> 0.445 | 0.5080 | 0.3137 | 5.01 |
| cache | gap64 | degrades | -0.3077 [-0.3237, -0.2917] | 0.591 -> 0.283 | 0.7113 | 0.4940 | 5.81 |
| cache | gap256 | degrades | -0.3587 [-0.3767, -0.3410] | 0.591 -> 0.232 | 0.7570 | 0.5260 | 8.36 |
| cache | gap1024 | degrades | -0.4964 [-0.5144, -0.4784] | 0.573 -> 0.076 | 0.9224 | 0.8636 | 10.6 |
| window | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.550 -> 0.550 | 0.0000 | 0.0000 | 0 |
| window | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.550 -> 0.550 | 0.0000 | 0.0000 | 0 |
| window | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.550 -> 0.550 | 0.0000 | 0.0000 | 0 |
| window | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.550 -> 0.550 | 0.0000 | 0.0000 | 0 |
| offset_bf16 | k16 | unchanged | +0.0013 [-0.0083, +0.0110] | 0.580 -> 0.581 | 0.1103 | 0.0883 | 0.207 |
| offset_bf16 | k64 | unchanged | -0.0097 [-0.0197, +0.0003] | 0.580 -> 0.570 | 0.1293 | 0.1027 | 0.273 |
| offset_bf16 | k256 | degrades | -0.0247 [-0.0350, -0.0147] | 0.580 -> 0.555 | 0.1740 | 0.1170 | 0.594 |
| offset_bf16 | k1024 | degrades | -0.3040 [-0.3224, -0.2860] | 0.569 -> 0.265 | 0.6488 | 0.2860 | 4.75 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- left-pad with naive absolute position ids: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy unchanged at every tested k (max answer-flip rate 0.000)
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy unchanged at every tested k (max answer-flip rate 0.000)
