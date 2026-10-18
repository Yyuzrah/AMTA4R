# Answers under shifted position indices: bloom-560m

- Hugging Face Hub model `bigscience/bloom-560m`; license: BigScience BLOOM RAIL License v1.0 (open RAIL; research use permitted with use restrictions)
- position class **ALiBi** (bloom), trained context 2048, 0.559 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.323 -> 0.323 | 0.0000 | 0.0000 | 0.0002 |
| offset | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.323 -> 0.323 | 0.0000 | 0.0000 | 0.0002 |
| offset | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.323 -> 0.323 | 0.0000 | 0.0000 | 0.0002 |
| offset | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.326 -> 0.326 | 0.0000 | 0.0000 | 0.0002 |
| leftpad | p16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.323 -> 0.323 | 0.0000 | 0.0000 | 0.0002 |
| leftpad | p64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.323 -> 0.323 | 0.0000 | 0.0000 | 0.0002 |
| leftpad | p256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.323 -> 0.323 | 0.0000 | 0.0000 | 0.0002 |
| leftpad | p1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.326 -> 0.326 | 0.0000 | 0.0000 | 0.0002 |
| cache | block16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.323 -> 0.323 | 0.0000 | 0.0000 | 0.0003 |
| cache | block64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.323 -> 0.323 | 0.0000 | 0.0000 | 0.0003 |
| cache | block256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.323 -> 0.323 | 0.0000 | 0.0000 | 0.0003 |
| cache | block1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.326 -> 0.326 | 0.0004 | 0.0004 | 0.0003 |
| cache | gap16 | degrades | -0.0213 [-0.0347, -0.0080] | 0.323 -> 0.301 | 0.5710 | 0.3587 | 5.5 |
| cache | gap64 | degrades | -0.1527 [-0.1673, -0.1383] | 0.323 -> 0.170 | 0.7387 | 0.4950 | 7.39 |
| cache | gap256 | degrades | -0.1493 [-0.1643, -0.1347] | 0.323 -> 0.173 | 0.7743 | 0.5057 | 9.49 |
| cache | gap1024 | degrades | -0.2812 [-0.2956, -0.2668] | 0.326 -> 0.044 | 0.9540 | 0.8784 | 35.9 |
| window | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.324 -> 0.324 | 0.0000 | 0.0000 | 0.0002 |
| window | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.324 -> 0.324 | 0.0000 | 0.0000 | 0.0002 |
| window | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.324 -> 0.324 | 0.0000 | 0.0000 | 0.0002 |
| window | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.324 -> 0.324 | 0.0000 | 0.0000 | 0.0002 |
| offset_bf16 | k16 | unchanged | +0.0003 [-0.0077, +0.0083] | 0.265 -> 0.265 | 0.4123 | 0.2257 | 2 |
| offset_bf16 | k64 | unchanged | -0.0023 [-0.0110, +0.0063] | 0.265 -> 0.263 | 0.4277 | 0.2267 | 2 |
| offset_bf16 | k256 | degrades | -0.0253 [-0.0347, -0.0160] | 0.265 -> 0.240 | 0.4730 | 0.2397 | 2 |
| offset_bf16 | k1024 | degrades | -0.2792 [-0.2920, -0.2668] | 0.304 -> 0.025 | 0.9044 | 0.4888 | 12 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- left-pad with naive absolute position ids: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy unchanged at every tested k (max answer-flip rate 0.000)
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy unchanged at every tested k (max answer-flip rate 0.000)
