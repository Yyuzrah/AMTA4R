# Answers under shifted position indices: bloom-3b

- Hugging Face Hub model `bigscience/bloom-3b`; license: BigScience BLOOM RAIL License v1.0 (open RAIL; research use permitted with use restrictions)
- position class **ALiBi** (bloom), trained context 2048, 3.0 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.425 -> 0.425 | 0.0000 | 0.0000 | 0 |
| offset | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.425 -> 0.425 | 0.0000 | 0.0000 | 0 |
| offset | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.425 -> 0.425 | 0.0000 | 0.0000 | 0 |
| offset | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.456 -> 0.456 | 0.0000 | 0.0000 | 0 |
| leftpad | p16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.425 -> 0.425 | 0.0000 | 0.0000 | 0 |
| leftpad | p64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.425 -> 0.425 | 0.0000 | 0.0000 | 0 |
| leftpad | p256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.425 -> 0.425 | 0.0000 | 0.0000 | 0 |
| leftpad | p1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.456 -> 0.456 | 0.0000 | 0.0000 | 0 |
| cache | block16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.425 -> 0.425 | 0.0000 | 0.0000 | 0 |
| cache | block64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.425 -> 0.425 | 0.0000 | 0.0000 | 0 |
| cache | block256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.425 -> 0.425 | 0.0000 | 0.0000 | 0 |
| cache | block1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.456 -> 0.456 | 0.0000 | 0.0000 | 0 |
| cache | gap16 | degrades | -0.0797 [-0.0957, -0.0643] | 0.425 -> 0.345 | 0.6423 | 0.4347 | 4.86 |
| cache | gap64 | degrades | -0.2590 [-0.2743, -0.2437] | 0.425 -> 0.166 | 0.8277 | 0.6427 | 6.67 |
| cache | gap256 | degrades | -0.1990 [-0.2167, -0.1817] | 0.425 -> 0.226 | 0.7927 | 0.5533 | 6.87 |
| cache | gap1024 | degrades | -0.4352 [-0.4516, -0.4188] | 0.456 -> 0.021 | 0.9684 | 0.9256 | 9.41 |
| window | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.434 -> 0.434 | 0.0000 | 0.0000 | 0 |
| window | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.434 -> 0.434 | 0.0000 | 0.0000 | 0 |
| window | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.434 -> 0.434 | 0.0000 | 0.0000 | 0 |
| window | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.434 -> 0.434 | 0.0000 | 0.0000 | 0 |
| offset_bf16 | k16 | unchanged | +0.0003 [-0.0087, +0.0090] | 0.418 -> 0.418 | 0.1257 | 0.0980 | 0.219 |
| offset_bf16 | k64 | unchanged | -0.0053 [-0.0140, +0.0033] | 0.418 -> 0.412 | 0.1323 | 0.0980 | 0.266 |
| offset_bf16 | k256 | degrades | -0.0333 [-0.0430, -0.0240] | 0.418 -> 0.384 | 0.1757 | 0.1030 | 0.562 |
| offset_bf16 | k1024 | degrades | -0.2880 [-0.3052, -0.2704] | 0.458 -> 0.170 | 0.7372 | 0.2980 | 3.75 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- left-pad with naive absolute position ids: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy unchanged at every tested k (max answer-flip rate 0.000)
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy unchanged at every tested k (max answer-flip rate 0.000)
