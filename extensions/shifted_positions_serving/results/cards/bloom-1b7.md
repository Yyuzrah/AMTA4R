# Answers under shifted position indices: bloom-1b7

- Hugging Face Hub model `bigscience/bloom-1b7`; license: BigScience BLOOM RAIL License v1.0 (open RAIL; research use permitted with use restrictions)
- position class **ALiBi** (bloom), trained context 2048, 1.72 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.403 -> 0.403 | 0.0000 | 0.0000 | 0 |
| offset | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.403 -> 0.403 | 0.0003 | 0.0000 | 0 |
| offset | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.403 -> 0.403 | 0.0003 | 0.0000 | 0 |
| offset | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.382 -> 0.382 | 0.0000 | 0.0000 | 0 |
| leftpad | p16 | unchanged | -0.0003 [-0.0010, +0.0000] | 0.403 -> 0.403 | 0.0003 | 0.0003 | 0 |
| leftpad | p64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.403 -> 0.403 | 0.0000 | 0.0000 | 0 |
| leftpad | p256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.403 -> 0.403 | 0.0000 | 0.0000 | 0 |
| leftpad | p1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.382 -> 0.382 | 0.0000 | 0.0000 | 0 |
| cache | block16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.403 -> 0.403 | 0.0003 | 0.0000 | 0 |
| cache | block64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.403 -> 0.403 | 0.0000 | 0.0000 | 0 |
| cache | block256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.403 -> 0.403 | 0.0000 | 0.0000 | 0 |
| cache | block1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.382 -> 0.382 | 0.0004 | 0.0000 | 0 |
| cache | gap16 | degrades | -0.0553 [-0.0693, -0.0420] | 0.403 -> 0.348 | 0.5667 | 0.3407 | 4.5 |
| cache | gap64 | degrades | -0.1877 [-0.2037, -0.1720] | 0.403 -> 0.215 | 0.7550 | 0.5433 | 6.2 |
| cache | gap256 | degrades | -0.1720 [-0.1877, -0.1560] | 0.403 -> 0.231 | 0.7477 | 0.4700 | 7.32 |
| cache | gap1024 | degrades | -0.3192 [-0.3356, -0.3024] | 0.382 -> 0.063 | 0.9356 | 0.8564 | 12.9 |
| window | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.367 -> 0.367 | 0.0000 | 0.0000 | 0 |
| window | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.367 -> 0.367 | 0.0004 | 0.0000 | 0 |
| window | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.367 -> 0.367 | 0.0004 | 0.0000 | 0 |
| window | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.367 -> 0.367 | 0.0000 | 0.0000 | 0 |
| offset_bf16 | k16 | unchanged | -0.0020 [-0.0100, +0.0063] | 0.349 -> 0.347 | 0.1800 | 0.1230 | 0.25 |
| offset_bf16 | k64 | unchanged | -0.0067 [-0.0153, +0.0017] | 0.349 -> 0.342 | 0.1903 | 0.1333 | 0.281 |
| offset_bf16 | k256 | degrades | -0.0190 [-0.0287, -0.0090] | 0.349 -> 0.330 | 0.2607 | 0.1543 | 0.656 |
| offset_bf16 | k1024 | degrades | -0.3456 [-0.3600, -0.3308] | 0.379 -> 0.033 | 0.9252 | 0.4584 | 6.75 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- left-pad with naive absolute position ids: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy unchanged at every tested k (max answer-flip rate 0.000)
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy unchanged at every tested k (max answer-flip rate 0.000)
