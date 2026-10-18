# Answers under shifted position indices: bloom-1b1

- Hugging Face Hub model `bigscience/bloom-1b1`; license: BigScience BLOOM RAIL License v1.0 (open RAIL; research use permitted with use restrictions)
- position class **ALiBi** (bloom), trained context 2048, 1.07 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.369 -> 0.369 | 0.0000 | 0.0000 | 0 |
| offset | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.369 -> 0.369 | 0.0000 | 0.0000 | 0 |
| offset | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.369 -> 0.369 | 0.0000 | 0.0000 | 0 |
| offset | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.399 -> 0.399 | 0.0000 | 0.0000 | 0 |
| leftpad | p16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.369 -> 0.369 | 0.0000 | 0.0000 | 0 |
| leftpad | p64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.369 -> 0.369 | 0.0000 | 0.0000 | 0 |
| leftpad | p256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.369 -> 0.369 | 0.0000 | 0.0000 | 0 |
| leftpad | p1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.399 -> 0.399 | 0.0000 | 0.0000 | 0 |
| cache | block16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.369 -> 0.369 | 0.0000 | 0.0000 | 0 |
| cache | block64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.369 -> 0.369 | 0.0000 | 0.0000 | 0 |
| cache | block256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.369 -> 0.369 | 0.0000 | 0.0000 | 0 |
| cache | block1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.399 -> 0.399 | 0.0000 | 0.0000 | 0 |
| cache | gap16 | unchanged | -0.0047 [-0.0180, +0.0087] | 0.369 -> 0.364 | 0.5773 | 0.3423 | 4.78 |
| cache | gap64 | degrades | -0.1773 [-0.1923, -0.1620] | 0.369 -> 0.192 | 0.7693 | 0.5487 | 4.76 |
| cache | gap256 | degrades | -0.1270 [-0.1423, -0.1117] | 0.369 -> 0.242 | 0.7310 | 0.5150 | 6.63 |
| cache | gap1024 | degrades | -0.3384 [-0.3548, -0.3216] | 0.399 -> 0.061 | 0.9396 | 0.8800 | 10.5 |
| window | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.385 -> 0.385 | 0.0000 | 0.0000 | 0 |
| window | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.385 -> 0.385 | 0.0000 | 0.0000 | 0 |
| window | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.385 -> 0.385 | 0.0000 | 0.0000 | 0 |
| window | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.385 -> 0.385 | 0.0000 | 0.0000 | 0 |
| offset_bf16 | k16 | unchanged | +0.0033 [-0.0037, +0.0103] | 0.346 -> 0.349 | 0.1817 | 0.1227 | 0.241 |
| offset_bf16 | k64 | unchanged | +0.0013 [-0.0053, +0.0080] | 0.346 -> 0.347 | 0.1883 | 0.1233 | 0.281 |
| offset_bf16 | k256 | unchanged | -0.0067 [-0.0150, +0.0017] | 0.346 -> 0.339 | 0.2697 | 0.1500 | 0.616 |
| offset_bf16 | k1024 | degrades | -0.3608 [-0.3760, -0.3456] | 0.396 -> 0.035 | 0.8976 | 0.4628 | 7.01 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- left-pad with naive absolute position ids: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy unchanged at every tested k (max answer-flip rate 0.000)
- append to a cached prefix with a position gap: accuracy changes at gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy unchanged at every tested k (max answer-flip rate 0.000)
