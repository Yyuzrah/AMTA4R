# Answers under shifted position indices: pythia-410m

- Hugging Face Hub model `EleutherAI/pythia-410m`; license: Apache-2.0
- position class **RoPE** (gpt_neox), trained context 2048, 0.405 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.437 -> 0.437 | 0.0000 | 0.0000 | 0.0011 |
| offset | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.437 -> 0.437 | 0.0000 | 0.0000 | 0.0011 |
| offset | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.437 -> 0.437 | 0.0000 | 0.0000 | 0.0011 |
| offset | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.425 -> 0.425 | 0.0000 | 0.0000 | 0.0011 |
| leftpad | p16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.437 -> 0.437 | 0.0003 | 0.0000 | 0.0011 |
| leftpad | p64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.437 -> 0.437 | 0.0000 | 0.0000 | 0.0011 |
| leftpad | p256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.437 -> 0.437 | 0.0000 | 0.0000 | 0.0011 |
| leftpad | p1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.425 -> 0.425 | 0.0000 | 0.0000 | 0.0011 |
| cache | block16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.437 -> 0.437 | 0.0000 | 0.0000 | 0.0011 |
| cache | block64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.437 -> 0.437 | 0.0000 | 0.0000 | 0.0011 |
| cache | block256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.437 -> 0.437 | 0.0003 | 0.0003 | 0.0011 |
| cache | block1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.425 -> 0.425 | 0.0000 | 0.0000 | 0.001 |
| cache | gap16 | degrades | -0.0400 [-0.0527, -0.0273] | 0.437 -> 0.397 | 0.4160 | 0.2660 | 2.4 |
| cache | gap64 | degrades | -0.0787 [-0.0923, -0.0653] | 0.437 -> 0.358 | 0.5030 | 0.3280 | 3.06 |
| cache | gap256 | degrades | -0.0823 [-0.0967, -0.0680] | 0.437 -> 0.355 | 0.5507 | 0.3513 | 3.82 |
| cache | gap1024 | degrades | -0.1464 [-0.1628, -0.1304] | 0.425 -> 0.279 | 0.6544 | 0.4180 | 4.53 |
| window | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.412 -> 0.412 | 0.0000 | 0.0000 | 0.0011 |
| window | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.412 -> 0.412 | 0.0000 | 0.0000 | 0.0011 |
| window | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.412 -> 0.412 | 0.0000 | 0.0000 | 0.0011 |
| window | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.412 -> 0.412 | 0.0000 | 0.0000 | 0.0011 |
| offset_bf16 | k16 | unchanged | -0.0077 [-0.0197, +0.0043] | 0.393 -> 0.385 | 0.3820 | 0.2337 | 2.84 |
| offset_bf16 | k64 | unchanged | -0.0100 [-0.0223, +0.0023] | 0.393 -> 0.383 | 0.3907 | 0.2397 | 2.87 |
| offset_bf16 | k256 | unchanged | -0.0040 [-0.0153, +0.0073] | 0.393 -> 0.389 | 0.3743 | 0.2320 | 2.89 |
| offset_bf16 | k1024 | unchanged | +0.0000 [-0.0112, +0.0116] | 0.392 -> 0.392 | 0.3316 | 0.1568 | 2.69 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- left-pad with naive absolute position ids: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy unchanged at every tested k (max answer-flip rate 0.000)
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy unchanged at every tested k (max answer-flip rate 0.000)
