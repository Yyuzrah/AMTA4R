# Answers under shifted position indices: pythia-1b

- Hugging Face Hub model `EleutherAI/pythia-1b`; license: Apache-2.0
- position class **RoPE** (gpt_neox), trained context 2048, 1.01 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.457 -> 0.457 | 0.0000 | 0.0000 | 0 |
| offset | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.457 -> 0.457 | 0.0000 | 0.0000 | 0 |
| offset | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.457 -> 0.457 | 0.0000 | 0.0000 | 0 |
| offset | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.484 -> 0.484 | 0.0000 | 0.0000 | 0 |
| leftpad | p16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.457 -> 0.457 | 0.0000 | 0.0000 | 0 |
| leftpad | p64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.457 -> 0.457 | 0.0000 | 0.0000 | 0 |
| leftpad | p256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.457 -> 0.457 | 0.0000 | 0.0000 | 0 |
| leftpad | p1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.484 -> 0.484 | 0.0000 | 0.0000 | 0 |
| cache | block16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.457 -> 0.457 | 0.0000 | 0.0000 | 0 |
| cache | block64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.457 -> 0.457 | 0.0000 | 0.0000 | 0 |
| cache | block256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.457 -> 0.457 | 0.0000 | 0.0000 | 0 |
| cache | block1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.484 -> 0.484 | 0.0000 | 0.0000 | 0 |
| cache | gap16 | degrades | -0.2700 [-0.2853, -0.2547] | 0.457 -> 0.187 | 0.6850 | 0.3870 | 3.9 |
| cache | gap64 | degrades | -0.4397 [-0.4560, -0.4237] | 0.457 -> 0.017 | 0.9513 | 0.6000 | 7.53 |
| cache | gap256 | degrades | -0.4360 [-0.4523, -0.4193] | 0.457 -> 0.021 | 0.9593 | 0.6610 | 7.32 |
| cache | gap1024 | degrades | -0.4460 [-0.4644, -0.4284] | 0.484 -> 0.038 | 0.9264 | 0.5788 | 6.43 |
| window | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.476 -> 0.476 | 0.0000 | 0.0000 | 0 |
| window | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.476 -> 0.476 | 0.0000 | 0.0000 | 0 |
| window | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.476 -> 0.476 | 0.0000 | 0.0000 | 0 |
| window | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.476 -> 0.476 | 0.0000 | 0.0000 | 0 |
| offset_bf16 | k16 | unchanged | -0.0013 [-0.0060, +0.0033] | 0.454 -> 0.453 | 0.0430 | 0.0283 | 0.156 |
| offset_bf16 | k64 | unchanged | +0.0010 [-0.0033, +0.0053] | 0.454 -> 0.455 | 0.0437 | 0.0250 | 0.156 |
| offset_bf16 | k256 | unchanged | +0.0017 [-0.0027, +0.0060] | 0.454 -> 0.456 | 0.0417 | 0.0237 | 0.158 |
| offset_bf16 | k1024 | unchanged | +0.0012 [-0.0036, +0.0060] | 0.482 -> 0.484 | 0.0436 | 0.0192 | 0.156 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- left-pad with naive absolute position ids: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy unchanged at every tested k (max answer-flip rate 0.000)
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy unchanged at every tested k (max answer-flip rate 0.000)
