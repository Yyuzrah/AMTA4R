# Answers under shifted position indices: smollm2-1.7b

- Hugging Face Hub model `HuggingFaceTB/SmolLM2-1.7B`; license: Apache-2.0
- position class **RoPE** (llama), trained context 8192, 1.71 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.562 -> 0.562 | 0.0000 | 0.0000 | 0 |
| offset | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.562 -> 0.562 | 0.0000 | 0.0000 | 0 |
| offset | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.562 -> 0.562 | 0.0000 | 0.0000 | 0 |
| offset | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.562 -> 0.562 | 0.0000 | 0.0000 | 0.0001 |
| leftpad | p16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.562 -> 0.562 | 0.0000 | 0.0000 | 0 |
| leftpad | p64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.562 -> 0.562 | 0.0000 | 0.0000 | 0 |
| leftpad | p256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.562 -> 0.562 | 0.0000 | 0.0000 | 0.0001 |
| leftpad | p1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.562 -> 0.562 | 0.0000 | 0.0000 | 0.0001 |
| cache | block16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.562 -> 0.562 | 0.0000 | 0.0000 | 0 |
| cache | block64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.562 -> 0.562 | 0.0000 | 0.0000 | 0 |
| cache | block256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.562 -> 0.562 | 0.0000 | 0.0000 | 0 |
| cache | block1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.562 -> 0.562 | 0.0000 | 0.0000 | 0.0001 |
| cache | gap16 | improves | +0.0473 [+0.0357, +0.0590] | 0.562 -> 0.610 | 0.2720 | 0.1690 | 3.95 |
| cache | gap64 | improves | +0.0147 [+0.0007, +0.0287] | 0.562 -> 0.577 | 0.3753 | 0.2103 | 4.83 |
| cache | gap256 | degrades | -0.0823 [-0.0983, -0.0663] | 0.562 -> 0.480 | 0.5383 | 0.2470 | 4.91 |
| cache | gap1024 | degrades | -0.1647 [-0.1807, -0.1487] | 0.562 -> 0.398 | 0.6357 | 0.2910 | 5.33 |
| window | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.595 -> 0.595 | 0.0000 | 0.0000 | 0 |
| window | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.595 -> 0.595 | 0.0000 | 0.0000 | 0 |
| window | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.595 -> 0.595 | 0.0000 | 0.0000 | 0 |
| window | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.595 -> 0.595 | 0.0000 | 0.0000 | 0.0001 |
| offset_bf16 | k16 | unchanged | -0.0007 [-0.0057, +0.0043] | 0.562 -> 0.561 | 0.0600 | 0.0370 | 0.31 |
| offset_bf16 | k64 | unchanged | -0.0020 [-0.0067, +0.0027] | 0.562 -> 0.560 | 0.0547 | 0.0300 | 0.312 |
| offset_bf16 | k256 | unchanged | -0.0007 [-0.0057, +0.0040] | 0.562 -> 0.561 | 0.0597 | 0.0343 | 0.312 |
| offset_bf16 | k1024 | unchanged | -0.0037 [-0.0087, +0.0013] | 0.562 -> 0.558 | 0.0617 | 0.0347 | 0.312 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- left-pad with naive absolute position ids: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy unchanged at every tested k (max answer-flip rate 0.000)
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy unchanged at every tested k (max answer-flip rate 0.000)
