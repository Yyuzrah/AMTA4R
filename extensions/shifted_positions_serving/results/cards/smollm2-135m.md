# Answers under shifted position indices: smollm2-135m

- Hugging Face Hub model `HuggingFaceTB/SmolLM2-135M`; license: Apache-2.0
- position class **RoPE** (llama), trained context 8192, 0.135 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.331 -> 0.331 | 0.0000 | 0.0000 | 0 |
| offset | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.331 -> 0.331 | 0.0000 | 0.0000 | 0 |
| offset | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.331 -> 0.331 | 0.0000 | 0.0000 | 0 |
| offset | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.331 -> 0.331 | 0.0000 | 0.0000 | 0.0001 |
| leftpad | p16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.331 -> 0.331 | 0.0000 | 0.0000 | 0 |
| leftpad | p64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.331 -> 0.331 | 0.0000 | 0.0000 | 0 |
| leftpad | p256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.331 -> 0.331 | 0.0000 | 0.0000 | 0 |
| leftpad | p1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.331 -> 0.331 | 0.0000 | 0.0000 | 0.0001 |
| cache | block16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.331 -> 0.331 | 0.0000 | 0.0000 | 0 |
| cache | block64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.331 -> 0.331 | 0.0000 | 0.0000 | 0 |
| cache | block256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.331 -> 0.331 | 0.0000 | 0.0000 | 0 |
| cache | block1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.331 -> 0.331 | 0.0000 | 0.0000 | 0.0001 |
| cache | gap16 | degrades | -0.1080 [-0.1210, -0.0957] | 0.331 -> 0.223 | 0.5800 | 0.1903 | 4.48 |
| cache | gap64 | degrades | -0.1370 [-0.1503, -0.1237] | 0.331 -> 0.194 | 0.6550 | 0.2427 | 4.85 |
| cache | gap256 | degrades | -0.1340 [-0.1483, -0.1200] | 0.331 -> 0.197 | 0.7050 | 0.2663 | 5.2 |
| cache | gap1024 | degrades | -0.1443 [-0.1583, -0.1300] | 0.331 -> 0.187 | 0.7643 | 0.3430 | 6.37 |
| window | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.386 -> 0.386 | 0.0000 | 0.0000 | 0 |
| window | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.386 -> 0.386 | 0.0000 | 0.0000 | 0 |
| window | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.386 -> 0.386 | 0.0000 | 0.0000 | 0 |
| window | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.386 -> 0.386 | 0.0000 | 0.0000 | 0.0001 |
| offset_bf16 | k16 | unchanged | -0.0027 [-0.0060, +0.0007] | 0.332 -> 0.329 | 0.0783 | 0.0230 | 0.372 |
| offset_bf16 | k64 | unchanged | -0.0017 [-0.0063, +0.0027] | 0.332 -> 0.330 | 0.0847 | 0.0253 | 0.375 |
| offset_bf16 | k256 | unchanged | -0.0003 [-0.0043, +0.0037] | 0.332 -> 0.331 | 0.0890 | 0.0310 | 0.375 |
| offset_bf16 | k1024 | unchanged | -0.0037 [-0.0080, +0.0007] | 0.332 -> 0.328 | 0.0850 | 0.0253 | 0.375 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- left-pad with naive absolute position ids: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy unchanged at every tested k (max answer-flip rate 0.000)
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy unchanged at every tested k (max answer-flip rate 0.000)
