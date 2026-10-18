# Answers under shifted position indices: opt-350m

- Hugging Face Hub model `facebook/opt-350m`; license: OPT-175B License Agreement (hub tag "other"; non-commercial research use permitted)
- position class **APE** (opt), trained context 2048, 0.331 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | degrades | -0.2230 [-0.2327, -0.2130] | 0.224 -> 0.001 | 0.9993 | 0.9880 | 14.3 |
| offset | k64 | degrades | -0.2227 [-0.2327, -0.2127] | 0.224 -> 0.001 | 0.9990 | 0.9857 | 14.8 |
| offset | k256 | degrades | -0.2237 [-0.2337, -0.2137] | 0.224 -> 0.000 | 0.9997 | 0.9973 | 15 |
| offset | k1024 | degrades | -0.2452 [-0.2556, -0.2352] | 0.245 -> 0.000 | 1.0000 | 1.0000 | 16.9 |
| leftpad | p16 | degrades | -0.2230 [-0.2330, -0.2137] | 0.224 -> 0.001 | 0.9993 | 0.9880 | 14.3 |
| leftpad | p64 | degrades | -0.2227 [-0.2323, -0.2130] | 0.224 -> 0.001 | 0.9990 | 0.9857 | 14.8 |
| leftpad | p256 | degrades | -0.2237 [-0.2337, -0.2140] | 0.224 -> 0.000 | 0.9997 | 0.9973 | 15 |
| leftpad | p1024 | degrades | -0.2452 [-0.2556, -0.2348] | 0.245 -> 0.000 | 1.0000 | 1.0000 | 16.9 |
| cache | block16 | degrades | -0.2230 [-0.2330, -0.2133] | 0.224 -> 0.001 | 0.9993 | 0.9880 | 14.3 |
| cache | block64 | degrades | -0.2227 [-0.2327, -0.2130] | 0.224 -> 0.001 | 0.9990 | 0.9857 | 14.8 |
| cache | block256 | degrades | -0.2237 [-0.2337, -0.2140] | 0.224 -> 0.000 | 0.9997 | 0.9973 | 15 |
| cache | block1024 | degrades | -0.2452 [-0.2556, -0.2348] | 0.245 -> 0.000 | 1.0000 | 1.0000 | 16.9 |
| cache | gap16 | degrades | -0.0100 [-0.0170, -0.0033] | 0.224 -> 0.214 | 0.2687 | 0.1710 | 1.18 |
| cache | gap64 | degrades | -0.1580 [-0.1683, -0.1477] | 0.224 -> 0.066 | 0.7390 | 0.5260 | 5.28 |
| cache | gap256 | degrades | -0.2160 [-0.2257, -0.2063] | 0.224 -> 0.008 | 0.9403 | 0.8587 | 10.6 |
| cache | gap1024 | degrades | -0.2432 [-0.2536, -0.2328] | 0.245 -> 0.002 | 0.9516 | 0.8528 | 10.9 |
| window | k16 | degrades | -0.2336 [-0.2440, -0.2228] | 0.234 -> 0.001 | 0.9956 | 0.9916 | 15 |
| window | k64 | degrades | -0.2332 [-0.2440, -0.2228] | 0.234 -> 0.001 | 0.9968 | 0.9836 | 14.4 |
| window | k256 | degrades | -0.2344 [-0.2452, -0.2236] | 0.234 -> 0.000 | 0.9928 | 0.9976 | 13.9 |
| window | k1024 | degrades | -0.2344 [-0.2452, -0.2236] | 0.234 -> 0.000 | 0.9964 | 1.0000 | 17.6 |
| offset_bf16 | k16 | degrades | -0.2223 [-0.2323, -0.2123] | 0.223 -> 0.001 | 0.9993 | 0.9883 | 14.4 |
| offset_bf16 | k64 | degrades | -0.2220 [-0.2320, -0.2123] | 0.223 -> 0.001 | 0.9990 | 0.9863 | 14.8 |
| offset_bf16 | k256 | degrades | -0.2230 [-0.2327, -0.2133] | 0.223 -> 0.000 | 0.9997 | 0.9963 | 15.1 |
| offset_bf16 | k1024 | degrades | -0.2428 [-0.2532, -0.2328] | 0.243 -> 0.000 | 1.0000 | 1.0000 | 16.9 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy changes at k16, k64, k256, k1024 -> recompute/reset positions
- left-pad with naive absolute position ids: accuracy changes at p16, p64, p256, p1024 -> recompute/reset positions
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy changes at block16, block64, block256, block1024 -> recompute/reset positions
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy changes at k16, k64, k256, k1024 -> recompute/reset positions
