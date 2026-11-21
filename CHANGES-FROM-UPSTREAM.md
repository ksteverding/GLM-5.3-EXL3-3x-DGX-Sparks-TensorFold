# Changes from upstream

This repository is a fork: a modified copy of [MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold) at commit `eb6414b`, licensed under Apache 2.0. The upstream LICENSE, NOTICE and CREDITS are unchanged. The upstream README is unchanged except for a fork notice at the top.

## Added

- `scripts/dual-128k-fp4.sh`: a preset for two parallel requests with 131,072 tokens each (FP4 KV cache, Q4 dense weights, DSpark drafter). Large prompt chunks (3,072 rows) when one request is alone. While another reply is being written, prompts are read in 1,024-row pieces that get half of the GPU turns.
- `docs/dual-128k-fp4.md`: how to run the preset, its memory use, and measured results.
- `patches/0142-glm-full-cp-dense-prefix.patch`: during prompt reading, skips index scores for the dense prefix of a chunk that crosses the dense limit.
- `patches/0143-glm-full-multi-replay-borrow-home-cache.patch`: scheduler features the preset uses:
  - a fully cached prompt can start before a fresh prompt is read;
  - optional borrowing of spare draft rows between streams (`TF_GLM_MULTI_DRAFT_BORROW`);
  - optional moving of kept prompt caches so a lone stream can use its faster graphs (`TF_GLM_MULTI_KEEP_HOME_CACHE`);
  - new `/health` counters for these features.
- `.gitignore` for the machine-local settings files.

## Modified

- `start.sh`: the memory plan mounts the Hugging Face cache at its own path, so the snapshot's symlinks resolve. `KEEP_OLD_PCACHES=1` keeps prompt caches made by other images.
- `scripts/prepare.sh`: the checkpoint check mounts the Hugging Face cache the same way.
