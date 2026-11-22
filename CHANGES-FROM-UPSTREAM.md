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
- `patches/0144-glm-full-expert-counts.patch`: optional counters of which routed experts each layer picks (`TF_GLM_EXPERT_COUNTS=1`), written as totals and hourly snapshots. Counts only, no request text.
- `patches/0145-glm-tool-call-stray-tags.patch`: a `<tool_call>` the model writes as text (a mention in its reasoning or answer) no longer swallows the real tool call after it. Before, the reply came back as plain text with the raw call inside and finish reason `stop`, so agent clients ended the task.
- `patches/0146-glm-full-multi-rows.patch`: `TF_GLM_MULTI_ROWS` sets the parallel verify window (16 by default, up to `TF_GLM_MAX_ROWS`).
- `tests/cpu/test_expert_counts.py`, `tests/cpu/test_tool_call_stray_tags.py`: CPU tests for 0144 and 0145.
- `patches/0147-glm-full-multi-disk-cache.patch`: the NVMe prompt cache (`TF_GLM_DISK_CACHE`) also under `--parallel` for full GLM-5.3: kept prompts are written to each Spark's disk in the background, and a prompt that left the pool (evicted, or lost in a restart) loads again in about half a second instead of being read again (a 92k prompt: 0.55 s instead of 157 s). Replies stay bit-exact. CPU tests in `tests/cpu/test_full_multi_disk.py`; the preset turns it on with `DISK_CACHE=1`.
- `.gitignore` for the machine-local settings files.

## Modified

- `start.sh`: the memory plan mounts the Hugging Face cache at its own path, so the snapshot's symlinks resolve. `KEEP_OLD_PCACHES=1` keeps prompt caches made by other images.
- `scripts/prepare.sh`: the checkpoint check mounts the Hugging Face cache the same way.
