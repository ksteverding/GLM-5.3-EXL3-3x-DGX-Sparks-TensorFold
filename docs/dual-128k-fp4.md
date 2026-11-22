# Two 128K FP4 slots on three DGX Sparks

This preset gives each of two simultaneous requests 131,072 tokens for its prompt and reply combined. It uses FP4 KV, Q4 dense weights, DSpark and grouped prefill. A full prompt chunk has 3,072 rows (PREFILL_ROWS; measured 2026-11-21: cold prompts ~15% faster than 1,024 rows, about 2.1 GiB more a rank, lowest free 10.9 GiB under load). While another request decodes, the scheduler processes 1,024 prompt rows per turn and gives prefill half of its turns (TF_GLM_FILL_ROWS, TF_GLM_FILL_SHARE; measured 2026-11-21 with a 27.6k-token prompt arriving beside a 2,500-token reply: 512 rows at 0.25 gave a first token after 79.8 s and both done at 129.7 s; 1,024 at 0.5 gave 53.5 s and 118.6 s with pauses under 2.3 s; 3,072 at 1 gave 43.5 s and 111.2 s but paused the other reply for 43 s). The preset enables retained-cache relocation and keeps draft borrowing, sliced prefill and NVMe prompt caching off. Admission uses a 16 GiB reserve, including a 6 GiB free-memory target. The runtime guard stops a rank below 3 GiB.

The preset also sets (2026-11-22; each can be overridden from the environment):

- `TF_GLM_MAX_ROWS=24` and `TF_GLM_MULTI_ROWS=24` (0146): verify windows of up to 24 rows, for one stream and for both streams together (16 before; with two streams each one drafted at most 7 tokens, now 11). Copy drafts can run longer. `TF_ROCE_MAX_KB=512` keeps these larger all-gathers on RoCE (16 KiB a row) instead of NCCL. `tools/exact.py` gave 12/12: replies still equal their serial references.
- `TF_GLM_CACHE_ENTRIES=24` (8 before): more kept prompts. With 8, one burst of other requests pushed out a shared system prompt, and the next agent turn read 19k tokens again (40-70 s instead of under 1 s). Costs about 6k tokens of the pool (266,240 instead of 272,384).
- `TF_GLM_EXPERT_COUNTS=1` (0144): counts which routed experts each layer uses, for pruning and quantization research. No measurable speed cost.

## Run on the head Spark

Use the usual worker and NFS settings in `scripts/local.sh` or `.env`. Build the image from this recipe's patches (0143 adds the dual-slot scheduler features; 0144-0146 the counters, the tool-call parser fix and the wider verify window; the check refuses an image without 0146) with `IMAGE=tensorfold-glm53-full:dual-128k-fp4 PULL=0 ./scripts/prepare.sh` (about the same steps as the main recipe; see the README).

```bash
export IMAGE=tensorfold-glm53-full:dual-128k-fp4
export PORT=8000 MASTER_PORT=29647
./scripts/dual-128k-fp4.sh check
DRY_RUN=1 ./scripts/dual-128k-fp4.sh start
```

The image check uses disposable CPU containers without network or mounts. It checks the replay and relocation features and compares the image patch label and engine and scheduler source hashes across the three ranks. The dry run uses the recipe's regular link, argument and memory checks. A running service's memory is counted in that preview. Once your existing service has drained and been stopped through its own rollout procedure, run `./scripts/dual-128k-fp4.sh start`. Check that `/health` reports `context_length: 131072`, `streams.max: 2`, `multi_prefill.replay_overtakes` and `home_cache`.

The preset owns containers named `glm53-full-tf-dual-128k-fp4` and state in `~/.local/state/glm53-full-tensorfold-dual-128k-fp4`. It uses the site's configured API and rendezvous ports, so use those same settings for its entire lifecycle. The wrapper preserves existing on-disk prompt caches and skips automatic image builds. Plain `./start.sh` keeps the recipe defaults.

Drain the preset's requests before running `./scripts/dual-128k-fp4.sh stop`. Use the original service's rollout procedure to restore it. The wrapper's `restart` and `stop` commands target only the preset containers.

## Keep an interactive request responsive

Leave the interactive client's default priority in place. Add the top-level JSON field `"priority": "background"` to local TensorFold requests from a bulk coding worker when it can yield. When both slots already held 256-token coding jobs, a simulated tool turn completed in 2.78 seconds with this setting on one job, versus 14.48 seconds with both jobs at default priority. Both coding jobs together took 15.60 versus 14.30 seconds. They returned the same 256 token IDs per job in both runs. This comparison used two fully cached 1,024-token coding prompts and one foreground tool turn per setting.

## Measured speed (2026-11-22)

An agent-like load: a shared 19,171-token code prompt, code replies of 700 tokens, two turns per conversation (the second turn reuses the first turn's cache), temperature 0.6 with a fixed seed. "Total" is all reply tokens divided by wall time, prompt reading included.

| Setup | 1 stream decode | 2 streams total | 3 requests total | Cold 96k prompt |
|---|---|---|---|---|
| Before: 16 rows, 8 kept prompts | 28.3 tok/s | 37.2 tok/s | – | 168 s |
| Preset: 24 rows, 24 kept prompts | 28.3 tok/s | 37.7 tok/s | 39.7 tok/s (the third waits) | 168 s |
| Tried: 3 slots of 88k, 24 rows | 28.3 tok/s | 38.2 tok/s | 32.6 tok/s (one cache miss; ~44 tok/s decode alone) | – |

A third slot added only about 6% total decode while each stream dropped from 20.8 to 14.7 tok/s, and it shrinks every window to 88k. Two slots with background priority for bulk work serve one agent plus a coding client better.

Not changed, with the reason:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: over an 8.5-hour run (136 requests, 7.3M prompt tokens) the lowest free memory stayed at 10.8 GiB, the same as right after loading. There is no leak to fix, and the setting changes how GPU memory is mapped for the RoCE transport.
- `KV=fp4x`: only needed for a third 128k slot (see above). On the README's quality suite it was not significantly worse than FP4, but leaned toward FP4 on ledger tracking.
- Keeping a waiting agent's cache: the scheduler already does this. Kept prompts stay in the pool, and eviction drops a conversation's earlier turns before the only state of a conversation that waits on a tool (`cap_victim`). More entries (above) were the missing piece.
- Suffix-style drafting: copy drafts already propose up to 15 tokens from matching text in the prompt (8+ tokens) and the reply (24+), checked by DSpark. The wider window lets them run longer under two streams.

## Hermes Agent with two slots

Use a separate Hermes profile so the default profile keeps its own model:

```bash
hermes profile create spark --no-alias
hermes -p spark config set model.provider custom
hermes -p spark config set model.default GLM-5.3-EXL3
hermes -p spark config set model.base_url http://<head-spark>:8000/v1
hermes -p spark config set model.api_key local
hermes -p spark config set model.context_length 131072
hermes -p spark config set compression.micro_compact false
hermes -p spark config set delegation.max_concurrent_children 2
hermes -p spark config set delegation.request_overrides '{"extra_body": {"priority": "background"}}'
```

- `micro_compact` stays off (the default): it rewrites sent history every turn and breaks the prompt cache.
- `max_concurrent_children: 2` instead of 10: the server decodes two requests at once and queues the rest. A batch larger than this limit is a tool error in Hermes, so do not set it to 1.
- `request_overrides.extra_body.priority: background` marks every subagent request as background, so a coding client's requests go first (see above). The main agent and its compression calls keep the default priority, because the agent waits for them. Session-title requests are background on the server already.
