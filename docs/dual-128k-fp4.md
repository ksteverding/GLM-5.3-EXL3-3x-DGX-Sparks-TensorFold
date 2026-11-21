# Two 128K FP4 slots on three DGX Sparks

This preset gives each of two simultaneous requests 131,072 tokens for its prompt and reply combined. It uses FP4 KV, Q4 dense weights, DSpark and grouped prefill. A full prompt chunk has 3,072 rows (PREFILL_ROWS; measured 2026-11-21: cold prompts ~15% faster than 1,024 rows, about 2.1 GiB more a rank, lowest free 10.9 GiB under load). While another request decodes, the scheduler processes 1,024 prompt rows per turn and gives prefill half of its turns (TF_GLM_FILL_ROWS, TF_GLM_FILL_SHARE; measured 2026-11-21 with a 27.6k-token prompt arriving beside a 2,500-token reply: 512 rows at 0.25 gave a first token after 79.8 s and both done at 129.7 s; 1,024 at 0.5 gave 53.5 s and 118.6 s with pauses under 2.3 s; 3,072 at 1 gave 43.5 s and 111.2 s but paused the other reply for 43 s). The preset enables retained-cache relocation and keeps draft borrowing, sliced prefill and NVMe prompt caching off. Admission uses a 16 GiB reserve, including a 6 GiB free-memory target. The runtime guard stops a rank below 3 GiB.

## Run on the head Spark

Use the usual worker and NFS settings in `scripts/local.sh` or `.env`. Build the image from this recipe's patches (0143 adds the dual-slot scheduler features) with `IMAGE=tensorfold-glm53-full:dual-128k-fp4 PULL=0 ./scripts/prepare.sh` (about the same steps as the main recipe; see the README).

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
