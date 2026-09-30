# Fork sync: upstream v0.1.29

Synced on 2026-09-30 with upstream `main` at `d6708a4` ([v0.1.29](https://github.com/Niko1221/Strata/releases/tag/v0.1.29)), bringing in all 67 upstream commits since v0.1.24. This includes the experimental AMD HIP backend, RTX 20 support, low-RAM mode, faster prompt and sampling kernels, and server and engine correctness fixes. The merge retained the independent fork fixes below without conflicts.

This fork now uses upstream Strata's **layer-split multi-GPU engine** (`docs/MULTI_GPU.md`) and its setup and monitoring paths. The earlier fork-only `--devices` adaptive expert-cache implementation has been retired: it was not hardware-validated, and combining it with upstream's independently developed layer split and helper-GPU caches would give two conflicting owners for the same experts. The old implementation and its tests remain in Git history (`cebd414`), but are not built or shipped in this tree.

The upstream `--gpus 0,2` option uses **nvidia-smi physical GPU numbers**, not the old `--devices 0,1` CUDA-visible ordinals. Do not translate those numbers by hand, especially if `CUDA_VISIBLE_DEVICES` was set. For an existing fork installation, close the server and re-run setup with `--setup --gpus <physical-indices>` (or let setup offer the supported cards). If an old `strata-*.json` still contains `--devices`, regenerate that config; v0.1.24 and later engines do not accept the old flag. Model files in `Strata-data` are reusable.

Upstream's layer split, its separate experimental helper caches (`docs/SECOND_GPU.md`), prefill, setup and telemetry take precedence. We retain only a few independent fork fixes: CMake compatibility with modern `FetchContent`, a valid position-zero native prompt in the CLI, cleanup of a device-owned GPU arena even if another GPU is current, and early rejection of retired `--devices` server configs with migration instructions.

## Sync validation (macOS, host-only)

- Python 3.12: 92 tests passed across the server, MCP, calibration, shard and packing suites; two Windows-only tests and three tokenizer-dependent tests were skipped.
- CMake host-only build: all six selected tests passed (`gguf_reader_test`, `suffix_drafter_test`, `controller_test`, `draft_policy_test`, `conv_cache_test`, `bf16_bits_test`). CUDA, HIP and native experts were disabled for this build.
- The `strata-plan` memory-planning smoke check, syntax checks of all 37 tracked Python files, and `git diff --check` passed.

CUDA/HIP compilation, real-model inference and multi-GPU performance still need validation on supported hardware; the host-only tests cannot establish those.
