# Fork sync: upstream v0.1.32

Synced on 2026-10-01 with upstream `main` at `c499bd1` ([v0.1.32](https://github.com/Niko1221/Strata/releases/tag/v0.1.32)), bringing in all 221 upstream commits since v0.1.29. This includes conversation caching, improved split-prompt performance, expanded AMD HIP support, experimental Unsloth UD-Q4_K_XL setup, reproducible installer dependencies, and server lifecycle and API improvements. The merge retained the independent fork fixes below without conflicts.

This fork now uses upstream Strata's **layer-split multi-GPU engine** (`docs/MULTI_GPU.md`) and its setup and monitoring paths. The earlier fork-only `--devices` adaptive expert-cache implementation has been retired: it was not hardware-validated, and combining it with upstream's independently developed layer split and helper-GPU caches would give two conflicting owners for the same experts. The old implementation and its tests remain in Git history (`cebd414`), but are not built or shipped in this tree.

The upstream `--gpus 0,2` option uses **nvidia-smi physical GPU numbers**, not the old `--devices 0,1` CUDA-visible ordinals. Do not translate those numbers by hand, especially if `CUDA_VISIBLE_DEVICES` was set. For an existing fork installation, close the server and re-run setup with `--setup --gpus <physical-indices>` (or let setup offer the supported cards). If an old `strata-*.json` still contains `--devices`, regenerate that config; v0.1.24 and later engines do not accept the old flag. Model files in `Strata-data` are reusable.

Upstream's layer split, its separate experimental helper caches (`docs/SECOND_GPU.md`), prefill, setup and telemetry take precedence. We retain only a few independent fork fixes: CMake compatibility with modern `FetchContent`, a valid position-zero native prompt in the CLI, cleanup of a device-owned GPU arena even if another GPU is current, and early rejection of retired `--devices` server configs with migration instructions.

## Sync validation (macOS, host-only)

- Python 3.12: 292 tests passed across the server and tools suites (301 discovered); two Windows-only tests, five tokenizer-dependent tests and two optional `jsonschema` tests were skipped. The packing tests used the existing external llama.cpp `gguf-py` checkout via `STRATA_GGUF_PY`.
- CMake host-only build: all ten selected tests passed (`gguf_reader_test`, `gguf_split_test`, `suffix_drafter_test`, `controller_test`, `draft_policy_test`, `conv_cache_test`, `coupled_draft_test`, `bf16_bits_test`, `conversation_cache_test`, `conversation_memory_test`). CUDA, HIP and native experts were disabled for this build; four x86 CPU expert/pool tests were excluded on this arm64 Mac.
- The `strata-plan` memory-planning smoke checks (valid budget accepted, oversized context rejected), syntax checks of all 67 tracked Python files, and `git diff --check` passed.

CUDA/HIP compilation, real-model inference and multi-GPU performance still need validation on supported hardware; the host-only tests cannot establish those.
