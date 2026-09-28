# Multi-GPU adaptive experts (experimental)

Strata can use additional NVIDIA GPUs as **complementary adaptive expert caches**. Each GPU computes its own resident experts; the engine transfers activations and result vectors, not expert weights on every invocation. One process retains one shared host expert arena. This is not a replicated model or a transparent pool of GPU memory.

**Validation status:** the host policy/setup tests pass and the new C++ host code has been syntax-checked. CUDA compilation, the supplied GPU integration test, end-to-end model parity, Windows execution, and performance on real multi-GPU hardware still need validation. Do not treat this as a benchmarked release. No multi-GPU speedup or new maximum context is promised.

## Upstream baseline

This implementation is integrated through upstream **commit `b742ff9` (engine 0.1.12)**, 50 commits after the original v0.1.2 checkout. It retains upstream's conversation checkpoints, short-prompt verification path, sampling, suffix drafting, IQ3_S support and corrected output-head/cache allocation order. It also includes the CPU-pool late-wakeup race fix (particularly important with high GPU expert coverage), request watchdog, restart-on-engine-failure handling and rebuild-on-source-change setup logic. Automatic rebuilds preserve the previously built CUDA architecture set.

Upstream also adds optional RAM-backed KV streaming and Hadamard-rotated Q4_0 **KV storage**. Those are separate from four-bit model weights. For multiple selected GPUs, setup keeps KV on the primary by default; `--kv-resident` opts into streaming. The upstream experimental speed projection remains **off by default**; no control vector is needed for multi-GPU operation.

## What is implemented

- `--devices 0,1` selects CUDA-visible devices; the **first** runs dense operations, attention, context state, the output head and MTP. Other selected devices compute cached routed experts during verification windows (including one-token windows).
- Device selection is relative to `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER`. Setup verifies CUDA ordinals against physical GPU UUIDs rather than assuming `nvidia-smi` index order.
- Each additional device budgets its cache from **its own** free VRAM, after allocating worker buffers and subtracting a reserve. There are no GPU-model names, equal-size assumptions or two-device arrays in the runtime.
- The startup profile is a ranking hint, not a cache-size ceiling. Its original 8,000 pairs are extended with a deterministic, layer-interleaved tail covering all 24,576 experts.
- Secondary caches exclude primary and previously placed secondary experts. Their weights stay local; bounded pinned-host buffers carry activations and results. No P2P support, NVLink, NCCL, special driver patches, or Linux-only communication library is required.
- CPU misses, primary-GPU expert work, and secondary-GPU work overlap. Remote results retain their token/router row, so the existing primary weighted combination applies each router weight once.
- One coordinator adapts all selected tiers using decayed routing counts. CPU-only hot experts replace colder same-layer residents; same-layer replacement preserves variable-size slot capacity. Count and byte budgets limit uploads. An incoming expert becomes visible only after its upload event completes; placement stays fixed within each verification window.
- Existing single-GPU configurations remain valid. Explicit device selection works with one GPU as well. CPU AVX2/AVX-512 dispatch is retained.
- The engine publishes selected GPU UUIDs for monitoring. Existing scalar Monitor charts track the primary; `/metrics` additionally exposes every selected GPU in `hardware.gpus`, rather than confusing CUDA-visible ordinals with NVML physical indices.

The runtime is generalized over the number of selected devices; **two GPUs are the initial integration-test target**, not a claim that N-GPU scaling has been measured. Cache startup fills devices in selection order. Dynamic migration/replication of already-GPU-resident experts for load balancing is not implemented yet.

## Start on Linux

Initially use the existing, validated model family and a conservative context:

```bash
./setup.sh --setup --devices 0,1 --model IQ3_XXS --context 131072 --kv int8 --vision no
```

Setup compiles the current sources for all selected CUDA architectures. An older downloadable engine does not understand these options, so **explicit device/cache selection requires a source build**. Subsequent normal launches reuse the saved configuration:

```bash
./setup.sh
```

Windows uses the same options with `START-HERE.bat`. The implementation preserves Windows build/transfer paths, but its multi-GPU execution must still be tested there.

For a controlled high-context experiment:

```bash
./setup.sh --setup --devices 0,1 --model IQ3_XXS --context 262144 --allow-high-context --vision no
```

`--allow-high-context` is a **setup-only override of the conservative IQ3_XXS/IQ3_S RAM safeguard**, not a memory optimization or certification that the request fits. Watch RAM, swap and GPU memory. The context allocation includes prompt and generated tokens; serving also reserves a small verification guard.

Upstream's higher-precision **IQ3_S** is another candidate (`--model IQ3_S`, original Qwen family only). It retains roughly **50.3 GB** of experts in host RAM, versus 42.9 GB for IQ3_XXS, so on 64 GB start conservatively and leave other applications closed. It is not a full four-bit quantization, nor is its multi-GPU/long-context behavior already validated.

### Runtime options

These options go to `strata` directly or into the `args` array in the generated `strata-<model>.json`:

| Option | Meaning |
| --- | --- |
| `--devices 0` | Default, one CUDA-visible primary device. |
| `--devices 1,0` | Visible device 1 is primary; visible device 0 holds complementary experts. |
| `--list-devices` | Report visible devices, architectures and free memory; no model needed. |
| `--expert-cache auto` | Existing primary-cache sizing, now not capped to the profile's 8,000 entries. |
| `--vram-reserve-mib N` | Existing primary-GPU reserve. |
| `--expert-vram-mib N` | Cache cap per secondary GPU; **0 = auto** (default). Does not cap primary cache size. |
| `--expert-reserve-mib N` | Free VRAM margin per secondary, after its worker buffers; default **1024 MiB**. |
| `--expert-adapt-mib N` | Total upload payload budget per multi-GPU adaptation round; default **128 MiB**. 0 disables admissions. |
| `--adapt-every N` | Existing adaptation interval; default 4 verification rounds, 0 disables adaptation. |
| `--adapt-swaps N` | Existing maximum admissions per adaptation; default 96. |

Setup exposes `--devices`, `--expert-vram-mib`, `--expert-reserve-mib`, `--kv` and `--kv-resident`; set the remaining engine options in the generated configuration or invoke the engine directly. Budget from genuinely free memory, not the cards' advertised capacity. Display applications and the optional vision process consume memory too. Setup pins GPU vision to the selected primary GPU's UUID without changing the engine's ordinal mapping. Re-run setup if changing that mapping and using vision.

Multiple selected devices require the normal captured native/speculative path: `--native`, `--spec 2..8`, `--prefill`, `--expert-profile`, and `--expert-cache auto|N`. Setup supplies these. Diagnostic graph-only, no-pool, uncaptured, per-layer admission and layer/half-dump modes are refused with multiple devices rather than silently ignoring them.

## Important limits

### Prefill and state remain on the primary

The **batched** prompt path still uses its existing primary-GPU cuBLAS implementation. It streams nonlocal experts from the shared host arena and may temporarily borrow primary cache slots. Remote slots are **never** exposed as local GPU addresses. Pending adaptive uploads are drained before prompt processing; lent primary slots are restored before verification resumes.

Upstream now handles short prompt segments through verification windows: **those segments also use the secondary expert workers**. Conversation checkpoints are retained, and compatible continuing requests reuse state. Batched prompt processing is not yet distributed; it does not promise faster long-prompt ingestion. Distributed grouped prefill, peer-to-peer transport and layer/state partitioning are separate follow-ups.

### Context and RAM

Extra GPU expert capacity reduces competition with context storage on the primary. With INT8 KV storage, the main context-dependent KV/scales/indexer/RoPE allocations are approximately 1.77 GiB at 131,072 cells and 3.53 GiB at 262,144 cells. These figures exclude dense weights, recurrent state, MTP, workspaces, cache slots and safety margins.

The complete host expert arena remains allocated: roughly 34/35.5/43/50 GB for Q2_0/IQ2_XS/IQ3_XXS/IQ3_S. Additional GPU caches do **not** subtract these bytes from system RAM. Conversation checkpoints also consume RAM (upstream defaults to six, approximately 118 MB each). IQ3 at 262K on 64 GB remains experimental. Additional GPU memory does not implement RoPE/context extension beyond the repository's validated 262,144-token window.

To explicitly enable upstream KV streaming use `--kv-resident 32768` in setup or the engine; `--kv-resident 0` keeps the full KV on the primary. Streaming adds a host KV backing store, and it is therefore not enabled automatically for multi-GPU setups. `--kv q4_0` reduces KV memory but changes precision; keep `--kv int8` as the quality-focused baseline.

### Higher-precision quantizations

Dense kernel support is not equivalent to whole-model support. Startup validates native expert gate/up and down types against the actual grouped-expert dispatcher and refuses unsupported formats before allocating model weights. Upstream's IQ3_S adds IQ4_XS grouped gate/up support, which the multi-GPU worker preserves. This still does **not establish compatibility with arbitrary four-bit GGUFs**: all their expert, dense, embedding and PLE formats and geometries must be supported. Q4_K routed expert support is not added by this change. No automatic lower-precision requantization is performed.

A future four-bit extension needs tensor-by-tensor format validation, matching decode/prefill/embedding/PLE kernels, parity tests, and a host-memory plan. The routed experts alone contain about 120.8B parameters: nominal four-bit storage is 56.25 GiB **before** scales and overhead. On a 64 GB machine a bounded host backing/cache design may be required; the current resident arena must not be assumed sufficient.

## Build and tests

CUDA kernels require compute capability **8.0 or newer**. A local build defaults to CMake's `native` architectures; an explicit list overrides it. Redistributable builds should specify every supported architecture and use a toolkit that supports that list. For example, with CUDA 13:

```bash
cmake -S . -B build-multi \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_BUILD_MULTI_GPU_TESTS=ON \
  -DSTRATA_PORTABLE=ON \
  '-DCMAKE_CUDA_ARCHITECTURES=86;89;120'
cmake --build build-multi --parallel --target strata expert_placement_test multi_gpu_test pool_stress
```

For a local 3090-only build, `-DCMAKE_CUDA_ARCHITECTURES=86` is enough; this is a build selection, not a source-code requirement. On Windows add `--config Release` to build commands and use the corresponding executable directory if using a multi-config generator. `STRATA_PORTABLE` chooses an AVX2 ggml baseline with runtime-selected Strata CPU kernels. Setup uses its own `build-setup/` tree and does not overwrite manually configured `build/` caches.

The older `STRATA_BUILD_TESTS` option references tests omitted from this published repository. Leave it **OFF**; the new tests are self-contained and independent of that suite.

### Host-only tests (no CUDA/model)

```bash
python3 -B -m unittest discover -s tools/tests -v
c++ -std=c++20 -DNDEBUG -Wall -Wextra -Werror -Iinclude \
  src/plan/expert_placement_test.cpp -o /tmp/strata-expert-placement-test
/tmp/strata-expert-placement-test
```

The C++ test covers device-list validation, profile completion, heterogeneous byte budgets, nonduplicated placement, in-flight admissions, hysteresis and bounded adaptive replacement. The Python tests mock hardware and cover visible-device/UUID mappings, source-build architecture lists, old-config compatibility, vision placement and context overrides.

### Real GPU integration test (no model)

```bash
ctest --test-dir build-multi --output-on-failure -R '^(expert_placement|multi_gpu_experts|pool_stress)$'
# Or select an explicit ordered pair:
./build-multi/multi_gpu_test --devices 0,1
```

`multi_gpu_test` allocates actual expert blobs and compares nonzero secondary results against the same grouped kernels on the primary, including native IQ3/IQ2 and IQ4_XS variable-sized layers when `STRATA_NATIVE_EXPERTS=ON` (the default). It covers T=1..8, mixed routing, no remote hits, all-remote routing, per-token duplicate rejection, adaptive replacement, output guards and current-device restoration. Default execution reverses primary/secondary as well. Fewer than two visible GPUs yields CTest **SKIP** (exit 77), not a passing GPU result. This tests transport and residency; it is not an independent arithmetic oracle or an end-to-end model test.

### Required model validation before relying on it

1. Build and run the synthetic GPU test on the target operating system.
2. Record single-GPU and two-GPU runs using identical model, prompt, context, KV type and speculative settings. Test static placement first (`--adapt-every 0`), then adaptation.
3. Compare expert outputs/logits as well as greedy tokens. Moving work from CPU to GPU changes existing backend arithmetic; do not assume bitwise agreement merely because the transport copies bytes exactly. Upstream replaced its obsolete GPU-cache correctness warning with measured CPU/GPU rounding results in `bench/results/2026-09-27-cache-parity/README.md`. Those are single-GPU Q2_0 measurements, not validation of this multi-GPU path.
4. Force wrong drafts with the existing `--spec-corrupt` diagnostic and verify rollback/commit. Exercise one-token prompts, cancellation, repeated requests and prefill borrowing/refill.
5. Increase context progressively while observing peak host memory and VRAM. Test both GPU orderings and configurations with a small secondary cache, no remote hits and high remote coverage.
6. Report prompt throughput separately from decode throughput. The stderr log reports each secondary's cached expert count and cumulative dispatches, routed rows, activation/result bytes, host submission time, exposed completion wait and adaptive admissions. Host timings are not GPU kernel timings.

## Code map

- `include/strata/plan/expert_placement.hpp`: CUDA-independent placement and adaptive policy.
- `include/strata/core/multi_gpu.hpp`, `src/core/multi_gpu.cpp`: device-owned secondary workers and coordinated cache lifecycle.
- `src/core/expert_source.cpp`: partition routed work, overlap producers, scatter remote outputs into the existing verifier handoff.
- `src/program/generate.cpp`: device selection, startup sizing, serving/generation wiring and adaptation boundaries.
- `src/core/device.cu`: per-thread device scope and architecture-image startup probe.
- `setup.py`, `serve/server.py`, `serve/telemetry.py`: portable installation/configuration, separate vision placement and UUID-based GPU monitoring.
