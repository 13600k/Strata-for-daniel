// Synthetic integration test for the REAL MultiGpuExperts transport, caches and grouped CUDA kernels.
// No model/pack, Catch2, P2P or NCCL. Requires >=2 CUDA GPUs; fewer GPUs returns SKIP (77).
// By default tests 0->1 and 1->0. --devices 2,0 tests that ordered pair only (CUDA-visible ordinals).
// Link with strata_engine and CUDA::cudart (transitively strata_core, strata_kernels, strata_kernels_cpu).
// Register with CTest SKIP_RETURN_CODE 77. This test is not a CPU or stubbed-CUDA simulation.
// Canonical Q2 always runs; STRATA_NATIVE_EXPERTS additionally enables a temporary IQ3/IQ2 manifest.
// Native builds use ggml.h through strata_kernels_cpu's PUBLIC ggml-base dependency (no ggml-common.h).
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/multi_gpu.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#if defined(STRATA_NATIVE_EXPERTS)
#include "ggml.h"
#endif

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace c = strata::kernels::cpu;
namespace k = strata::kernels;
namespace core = strata::core;
namespace p = strata::plan;

namespace {
constexpr int LAYERS = 2, EXPERTS = 16, TOPK = 10, MAX_ROWS = c::MAXT * TOPK;
constexpr size_t GUARD = 16;
int cases = 0, rejected_cases = 0, cleanup_failures = 0;

// Exceptions unwind all CUDA/cache/source owners. Do not replace these with std::exit or assert.
void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}
void cu(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}
void api(bool ok, const char* what, const std::string& err) {
    if (!ok) throw std::runtime_error(std::string(what) + ": " + err);
}
void current_is(int expected, const char* operation) {
    int device = -1;
    cu(cudaGetDevice(&device), "cudaGetDevice");
    require(device == expected, std::string(operation) + " did not restore current device " + std::to_string(expected));
}
void cleanup(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "FAIL cleanup %s: %s\n", what, cudaGetErrorString(status));
        ++cleanup_failures;
    }
}

// All test allocations have an owning device and drain before destruction, including on an exception.
class Buffers {
public:
    explicit Buffers(int device) : device_(device) {
        current_is(device_, "test buffer setup");
        cu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "test stream creation");
    }
    ~Buffers() {
        int previous = -1;
        cleanup(cudaGetDevice(&previous), "get previous device");
        const auto status = cudaSetDevice(device_);
        cleanup(status, "set buffer owner");
        if (status == cudaSuccess) {
            cleanup(cudaStreamSynchronize(stream), "drain test stream");
            for (void* ptr : allocations_) cleanup(cudaFree(ptr), "free test allocation");
            cleanup(cudaStreamDestroy(stream), "destroy test stream");
        }
        if (previous >= 0) cleanup(cudaSetDevice(previous), "restore buffer caller device");
    }
    Buffers(const Buffers&) = delete;
    Buffers& operator=(const Buffers&) = delete;
    template <typename T> T* alloc(size_t count) {
        T* ptr = nullptr;
        cu(cudaMalloc((void**) &ptr, count * sizeof(T)), "test buffer allocation");
        try { allocations_.push_back(ptr); }
        catch (...) { (void) cudaFree(ptr); throw; }
        return ptr;
    }
    cudaStream_t stream = nullptr;
private:
    int device_;
    std::vector<void*> allocations_;
};

// Runtime-only fixture files. Atomic directory creation avoids collisions between concurrent test runs.
class TempDirectory {
public:
    TempDirectory() {
        const auto root = std::filesystem::temp_directory_path();
        std::mt19937_64 rng((uint64_t) std::chrono::steady_clock::now().time_since_epoch().count() ^
                            (uint64_t) std::random_device{}());
        for (int attempt = 0; attempt < 100; ++attempt) {
            path_ = root / ("strata-multi-gpu-test-" + std::to_string(rng()));
            if (std::filesystem::create_directory(path_)) return;
        }
        throw std::runtime_error("cannot create unique multi-GPU fixture directory");
    }
    ~TempDirectory() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
        if (ec) {
            std::fprintf(stderr, "FAIL cleanup fixture directory: %s\n", ec.message().c_str());
            ++cleanup_failures;
        }
    }
    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;
    const std::filesystem::path& path() const { return path_; }
private:
    std::filesystem::path path_;
};

// The layout is process-global. Keep this owner outside all sources, caches and CUDA references, and
// reset it to canonical only after those owners are destroyed. The directory member also cleans up if
// manifest construction/loading throws. No experts.bin or real model is needed.
class TestLayout {
public:
    explicit TestLayout(bool native, bool iq4 = false) {
        (void) iq4; // native formats are optional at build time
        canonical_dir_ = directory_.path() / "canonical";
        require(std::filesystem::create_directory(canonical_dir_), "create empty canonical fixture directory");
        std::string err;
        if (native) {
#if defined(STRATA_NATIVE_EXPERTS)
            require(c::native_experts_available(), "native build must provide ggml-cpu expert traits");
            const int gu[LAYERS] = {iq4 ? 23 : 18, 17}; // IQ4_XS (IQ3_S model), IQ3_XXS, IQ2_XS
            const int down[LAYERS] = {42, 20}; // Q2_0, IQ4_NL
            std::ofstream manifest(directory_.path() / "native_experts.txt");
            require((bool) manifest, "create native fixture manifest");
            manifest << "# strata native experts v3 (n_expert " << EXPERTS << ")\n";
            uint64_t offset = 0;
            for (int l = 0; l < LAYERS; ++l) {
                c::NativeFmt f;
                api(c::native_fmt(gu[l], down[l], c::H, c::FF, f, err), "native fixture traits", err);
                const auto gpu = k::native_expert_layout(f.gu_type, f.d_type, c::H, c::FF);
                require(gpu.gu_row == f.gu_row && gpu.d_row == f.d_row && gpu.up_off == f.up_off &&
                            gpu.down_off == f.down_off && gpu.bytes == f.bytes, "CPU/GPU native fixture layout agreement");
                manifest << l << ' ' << f.gu_type << ' ' << f.d_type << ' ' << offset << ' ' << f.bytes << '\n';
                offset += (uint64_t) EXPERTS * f.bytes;
            }
            manifest.close();
            require((bool) manifest, "write native fixture manifest");
#else
            throw std::runtime_error("native fixture requested without STRATA_NATIVE_EXPERTS");
#endif
        }
        // Native packs override the canonical count from their header, as pruned Coder packs do.
        api(c::expert_layout_load(directory_.path().string(), LAYERS, native ? c::NE : EXPERTS, err),
            "load synthetic layout", err);
        require(c::expert_layout().n_expert == EXPERTS, "native header overrides caller's canonical expert count");
        require(c::expert_layout().native == native, "synthetic layout mode");
        if (native) require(c::expert_layout().blob_bytes(0) != c::expert_layout().blob_bytes(1),
                            "native fixture must exercise unequal layer payloads");
    }
    ~TestLayout() {
        try {
            std::string err;
            api(c::expert_layout_load(canonical_dir_.string(), LAYERS, EXPERTS, err), "reset synthetic layout", err);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "FAIL cleanup layout: %s\n", e.what());
            ++cleanup_failures;
        }
    }
    TestLayout(const TestLayout&) = delete;
    TestLayout& operator=(const TestLayout&) = delete;
private:
    TempDirectory directory_;
    std::filesystem::path canonical_dir_;
};

#if defined(STRATA_NATIVE_EXPERTS)
void native_matrix(uint8_t* dst, int type, int rows, int cols, size_t row_bytes, std::mt19937& rng) {
    // All bits after d are legal packed codes/signs/subscales for these four types. Only the leading
    // fp16 scale needs constraining. Get block sizes from ggml, never guess IQ2_XS's byte stride.
    require(type == 18 || type == 17 || type == 23 || type == 42 || type == 20, "fixture type has leading fp16 scale");
    const auto t = (ggml_type) type;
    const int64_t block_values = ggml_blck_size(t);
    const size_t block_bytes = ggml_type_size(t);
    require(block_values > 0 && cols % block_values == 0 && block_bytes > sizeof(uint16_t), "native block geometry");
    require(row_bytes == (size_t) (cols / block_values) * block_bytes &&
                row_bytes == ggml_row_size(t, cols) && row_bytes == k::iq_row_bytes(type, cols), "native row byte agreement");
    const size_t blocks = (size_t) rows * (size_t) (cols / block_values);
    for (size_t block = 0; block < blocks; ++block) {
        uint8_t* b = dst + block * block_bytes;
        const uint16_t scale = (uint16_t) (0x1800 + rng() % 0x0400); // small, positive and finite
        std::memcpy(b, &scale, sizeof(scale));
        for (size_t i = sizeof(scale); i < block_bytes; ++i) b[i] = (uint8_t) rng();
    }
}
#endif

class SyntheticSource : public core::ExpertSource {
public:
    SyntheticSource() : layout_(c::expert_layout()), bytes_((size_t) layout_.total) {
        require(layout_.n_layers == LAYERS && layout_.n_expert == EXPERTS, "fixture layout geometry");
        std::mt19937 rng(1977);
        for (int l = 0; l < LAYERS; ++l) for (int e = 0; e < EXPERTS; ++e) {
            uint8_t* b = bytes_.data() + offset(l, e);
            if (layout_.native) {
#if defined(STRATA_NATIVE_EXPERTS)
                const auto& f = layout_.fmt.at((size_t) l);
                native_matrix(b, f.gu_type, c::FF, c::H, f.gu_row, rng);
                native_matrix(b + f.up_off, f.gu_type, c::FF, c::H, f.gu_row, rng);
                native_matrix(b + f.down_off, f.d_type, c::H, c::FF, f.d_row, rng);
#else
                throw std::runtime_error("native source requested without STRATA_NATIVE_EXPERTS");
#endif
            } else {
                // expert.hpp canonical layout: interleaved gate/up codes, down codes, then fp16 scale planes.
                // Every code byte holds four legal 2-bit codes. Scales are finite positive FP16 bit patterns;
                // randomizing scale bytes indiscriminately would manufacture NaNs, not a valid Q2 fixture.
                for (size_t i = c::O_GU_CODES; i < c::O_GU_SCALES; ++i) b[i] = (uint8_t) rng();
                for (size_t i = c::O_GU_SCALES; i < c::BLOB; i += sizeof(uint16_t)) {
                    const uint16_t scale = (uint16_t) (0x1c00 + rng() % 0x0800);
                    std::memcpy(b + i, &scale, sizeof(scale));
                }
            }
        }
    }
    const uint8_t* blob(int64_t layer, int64_t expert) override {
        if (layer < 0 || layer >= LAYERS || expert < 0 || expert >= EXPERTS) return nullptr;
        return bytes_.data() + offset((int) layer, (int) expert);
    }
    size_t offset(int layer, int expert) const { return (size_t) layout_.blob_offset(layer, expert); }
    size_t layer_bytes(int layer) const { return (size_t) layout_.blob_bytes(layer); }
    uint64_t payload_budget(int per_layer) const {
        uint64_t total = 0;
        for (int l = 0; l < LAYERS; ++l) total += (uint64_t) per_layer * layer_bytes(l);
        return total;
    }
    uint64_t slot_budget(int per_layer) const {
        uint64_t total = 0;
        for (int l = 0; l < LAYERS; ++l) total += (uint64_t) per_layer * p::expert_slot_bytes(layer_bytes(l));
        return total;
    }
    const c::ExpertLayout& layout() const { return layout_; }
    const std::vector<uint8_t>& bytes() const { return bytes_; }
private:
    c::ExpertLayout layout_;
    std::vector<uint8_t> bytes_;
};

// Independent destination plan on the primary: sort groups by expert ID and write directly to original
// token*TOPK+router rows. The remote path instead uses first-seen groups, compact output and a host scatter.
// Arithmetic is intentionally identical CUDA kernels; this is transport/placement parity, not an oracle
// for the kernels themselves. All blobs are separately uploaded, not borrowed from the cache under test.
class PrimaryReference {
public:
    PrimaryReference(int primary, const SyntheticSource& source) : mem_(primary), layout_(source.layout()) {
        blobs_ = mem_.alloc<uint8_t>(source.bytes().size());
        cu(cudaMemcpy(blobs_, source.bytes().data(), source.bytes().size(), cudaMemcpyHostToDevice), "reference blobs");
        x_ = mem_.alloc<float>((size_t) c::MAXT * c::H);
        xq_ = mem_.alloc<uint8_t>((size_t) c::MAXT * (c::H / 32) * 36);
        scales_ = mem_.alloc<float>((size_t) c::MAXT * (c::H / 32));
        output_ = mem_.alloc<float>((size_t) MAX_ROWS * c::H);
        scratch_ = mem_.alloc<uint8_t>((size_t) std::max<uint64_t>(
            k::moe_hit_grouped_scratch_bytes(MAX_ROWS, c::H, c::FF), k::native_expert_scratch_bytes(MAX_ROWS, c::FF)));
        ptr_ = mem_.alloc<unsigned long long>(MAX_ROWS);
        start_ = mem_.alloc<int32_t>(MAX_ROWS + 1);
        dst_ = mem_.alloc<int32_t>(MAX_ROWS);
        tok_ = mem_.alloc<int32_t>(MAX_ROWS);
        count_ = mem_.alloc<int32_t>(1);
    }
    std::vector<float> run(int layer, int tokens, const std::vector<float>& x, const std::vector<int32_t>& ids) {
        const int n = tokens * TOPK;
        std::vector<unsigned long long> ptr;
        std::vector<int32_t> start, dst, tok;
        for (int e = 0; e < EXPERTS; ++e) {
            const auto begin = dst.size();
            for (int i = 0; i < n; ++i) if (ids[(size_t) i] == e) {
                dst.push_back(i);
                tok.push_back(i / TOPK);
            }
            if (dst.size() != begin) {
                require(dst.size() - begin <= (size_t) c::MAXT, "fixture exceeds grouped kernel's eight-entry limit");
                start.push_back((int32_t) begin);
                ptr.push_back((unsigned long long) (blobs_ + layout_.blob_offset(layer, e)));
            }
        }
        require(dst.size() == ids.size(), "reference routing coverage");
        start.push_back(n);
        const int32_t count = (int32_t) ptr.size();
        auto upload = [&](void* d, const void* h, size_t bytes) {
            cu(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, mem_.stream), "reference metadata upload");
        };
        upload(x_, x.data(), x.size() * sizeof(float));
        upload(ptr_, ptr.data(), ptr.size() * sizeof(ptr[0]));
        upload(start_, start.data(), start.size() * sizeof(int32_t));
        upload(dst_, dst.data(), dst.size() * sizeof(int32_t));
        upload(tok_, tok.data(), tok.size() * sizeof(int32_t));
        upload(count_, &count, sizeof(count));
        cu(cudaMemsetAsync(output_, 0xff, (size_t) n * c::H * sizeof(float), mem_.stream), "poison reference output");
        if (layout_.native) {
            k::quantize_q8_1_rows(x_, tokens, c::H, xq_, mem_.stream);
            const auto& f = layout_.fmt.at((size_t) layer);
            const auto l = k::native_expert_layout(f.gu_type, f.d_type, c::H, c::FF);
            k::native_expert_grouped(l, ptr_, start_, count_, dst_, tok_, count, n, xq_, scratch_, output_, mem_.stream);
        } else {
            k::quantize_q8_0_scaled(x_, xq_, scales_, (int64_t) tokens * c::H, mem_.stream);
            k::moe_grouped_s2(ptr_, start_, count_, dst_, tok_, count, n, xq_, scales_, scratch_, output_, mem_.stream);
        }
        cu(cudaGetLastError(), "reference grouped kernels");
        std::vector<float> out((size_t) n * c::H);
        cu(cudaMemcpyAsync(out.data(), output_, out.size() * sizeof(float), cudaMemcpyDeviceToHost, mem_.stream),
           "reference output download");
        cu(cudaStreamSynchronize(mem_.stream), "reference completion");
        return out;
    }
private:
    Buffers mem_;
    c::ExpertLayout layout_;
    uint8_t *blobs_ = nullptr, *xq_ = nullptr, *scratch_ = nullptr;
    float *x_ = nullptr, *scales_ = nullptr, *output_ = nullptr;
    unsigned long long* ptr_ = nullptr;
    int32_t *start_ = nullptr, *dst_ = nullptr, *tok_ = nullptr, *count_ = nullptr;
};

// mode: 0=mixed, 1=no remote hits, 2=all remote (uses a separately larger tier).
std::vector<int32_t> routing(const core::MultiGpuExperts& remote, int layer, int tokens, int mode) {
    std::vector<int32_t> primary, secondary, cpu;
    for (int e = 0; e < EXPERTS; ++e) {
        const auto loc = remote.location(layer, e);
        if (loc.owner == 0) primary.push_back(e);
        else if (loc.owner > 0) secondary.push_back(e);
        else if (loc.owner == -1) cpu.push_back(e);
    }
    if (mode == 2) require(secondary.size() >= TOPK, "all-remote fixture needs ten distinct resident experts");
    else require(primary.size() == 1 && secondary.size() == 2 && cpu.size() >= 10, "fixture residency distribution");
    std::vector<int32_t> ids((size_t) tokens * TOPK);
    for (int t = 0; t < tokens; ++t) {
        std::vector<int32_t> row;
        if (mode == 2) {
            for (int j = 0; j < TOPK; ++j) row.push_back(secondary[(size_t) (j + t) % secondary.size()]);
        } else if (mode == 1) {
            row.push_back(primary[0]);
            row.insert(row.end(), cpu.begin(), cpu.begin() + 9);
        } else {
            row = {cpu[0], secondary[1], primary[0], cpu[1], secondary[0], cpu[2], cpu[3], cpu[4], cpu[5], cpu[6]};
            if (t & 1) row[1] = cpu[7]; // unequal group lengths: T vs ceil(T/2)
        }
        // Repeated IDs ACROSS tokens (valid top-10 routing), never twice within one token. Both groups
        // and destination rows are interleaved, and the first encountered group changes with the token.
        std::rotate(row.begin(), row.begin() + (3 * t % TOPK), row.end());
        if (t & 1) std::reverse(row.begin(), row.end());
        std::copy(row.begin(), row.end(), ids.begin() + t * TOPK);
    }
    return ids;
}

void dispatch_case(core::MultiGpuExperts& remote, PrimaryReference& reference, int primary,
                   int layer, int tokens, int mode) {
    const bool no_hits = mode == 1;
    const auto ids = routing(remote, layer, tokens, mode);
    const size_t n = ids.size();
    std::vector<float> x((size_t) tokens * c::H);
    for (int t = 0; t < tokens; ++t) for (int i = 0; i < c::H; ++i)
        x[(size_t) t * c::H + i] = (float) ((i * 31 + t * 47 + layer * 19) % 211 - 105) * (0.003f * (t + 1));
    std::vector<uint8_t> mask(n + 2, 0xa5);
    std::vector<float> output(n * c::H + 2 * GUARD, -98765.25f);
    float* out = output.data() + GUARD;
    for (size_t i = 0; i < n * c::H; ++i) out[i] = -12345.25f - (float) (i % 47) * 0.125f;
    const auto before = output;
    std::string err;
    api(remote.launch(layer, x.data(), ids.data(), tokens, TOPK, mask.data() + 1, err), "remote launch", err);
    current_is(primary, "launch");
    require(mask.front() == 0xa5 && mask.back() == 0xa5, "remote mask overwrote guards");
    size_t hits = 0;
    for (size_t i = 0; i < n; ++i) {
        const bool hit = remote.location(layer, ids[i]).owner > 0;
        require(mask[i + 1] == (uint8_t) hit, "remote mask differs from actual ownership at row " + std::to_string(i));
        hits += hit;
    }
    require(mode == 2 ? hits == n : no_hits ? hits == 0 : hits > 0,
            "scenario must actually exercise its requested hit/miss path");
    if (!no_hits && tokens == c::MAXT) {
        std::vector<float> usage((size_t) LAYERS * EXPERTS, 0);
        require(!remote.adapt(usage, 1, c::BLOB, err), "adapt must reject an in-flight dispatch");
        current_is(primary, "rejected in-flight adapt");
        require(!remote.publish(false, err), "publish must reject an in-flight dispatch");
        current_is(primary, "rejected in-flight publish");
    }
    api(remote.finish(out, err), "remote finish", err);
    current_is(primary, "finish");
    for (size_t i = 0; i < GUARD; ++i)
        require(output[i] == before[i] && output[GUARD + n * c::H + i] == before[GUARD + n * c::H + i],
                "remote output overwrote guards");
    const auto expected = no_hits ? std::vector<float>{} : reference.run(layer, tokens, x, ids);
    for (size_t row = 0; row < n; ++row) {
        double norm = 0;
        for (int col = 0; col < c::H; ++col) {
            const size_t i = row * c::H + col;
            if (mask[row + 1] == 0) {
                require(out[i] == before[GUARD + i], "finish changed a primary/CPU row " + std::to_string(row));
                continue;
            }
            const float a = out[i], b = expected[i];
            // Mixed GPU architectures may differ in the last bits of SiLU. This is still a tight
            // per-element transport check, not a permissive aggregate norm or a zero-vs-zero pass.
            if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a - b) > 3e-5f + 3e-4f * std::abs(b)) {
                char message[320];
                std::snprintf(message, sizeof(message), "remote parity L=%d T=%d row=%zu expert=%d col=%d: remote %.9g primary %.9g",
                              layer, tokens, row, (int) ids[row], col, (double) a, (double) b);
                throw std::runtime_error(message);
            }
            norm += (double) a * a;
        }
        if (mask[row + 1]) require(norm > 1e-12, "remote row must be genuinely nonzero");
    }
    // Idempotent finish must not scatter stale rows from a preceding layer.
    const auto finished = output;
    api(remote.finish(out, err), "second finish", err);
    current_is(primary, "second finish");
    require(output == finished, "second finish replays stale remote output");
    ++cases;
}

void suite(core::MultiGpuExperts& remote, PrimaryReference& reference, int primary) {
    {
        auto ids = routing(remote, 0, c::MAXT, false);
        ids[1] = ids[4]; // duplicate remote expert within every first-token group is invalid
        std::vector<float> x((size_t) c::MAXT * c::H, 0.1f), out((size_t) MAX_ROWS * c::H, 123.0f);
        std::vector<uint8_t> mask((size_t) MAX_ROWS);
        std::string err;
        require(!remote.launch(0, x.data(), ids.data(), c::MAXT, TOPK, mask.data(), err),
                "duplicate per-token expert must be rejected before grouped kernels run");
        api(remote.finish(out.data(), err), "finish after rejected routing", err);
        require(std::all_of(out.begin(), out.end(), [](float x) { return x == 123.0f; }),
                "invalid routing modified output");
        current_is(primary, "rejected routing");
        ++rejected_cases;
    }
    for (int layer = 0; layer < LAYERS; ++layer) for (int tokens = 1; tokens <= c::MAXT; ++tokens) {
        dispatch_case(remote, reference, primary, layer, tokens, false);
        dispatch_case(remote, reference, primary, layer, tokens, true); // after real work, catch stale-row writes
    }
}

void check_primary_map(const core::MultiGpuExperts& remote, const std::vector<int32_t>& host, int32_t* device) {
    std::vector<int32_t> downloaded(host.size());
    cu(cudaMemcpy(downloaded.data(), device, downloaded.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
       "primary residency download");
    require(downloaded == host, "publish did not update device primary residency");
    for (int l = 0; l < LAYERS; ++l) for (int e = 0; e < EXPERTS; ++e) {
        const auto loc = remote.location(l, e);
        require(host[(size_t) l * EXPERTS + e] == (loc.owner == 0 ? loc.slot : -1),
                "primary residency exposes a remote/CPU/in-flight expert or misses a primary slot");
    }
}

void swap_round(core::MultiGpuExperts& remote, core::ExpertCache& cache, SyntheticSource& source,
                int primary, std::vector<int32_t>& res, int32_t* res_device,
                int old_primary, int old_remote, int hot_primary, int hot_remote) {
    std::vector<float> usage((size_t) LAYERS * EXPERTS, 0);
    std::vector<p::ExpertLocation> primary_dst, remote_dst;
    for (int l = 0; l < LAYERS; ++l) {
        for (int e = 0; e < EXPERTS; ++e)
            if (remote.location(l, e).owner >= 0) usage[(size_t) l * EXPERTS + e] = 100;
        usage[(size_t) l * EXPERTS + old_primary] = 0;
        usage[(size_t) l * EXPERTS + old_remote] = 1;
        usage[(size_t) l * EXPERTS + hot_primary] = (float) (40 + l * 10);
        usage[(size_t) l * EXPERTS + hot_remote] = (float) (30 + l * 10);
        primary_dst.push_back(remote.location(l, old_primary));
        remote_dst.push_back(remote.location(l, old_remote));
        require(primary_dst.back().owner == 0 && remote_dst.back().owner == 1, "adaptive destinations start on both GPUs");
        require(remote.location(l, hot_primary).owner == -1 && remote.location(l, hot_remote).owner == -1,
                "adaptive incoming experts start CPU-only");
    }
    std::string err;
    api(remote.adapt(usage, 2 * LAYERS, source.payload_budget(2), err), "adapt", err);
    current_is(primary, "adapt");
    for (int l = 0; l < LAYERS; ++l) {
        require(remote.location(l, old_primary).owner == -1 && remote.location(l, old_remote).owner == -1,
                "adaptive victims become CPU-only before publication");
        require(remote.location(l, hot_primary).owner == -2 && remote.location(l, hot_remote).owner == -2,
                "uploads must remain in-flight until publish (even if DMA already completed)");
        require(res[(size_t) l * EXPERTS + old_primary] == -1 && res[(size_t) l * EXPERTS + hot_primary] == -1,
                "unpublished primary admission must not be visible in host residency");
    }
    const auto decayed = usage;
    api(remote.adapt(usage, 2 * LAYERS, source.payload_budget(2), err), "adapt while publication pending", err);
    current_is(primary, "pending adapt");
    require(usage == decayed, "pending adaptation must not run a second time");
    api(remote.publish(false, err), "nonblocking publish", err);
    current_is(primary, "nonblocking publish");
    // May or may not have finished: do not make GPU timing a test assumption. Evictions must be uploaded either way.
    check_primary_map(remote, res, res_device);
    api(remote.publish(true, err), "blocking publish", err);
    current_is(primary, "blocking publish");
    check_primary_map(remote, res, res_device);
    for (int l = 0; l < LAYERS; ++l) {
        const auto a = remote.location(l, hot_primary), b = remote.location(l, hot_remote);
        require(a.owner == primary_dst[(size_t) l].owner && a.slot == primary_dst[(size_t) l].slot &&
                    b.owner == remote_dst[(size_t) l].owner && b.slot == remote_dst[(size_t) l].slot,
                "published experts must inherit exact same-layer primary and secondary slots");
        require(remote.location(l, old_primary).owner == -1 && remote.location(l, old_remote).owner == -1,
                "published eviction must not resurrect old expert ownership");
        api(cache.verify_slot(a.slot, source.blob(l, hot_primary), err, (int64_t) source.layer_bytes(l)), "primary replacement bytes", err);
    }
    api(remote.publish(true, err), "idempotent publish", err);
    current_is(primary, "idempotent publish");
    check_primary_map(remote, res, res_device);
}

void run_pair(int primary, int secondary, bool native, bool iq4 = false) {
    std::printf("multi_gpu_test: primary %d, secondary %d, %s\n", primary, secondary,
                iq4 ? "native IQ4_XS/Q2_0 and IQ2_XS/IQ4_NL" :
                native ? "native IQ3_XXS/Q2_0 and IQ2_XS/IQ4_NL" : "canonical Q2_0");
    cu(cudaSetDevice(primary), "select test primary");
    TestLayout layout(native, iq4);
    SyntheticSource source;
    core::ExpertCache cache;
    Buffers primary_memory(primary);
    std::vector<int32_t> res((size_t) LAYERS * EXPERTS, -1);
    int32_t* res_device = primary_memory.alloc<int32_t>(res.size());
    std::string err;
    api(cache.open_sized({(int64_t) source.layer_bytes(0), (int64_t) source.layer_bytes(1)},
                         LAYERS, EXPERTS, err), "primary cache open", err);
    for (int l = 0; l < LAYERS; ++l) {
        const int32_t slot = cache.admit(l, 0);
        require(slot >= 0, "primary cache admission");
        res[(size_t) l * EXPERTS] = slot;
        api(cache.fill_slot_blocking(slot, source.blob(l, 0), err, (int64_t) source.layer_bytes(l)), "primary cache fill", err);
    }
    cu(cudaMemcpy(res_device, res.data(), res.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "initial residency upload");
    PrimaryReference reference(primary, source);
    {
        // Destroy the runtime before the cache, source, host map and device map, also on exceptions.
        core::MultiGpuExperts remote;
        const std::vector<p::ExpertId> partial_ranking{{1, 2}, {0, 1}};
        api(remote.init(primary, {secondary}, cache, res, res_device, source, partial_ranking,
                        LAYERS, EXPERTS, c::MAXT, 0, source.slot_budget(2), err), "remote init", err);
        current_is(primary, "init");
        for (int l = 0; l < LAYERS; ++l) for (int e = 0; e < EXPERTS; ++e) {
            const int expected = e == 0 ? 0 : e <= 2 ? 1 : -1;
            require(remote.location(l, e).owner == expected, "initial complementary residency");
        }
        check_primary_map(remote, res, res_device);
        suite(remote, reference, primary);
        swap_round(remote, cache, source, primary, res, res_device, 0, 1, 3, 4);
        suite(remote, reference, primary); // verifies actual new remote blob contents, not just location metadata
        swap_round(remote, cache, source, primary, res, res_device, 3, 4, 5, 6);
        suite(remote, reference, primary);
        remote.report();
    }
    current_is(primary, "MultiGpuExperts destruction");
    {
        core::MultiGpuExperts all_remote;
        api(all_remote.init(primary, {secondary}, cache, res, res_device, source, {},
                            LAYERS, EXPERTS, c::MAXT, 0, source.slot_budget(EXPERTS - 1), err),
            "all-remote tier init", err);
        current_is(primary, "all-remote init");
        for (int l = 0; l < LAYERS; ++l)
            for (int t = 1; t <= c::MAXT; ++t) dispatch_case(all_remote, reference, primary, l, t, 2);
    }
    current_is(primary, "all-remote destruction");
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<int> selected;
        if (argc == 3 && std::strcmp(argv[1], "--devices") == 0) {
            selected = p::parse_devices(argv[2]);
            require(selected.size() == 2, "--devices requires exactly two distinct ordinals (primary,secondary)");
        } else if (argc != 1) {
            throw std::invalid_argument("usage: multi_gpu_test [--devices PRIMARY,SECONDARY]");
        }
        int count = 0;
        const auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || (status == cudaSuccess && count < 2)) {
            std::printf("multi_gpu_test: SKIP (requires >=2 CUDA GPUs; found %d)\n", count);
            return 77;
        }
        cu(status, "enumerate CUDA GPUs"); // a broken driver is a failure, not a false skip
        auto run = [&](bool native, bool iq4 = false) {
            if (!selected.empty()) {
                require(selected[0] < count && selected[1] < count, "selected CUDA ordinal outside visible device range");
                run_pair(selected[0], selected[1], native, iq4);
            } else {
                run_pair(0, 1, native, iq4);
                run_pair(1, 0, native, iq4); // proves ownership ordinals are not hard-coded to device zero
            }
        };
        run(false);
#if defined(STRATA_NATIVE_EXPERTS)
        run(true);
        run(true, true); // upstream IQ3_S support uses IQ4_XS gate/up experts
#else
        std::printf("multi_gpu_test: native IQ3/IQ2 cases NOT TESTED (STRATA_NATIVE_EXPERTS=OFF)\n");
#endif
        require(cleanup_failures == 0, "CUDA resource cleanup failed");
        std::printf("multi_gpu_test: OK (%d dispatch cases, %d rejected invalid routes; T=1..8, real nonzero "
                    "remote parity, no-hits, two adaptive rounds)\n", cases, rejected_cases);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "multi_gpu_test: FAIL: %s\n", e.what());
        return 1;
    }
}
