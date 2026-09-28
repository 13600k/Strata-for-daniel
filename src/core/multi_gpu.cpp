#include "strata/core/multi_gpu.hpp"
#include "strata/core/device.hpp"
#include "strata/core/progress.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace strata::core {
namespace {
constexpr int H = kernels::cpu::H, FF = kernels::cpu::FF, K = 10;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

struct Worker {
    int device = -1, cap = 0, max_tokens = 0;
    ExpertCache cache;
    cudaStream_t compute = nullptr, refill = nullptr;
    cudaEvent_t filled = nullptr;
    std::vector<void*> allocations, pinned;
    float *x = nullptr, *xs = nullptr, *output = nullptr, *hx = nullptr, *hy = nullptr;
    uint8_t* xq = nullptr;
    void* scratch = nullptr;
    int32_t *start = nullptr, *dst = nullptr, *tok = nullptr, *count = nullptr;
    int32_t *hs = nullptr, *hd = nullptr, *ht = nullptr, *hc = nullptr;
    unsigned long long *ptr = nullptr, *hp = nullptr;
    std::vector<int32_t> rows, group_expert;
    int entries = 0, groups = 0;
    bool running = false;
    uint64_t calls = 0, routed = 0, distinct = 0, transfer_bytes = 0;
    double submit_ms = 0, wait_ms = 0;

    ~Worker() {
        // Even a partial initialization is cleaned up on the owning device.
        int previous = -1;
        (void) cudaGetDevice(&previous);
        if (device >= 0 && cudaSetDevice(device) == cudaSuccess) {
            if (compute) (void) cudaStreamSynchronize(compute);
            if (refill) (void) cudaStreamSynchronize(refill);
            if (filled) (void) cudaEventDestroy(filled);
            if (compute) (void) cudaStreamDestroy(compute);
            if (refill) (void) cudaStreamDestroy(refill);
            for (void* p : allocations) (void) cudaFree(p);
            for (void* p : pinned) (void) cudaFreeHost(p);
            cache.close();
        }
        if (previous >= 0) (void) cudaSetDevice(previous);
    }

    template <typename T> void alloc(T*& p, size_t n) {
        check(cudaMalloc((void**) &p, n * sizeof(T)), "remote expert buffer allocation");
        allocations.push_back(p);
    }
    template <typename T> void host(T*& p, size_t n) {
        check(cudaHostAlloc((void**) &p, n * sizeof(T), cudaHostAllocPortable), "remote expert pinned allocation");
        pinned.push_back(p);
    }
    void init(int ordinal, int nt) {
        device = ordinal;
        DeviceScope scope(device);
        max_tokens = nt; cap = nt * K;
        check(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking), "remote compute stream");
        check(cudaStreamCreateWithFlags(&refill, cudaStreamNonBlocking), "remote refill stream");
        check(cudaEventCreateWithFlags(&filled, cudaEventDisableTiming), "remote refill event");
        alloc(x, (size_t) nt * H); alloc(xs, (size_t) nt * (H / 32));
        alloc(xq, (size_t) nt * (H / 32) * 36); // enough for q8_1 or q8_0
        alloc(output, (size_t) cap * H);
        const size_t sb = std::max<uint64_t>(kernels::moe_hit_grouped_scratch_bytes(cap, H, FF),
                                            kernels::native_expert_scratch_bytes(cap, FF));
        uint8_t* bytes = nullptr; alloc(bytes, sb); scratch = bytes;
        alloc(start, cap + 1); alloc(dst, cap); alloc(tok, cap); alloc(count, 1); alloc(ptr, cap);
        host(hx, (size_t) nt * H); host(hy, (size_t) cap * H);
        host(hs, cap + 1); host(hd, cap); host(ht, cap); host(hc, 1); host(hp, cap);
        rows.resize((size_t) cap); group_expert.resize((size_t) cap);
    }

    void enqueue(const float* input, int nt, int64_t layer) {
        if (!entries) return;
        const auto t0 = std::chrono::steady_clock::now();
        DeviceScope scope(device);
        std::memcpy(hx, input, (size_t) nt * H * sizeof(float));
        *hc = groups;
        hs[groups] = entries;
        // Mark before the first submission so a partial enqueue is drained on failure.
        running = true;
        auto upload = [&](void* d, const void* h, size_t n) {
            check(cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, compute), "remote expert upload");
        };
        upload(x, hx, (size_t) nt * H * sizeof(float));
        upload(start, hs, (size_t) (groups + 1) * sizeof(int32_t));
        upload(dst, hd, (size_t) entries * sizeof(int32_t));
        upload(tok, ht, (size_t) entries * sizeof(int32_t));
        upload(count, hc, sizeof(int32_t));
        upload(ptr, hp, (size_t) groups * sizeof(unsigned long long));
        const auto& layout = kernels::cpu::expert_layout();
        if (layout.native) {
            kernels::quantize_q8_1_rows(x, nt, H, xq, compute);
            const auto& f = layout.fmt.at((size_t) layer);
            const auto l = kernels::native_expert_layout(f.gu_type, f.d_type, H, FF);
            kernels::native_expert_grouped(l, ptr, start, count, dst, tok, groups, entries, xq, scratch, output, compute);
        } else {
            kernels::quantize_q8_0_scaled(x, xq, xs, (int64_t) nt * H, compute);
            kernels::moe_grouped_s2(ptr, start, count, dst, tok, groups, entries, xq, xs, scratch, output, compute);
        }
        check(cudaGetLastError(), "remote expert kernel");
        check(cudaMemcpyAsync(hy, output, (size_t) entries * H * sizeof(float), cudaMemcpyDeviceToHost, compute),
              "remote expert result transfer");
        const auto q = cudaStreamQuery(compute); // also flushes submissions on WDDM
        if (q != cudaSuccess && q != cudaErrorNotReady) check(q, "remote expert submission");
        ++calls; routed += entries; distinct += groups;
        transfer_bytes += (uint64_t) (nt + entries) * H * sizeof(float);
        submit_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
};
} // namespace

struct MultiGpuExperts::Impl {
    int primary = -1, max_tokens = 0;
    int64_t experts = 0;
    ExpertCache* primary_cache = nullptr;
    std::vector<int32_t>* primary_res = nullptr;
    int32_t* primary_res_device = nullptr;
    ExpertSource* source = nullptr;
    std::vector<uint64_t> layer_bytes;
    std::vector<plan::ExpertLocation> locations;
    std::vector<std::unique_ptr<Worker>> workers;
    cudaStream_t primary_refill = nullptr;
    cudaEvent_t primary_filled = nullptr;
    std::vector<plan::ExpertSwap> pending;
    bool in_flight = false, initialized = false, dirty = false;
    std::string failure; // CUDA/source failures latch; never publish a partial failed upload
    uint64_t swaps = 0;

    ~Impl() {
        int previous = -1;
        (void) cudaGetDevice(&previous);
        if (primary >= 0 && cudaSetDevice(primary) == cudaSuccess) {
            if (primary_refill) (void) cudaStreamSynchronize(primary_refill);
            if (primary_filled) (void) cudaEventDestroy(primary_filled);
            if (primary_refill) (void) cudaStreamDestroy(primary_refill);
        }
        workers.clear();
        if (previous >= 0) (void) cudaSetDevice(previous);
    }

    void upload_primary_map() {
        if (!dirty) return;
        DeviceScope scope(primary);
        check(cudaMemcpy(primary_res_device, primary_res->data(), primary_res->size() * sizeof(int32_t),
                         cudaMemcpyHostToDevice), "primary expert residency upload");
        dirty = false;
    }
};

MultiGpuExperts::MultiGpuExperts() : impl_(new Impl) {}
MultiGpuExperts::~MultiGpuExperts() = default;

bool MultiGpuExperts::init(int primary, const std::vector<int>& secondary, ExpertCache& primary_cache,
                           std::vector<int32_t>& primary_res, int32_t* primary_res_device,
                           ExpertSource& source, const std::vector<plan::ExpertId>& ranking,
                           int64_t layers, int64_t experts, int max_tokens, uint64_t reserve_bytes,
                           uint64_t cap_bytes, std::string& err) {
    try {
        auto& m = *impl_;
        if (m.initialized || m.primary >= 0) throw std::runtime_error("multi-GPU tier already initialized");
        if (secondary.empty() || layers <= 0 || experts <= 0 || max_tokens < 2 || max_tokens > kernels::cpu::MAXT ||
            primary_res.size() != (size_t) (layers * experts) || !primary_res_device || !primary_cache.valid())
            throw std::runtime_error("multi-GPU tier needs primary residency and a valid verification geometry");
        std::vector<int> seen{primary};
        for (int d : secondary) {
            if (d < 0 || std::find(seen.begin(), seen.end(), d) != seen.end())
                throw std::runtime_error("duplicate or invalid secondary GPU");
            seen.push_back(d);
        }
        const auto order = plan::complete_expert_ranking(ranking, layers, experts);
        m.primary = primary; m.primary_cache = &primary_cache;
        m.primary_res = &primary_res; m.primary_res_device = primary_res_device;
        m.source = &source; m.experts = experts; m.max_tokens = max_tokens;
        m.locations.resize(primary_res.size());
        for (size_t i = 0; i < primary_res.size(); ++i)
            if (primary_res[i] >= 0) m.locations[i] = {0, primary_res[i]};
        const auto& layout = kernels::cpu::expert_layout();
        for (int64_t l = 0; l < layers; ++l) {
            m.layer_bytes.push_back(layout.blob_bytes(l));
            if (layout.native) {
                const auto& f = layout.fmt.at((size_t) l);
                if (!kernels::native_expert_supported(f.gu_type, f.d_type))
                    throw std::runtime_error("unsupported grouped expert format at layer " + std::to_string(l));
            }
        }
        {
            DeviceScope scope(primary);
            check(cudaStreamCreateWithFlags(&m.primary_refill, cudaStreamNonBlocking), "primary adaptive stream");
            check(cudaEventCreateWithFlags(&m.primary_filled, cudaEventDisableTiming), "primary adaptive event");
        }
        for (int device : secondary) {
            auto w = std::make_unique<Worker>();
            w->init(device, max_tokens);
            DeviceScope scope(device);
            size_t free = 0, total = 0;
            check(cudaMemGetInfo(&free, &total), "secondary free VRAM");
            uint64_t budget = free > reserve_bytes ? free - reserve_bytes : 0;
            if (cap_bytes) budget = std::min(budget, cap_bytes);
            std::vector<plan::ExpertId> placed;
            for (int attempt = 0; attempt < 8; ++attempt) {
                placed = plan::fit_experts(order, m.locations, m.layer_bytes, experts, budget);
                if (placed.empty()) break;
                std::vector<int64_t> sizes;
                for (auto id : placed) sizes.push_back((int64_t) m.layer_bytes[(size_t) id.first]);
                if (!w->cache.open_sized(sizes, layers, experts, err)) return false;
                // WDDM can report optimistic free memory until an allocation is touched.
                check(cudaStreamSynchronize(nullptr), "secondary cache residency check");
                check(cudaMemGetInfo(&free, &total), "secondary remaining VRAM");
                if (free >= reserve_bytes) break;
                const uint64_t give = std::max<uint64_t>(reserve_bytes - free, (uint64_t) w->cache.bytes() / 8);
                budget = (uint64_t) w->cache.bytes() > give ? (uint64_t) w->cache.bytes() - give : 0;
                w->cache.close();
                if (attempt == 7) throw std::runtime_error("secondary cache cannot maintain its VRAM reserve");
            }
            if (placed.empty()) {
                std::fprintf(stderr, "strata multi-GPU: device %d has no room/unassigned experts; no worker created\n", device);
                continue;
            }
            const int owner = (int) m.workers.size() + 1;
            for (auto id : placed) {
                const int32_t slot = w->cache.admit(id.first, id.second);
                const uint8_t* b = source.blob(id.first, id.second);
                if (slot < 0 || !b || !w->cache.fill_slot(slot, b, w->refill, err,
                                                        (int64_t) m.layer_bytes[(size_t) id.first]))
                    throw std::runtime_error("secondary cache fill: " + err);
                m.locations[(size_t) (id.first * experts + id.second)] = {owner, slot};
            }
            check(cudaStreamSynchronize(w->refill), "secondary initial cache fill");
            // Verify an actual uploaded blob, not just the residency arithmetic.
            const auto first = placed.front();
            if (!w->cache.verify_slot(0, source.blob(first.first, first.second), err,
                                      (int64_t) m.layer_bytes[(size_t) first.first])) return false;
            std::fprintf(stderr, "strata multi-GPU: device %d: %zu complementary experts, %.2f GiB; "
                                 "%.0f MiB reserve, pinned-host activation/result transport\n",
                         device, placed.size(), w->cache.gib(), (double) reserve_bytes / 1048576.0);
            m.workers.push_back(std::move(w));
        }
        if (m.workers.empty())
            std::fprintf(stderr, "strata multi-GPU: no secondary cache allocated; all experts already fit or "
                                 "secondary budgets are too small. Using primary/CPU execution.\n");
        m.initialized = true;
        return true;
    } catch (const std::exception& e) { err = e.what(); return false; }
}

bool MultiGpuExperts::launch(int64_t layer, const float* input, const int32_t* ids, int tokens, int k,
                             uint8_t* remote, std::string& err) {
    auto& m = *impl_;
    if (m.in_flight || !m.failure.empty()) {
        err = m.failure.empty() ? "previous remote layer has not completed" : m.failure;
        return false;
    }
    try {
        if (!m.initialized || layer < 0 || layer >= (int64_t) m.layer_bytes.size() ||
            tokens < 1 || tokens > m.max_tokens || k != K || !input || !ids || !remote)
            throw std::runtime_error("invalid remote expert dispatch or an unfinished previous layer");
        const int n = tokens * k;
        std::memset(remote, 0, (size_t) n);
        for (int i = 0; i < n; ++i) {
            if (ids[i] < 0 || ids[i] >= m.experts) throw std::runtime_error("remote expert id is out of range");
            // The grouped canonical kernel has at most MAXT entries per expert.
            // A top-k router emits each expert once per token, never duplicate rows.
            for (int j = (i / k) * k; j < i; ++j)
                if (ids[j] == ids[i]) throw std::runtime_error("duplicate expert within a token's top-k routes");
        }
        m.in_flight = true;
        for (size_t wi = 0; wi < m.workers.size(); ++wi) {
            auto& w = *m.workers[wi];
            w.entries = w.groups = 0;
            // Group repeated experts, preserving each token's router destination.
            for (int i = 0; i < n; ++i) {
                const auto loc = m.locations[(size_t) (layer * m.experts + ids[i])];
                if (loc.owner != (int) wi + 1) continue;
                bool seen = false;
                for (int q = 0; q < w.groups; ++q) seen = seen || w.group_expert[(size_t) q] == ids[i];
                if (seen) continue;
                w.group_expert[(size_t) w.groups] = ids[i];
                w.hp[w.groups] = (unsigned long long) w.cache.device_slot(loc.slot);
                w.hs[w.groups++] = w.entries;
                for (int j = i; j < n; ++j) if (ids[j] == ids[i]) {
                    const int dst = w.entries++;
                    w.rows[(size_t) dst] = j;
                    w.hd[dst] = dst;
                    w.ht[dst] = j / k;
                    remote[j] = 1;
                }
            }
            w.enqueue(input, tokens, layer);
        }
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        if (m.in_flight) m.failure = err;
        std::string ignored;
        finish(nullptr, ignored); // drain any partial submissions before reusing staging
        return false;
    }
}

bool MultiGpuExperts::finish(float* out, std::string& err) {
    bool ok = true;
    auto& m = *impl_;
    for (auto& wp : m.workers) {
        auto& w = *wp;
        if (!w.running) continue;
        try {
            DeviceScope scope(w.device);
            const auto t0 = std::chrono::steady_clock::now();
            progress_at("verify window: waiting for secondary expert GPU", w.device);
            check(cudaStreamSynchronize(w.compute), "remote expert completion");
            w.wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (out)
                for (int i = 0; i < w.entries; ++i)
                    std::memcpy(out + (size_t) w.rows[(size_t) i] * H, w.hy + (size_t) i * H, H * sizeof(float));
        } catch (const std::exception& e) { if (ok) err = e.what(); m.failure = err; ok = false; }
        w.running = false;
    }
    m.in_flight = false;
    return ok;
}

bool MultiGpuExperts::adapt(std::vector<float>& usage, int max_swaps, uint64_t max_bytes, std::string& err) {
    auto& m = *impl_;
    if (!m.initialized || m.in_flight || !m.failure.empty()) {
        err = m.failure.empty() ? "adaptation requires an initialized, idle expert tier" : m.failure;
        return false;
    }
    try {
        if (!m.pending.empty()) return true;
        m.pending = plan::choose_expert_swaps(usage, m.locations, m.layer_bytes, m.experts, max_swaps, max_bytes);
        for (const auto& s : m.pending) {
            const auto dst = s.destination;
            const int device = dst.owner == 0 ? m.primary : m.workers[(size_t) dst.owner - 1]->device;
            DeviceScope scope(device);
            ExpertCache& cache = dst.owner == 0 ? *m.primary_cache : m.workers[(size_t) dst.owner - 1]->cache;
            const auto stream = dst.owner == 0 ? m.primary_refill : m.workers[(size_t) dst.owner - 1]->refill;
            const uint8_t* b = m.source->blob((int64_t) s.incoming / m.experts, (int64_t) s.incoming % m.experts);
            if (!b) throw std::runtime_error("adaptive expert source is missing a blob");
            // There is no verification/prefill reader here. Keep both entries out of
            // routing until the upload completes; an in-flight admission is not a hit.
            m.locations[s.outgoing] = {};
            m.locations[s.incoming] = {-2, -1};
            if (dst.owner == 0) {
                (*m.primary_res)[s.outgoing] = -1;
                m.dirty = true;
            }
            if (!cache.fill_slot(dst.slot, b, stream, err, (int64_t) s.bytes)) throw std::runtime_error(err);
        }
        if (!m.pending.empty()) {
            { DeviceScope scope(m.primary); check(cudaEventRecord(m.primary_filled, m.primary_refill), "primary refill event"); }
            for (auto& w : m.workers) {
                DeviceScope scope(w->device);
                check(cudaEventRecord(w->filled, w->refill), "secondary refill event");
            }
        }
        m.swaps += m.pending.size();
        for (float& u : usage) u *= 0.7f;
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        if (!m.pending.empty()) m.failure = err;
        return false;
    }
}

bool MultiGpuExperts::publish(bool wait, std::string& err) {
    auto& m = *impl_;
    if (!m.initialized || m.in_flight || !m.failure.empty()) {
        err = m.failure.empty() ? "publishing requires an initialized, idle expert tier" : m.failure;
        return false;
    }
    try {
        if (!m.pending.empty()) {
            bool ready = true;
            auto completed = [&](int device, cudaEvent_t event) {
                DeviceScope scope(device);
                const auto status = wait ? cudaEventSynchronize(event) : cudaEventQuery(event);
                if (status == cudaErrorNotReady) ready = false;
                else check(status, "expert refill completion");
            };
            completed(m.primary, m.primary_filled);
            for (auto& w : m.workers) completed(w->device, w->filled);
            if (ready) {
                for (const auto& s : m.pending) {
                    m.locations[s.incoming] = s.destination;
                    if (s.destination.owner == 0) {
                        (*m.primary_res)[s.incoming] = s.destination.slot;
                        m.dirty = true;
                    }
                }
                m.pending.clear();
            }
        }
        // Evictions must reach the primary graph even if admissions are not ready.
        m.upload_primary_map();
        return true;
    } catch (const std::exception& e) { err = e.what(); m.failure = err; return false; }
}

plan::ExpertLocation MultiGpuExperts::location(int64_t layer, int64_t expert) const {
    const auto& m = *impl_;
    if (layer < 0 || layer >= (int64_t) m.layer_bytes.size() || expert < 0 || expert >= m.experts) return {};
    return m.locations[(size_t) (layer * m.experts + expert)];
}

void MultiGpuExperts::report() const {
    const auto& m = *impl_;
    for (const auto& w : m.workers)
        std::fprintf(stderr, "strata multi-GPU: device %d: %llu dispatches, %llu expert groups, %llu routed rows, "
                             "%.1f MiB activation/results, %.1f ms submission / %.1f ms exposed wait (cumulative)\n", w->device,
                     (unsigned long long) w->calls, (unsigned long long) w->distinct, (unsigned long long) w->routed,
                     (double) w->transfer_bytes / 1048576.0, w->submit_ms, w->wait_ms);
    std::fprintf(stderr, "strata multi-GPU: %llu coordinated adaptive admissions (cumulative)\n",
                 (unsigned long long) m.swaps);
}
} // namespace strata::core
