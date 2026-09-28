// Additional GPUs hold complementary expert weights and COMPUTE them locally.
// Inputs/results use bounded pinned-host buffers. No P2P, NCCL or NVLink dependency.
// The primary's attention, state, MTP and prefill paths stay on the primary device.
#pragma once

#include "strata/plan/expert_placement.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {
class ExpertCache;
class ExpertSource;

class MultiGpuExperts {
public:
    MultiGpuExperts();
    ~MultiGpuExperts();
    MultiGpuExperts(const MultiGpuExperts&) = delete;
    MultiGpuExperts& operator=(const MultiGpuExperts&) = delete;

    // Call after primary residency is established, before any verification window.
    // primary_res and source must outlive this object; primary_res must not resize.
    // Source blobs must be immutable, stable host storage (ArenaExpertSource or
    // FileExpertSource), not a reused single-expert staging buffer. Uploads are async.
    // cap_bytes==0: each secondary uses free VRAM minus its workspace and reserve.
    bool init(int primary, const std::vector<int>& secondary, ExpertCache& primary_cache,
              std::vector<int32_t>& primary_res, int32_t* primary_res_device,
              ExpertSource& source, const std::vector<plan::ExpertId>& ranking,
              int64_t layers, int64_t experts, int max_tokens, uint64_t reserve_bytes,
              uint64_t cap_bytes, std::string& err);

    // A single in-flight layer; entries retain their original token*k+router_index.
    // Launch all remote devices before the CPU pool, then finish AFTER CPU work.
    // finish writes ONLY remote-owned rows; primary/CPU rows are untouched.
    bool launch(int64_t layer, const float* x, const int32_t* ids, int tokens, int k,
                uint8_t* remote_entries, std::string& err);
    bool finish(float* out, std::string& err);

    // Between windows ONLY; adaptation may overlap MTP/commit on another host thread.
    // publish must run on the coordinator before the next window or before prefill.
    bool adapt(std::vector<float>& usage, int max_swaps, uint64_t max_bytes, std::string& err);
    bool publish(bool wait, std::string& err);
    void report() const; // stderr only, safe for the --serve protocol

    // Inspection for tests/diagnostics. -2 means an upload has not completed yet.
    plan::ExpertLocation location(int64_t layer, int64_t expert) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::core
