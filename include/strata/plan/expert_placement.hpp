// Host-only policies shared by startup, the multi-GPU tier and synthetic tests.
// No CUDA or model files are needed to exercise placement and eviction decisions.
#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strata::plan {

inline std::vector<int> parse_devices(const std::string& text) {
    std::vector<int> out;
    size_t at = 0;
    while (at < text.size()) {
        const size_t end = text.find(',', at);
        const size_t stop = end == std::string::npos ? text.size() : end;
        int ordinal = -1;
        const auto r = std::from_chars(text.data() + at, text.data() + stop, ordinal);
        if (r.ec != std::errc() || r.ptr != text.data() + stop || ordinal < 0 ||
            std::find(out.begin(), out.end(), ordinal) != out.end())
            throw std::invalid_argument("--devices requires distinct nonnegative CUDA ordinals, e.g. 0,1");
        out.push_back(ordinal);
        if (stop == text.size()) return out;
        at = stop + 1;
    }
    throw std::invalid_argument("--devices must not be empty or end in a comma");
}

using ExpertId = std::pair<int32_t, int32_t>; // layer, expert

// The bundled profile ranks only 8,000 experts. Its length is a warm-start hint,
// not a limit on how many experts larger cards may hold. The unranked tail is
// interleaved across layers, so an incomplete profile cannot starve late layers.
inline std::vector<ExpertId> complete_expert_ranking(const std::vector<ExpertId>& ranked,
                                                    int64_t layers, int64_t experts) {
    if (layers <= 0 || experts <= 0 || layers > INT32_MAX || experts > INT32_MAX ||
        layers > INT32_MAX / experts)
        throw std::invalid_argument("invalid expert geometry");
    std::vector<uint8_t> seen((size_t) (layers * experts), 0);
    std::vector<ExpertId> out;
    out.reserve(seen.size());
    for (auto id : ranked) {
        if (id.first < 0 || id.first >= layers || id.second < 0 || id.second >= experts)
            throw std::invalid_argument("expert profile contains an out-of-range pair");
        auto& used = seen[(size_t) (id.first * experts + id.second)];
        if (used) throw std::invalid_argument("expert profile contains a duplicate pair");
        used = 1;
        out.push_back(id);
    }
    for (int32_t e = 0; e < experts; ++e)
        for (int32_t l = 0; l < layers; ++l)
            if (!seen[(size_t) (l * experts + e)]) out.emplace_back(l, e);
    return out;
}

inline uint64_t expert_slot_bytes(uint64_t bytes) {
    if (bytes == 0 || bytes > UINT64_MAX - 255)
        throw std::invalid_argument("invalid expert byte size");
    return (bytes + 255) & ~uint64_t(255);
}

// owner 0 is the primary tier, 1..N the secondary tiers (NOT CUDA ordinals).
// -1 is CPU-only; -2 reserves an admission until its upload event completes.
struct ExpertLocation {
    int owner = -1;
    int32_t slot = -1;
};

inline std::vector<ExpertId> fit_experts(const std::vector<ExpertId>& ranked,
                                        const std::vector<ExpertLocation>& location,
                                        const std::vector<uint64_t>& layer_bytes,
                                        int64_t experts, uint64_t budget) {
    std::vector<ExpertId> out;
    for (const auto& id : ranked) {
        const size_t i = (size_t) (id.first * experts + id.second);
        if (location.at(i).owner != -1) continue;
        const uint64_t n = expert_slot_bytes(layer_bytes.at((size_t) id.first));
        if (n > budget) continue; // a later layer may have smaller blobs
        out.push_back(id);
        budget -= n;
    }
    return out;
}

struct ExpertSwap {
    size_t incoming = 0, outgoing = 0;
    ExpertLocation destination;
    uint64_t bytes = 0;
    float gain = 0;
};

// Same-layer replacement preserves variable-sized slot capacities. One global
// candidate set prevents two GPUs admitting the same CPU-only expert. Ties are
// deterministic. Uploads are bounded by BOTH count and bytes.
inline std::vector<ExpertSwap> choose_expert_swaps(const std::vector<float>& usage,
                                                  const std::vector<ExpertLocation>& location,
                                                  const std::vector<uint64_t>& layer_bytes,
                                                  int64_t experts, int max_swaps,
                                                  uint64_t max_bytes) {
    if (experts <= 0 || usage.size() != location.size() ||
        location.size() != layer_bytes.size() * (size_t) experts)
        throw std::invalid_argument("adaptive expert table dimensions differ");
    if (max_swaps <= 0 || max_bytes == 0) return {};
    std::vector<ExpertSwap> candidates;
    std::vector<size_t> hot, cold;
    auto score = [&](size_t i) { return std::isfinite(usage[i]) ? usage[i] : 0.0f; };
    for (size_t l = 0; l < layer_bytes.size(); ++l) {
        hot.clear(); cold.clear();
        for (size_t i = l * (size_t) experts; i < (l + 1) * (size_t) experts; ++i) {
            if (location[i].owner == -1 && score(i) >= 2.0f) hot.push_back(i);
            else if (location[i].owner >= 0 && location[i].slot >= 0) cold.push_back(i);
        }
        std::sort(hot.begin(), hot.end(), [&](size_t a, size_t b) {
            return score(a) != score(b) ? score(a) > score(b) : a < b;
        });
        std::sort(cold.begin(), cold.end(), [&](size_t a, size_t b) {
            return score(a) != score(b) ? score(a) < score(b) : a < b;
        });
        for (size_t j = 0; j < std::min(hot.size(), cold.size()); ++j) {
            const float gain = score(hot[j]) - score(cold[j]);
            if (gain < 1.5f) break;
            candidates.push_back({hot[j], cold[j], location[cold[j]], layer_bytes[l], gain});
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const ExpertSwap& a, const ExpertSwap& b) {
        const double aa = a.gain / (double) a.bytes, bb = b.gain / (double) b.bytes;
        return aa != bb ? aa > bb : a.incoming < b.incoming;
    });
    std::vector<ExpertSwap> out;
    for (const auto& s : candidates) {
        if ((int) out.size() == max_swaps) break;
        if (s.bytes > max_bytes) continue;
        out.push_back(s);
        max_bytes -= s.bytes;
    }
    return out;
}

} // namespace strata::plan
