// Standalone host-only placement policy tests: no CUDA, model files, Catch2 or assert/NDEBUG dependency.
// c++ -std=c++20 -DNDEBUG -Iinclude src/plan/expert_placement_test.cpp -o expert_placement_test
#include "strata/plan/expert_placement.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cstdio>
#include <exception>
#include <set>

namespace p = strata::plan;

namespace {
int failures = 0;
int checks = 0;

void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what.c_str()); ++failures; }
}

template <typename F> void invalid(F&& f, const std::string& what) {
    try {
        f();
        check(false, what + " (did not throw)");
    } catch (const std::invalid_argument&) {
        check(true, what);
    } catch (const std::exception& e) {
        check(false, what + " (wrong exception: " + e.what() + ")");
    }
}

void expert_formats() {
    for (int gu : {16, 17, 18, 21, 22, 23, 29, 42}) {
        check(strata::kernels::native_expert_supported(gu, 20), "native gate/up with IQ4_NL down");
        check(strata::kernels::native_expert_supported(gu, 42), "native gate/up with Q2_0 down");
    }
    check(strata::kernels::native_expert_supported(23, 42), "upstream IQ3_S uses IQ4_XS gate/up");
    for (auto types : {std::pair{12, 42}, std::pair{2, 42}, std::pair{23, 12}, std::pair{-1, 42}})
        check(!strata::kernels::native_expert_supported(types.first, types.second),
              "dense format support must not imply grouped expert support");
}

void devices() {
    check(p::parse_devices("0") == std::vector<int>{0}, "single device");
    check(p::parse_devices("2,0,7") == std::vector<int>({2, 0, 7}), "preserve primary/device order");
    check(p::parse_devices("00,01,2") == std::vector<int>({0, 1, 2}), "decimal leading zeroes");
    const auto max = std::numeric_limits<int>::max();
    check(p::parse_devices(std::to_string(max)) == std::vector<int>{max}, "maximum int ordinal parses");
    for (const std::string& text : std::vector<std::string>{
             "", ",", ",0", "0,", "0,1,", "0,,1", "0,,,1", "0,0", "1,0,1", "01,1",
             "-1", "0,-2", "-2147483648", "-2147483649", "+1", "0,+1", " 0", "0 ",
             "0, 1", "0,1 ", "0\t", "0\n", "x", "1x", "0,x", "1.0", "0x1", "1;2",
             std::to_string((uint64_t) max + 1), "0," + std::to_string((uint64_t) max + 1),
             "999999999999999999999999999999999999999999", std::string("0\0,1", 4)}) {
        invalid([&] { (void) p::parse_devices(text); }, "reject device list '" + text + "'");
    }
}

void ranking() {
    const std::vector<p::ExpertId> prefix{{2, 1}, {0, 0}};
    const std::vector<p::ExpertId> expected{{2, 1}, {0, 0}, {1, 0}, {2, 0}, {0, 1}, {1, 1},
                                          {0, 2}, {1, 2}, {2, 2}};
    check(p::complete_expert_ranking(prefix, 3, 3) == expected, "keep prefix; complete tail across layers");
    check(p::complete_expert_ranking(expected, 3, 3) == expected, "full ranking unchanged");
    check(p::complete_expert_ranking({}, 2, 3) ==
              std::vector<p::ExpertId>({{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0, 2}, {1, 2}}),
          "empty profile still ranks every expert, interleaved");
    check(p::complete_expert_ranking({}, 1, 1) == std::vector<p::ExpertId>{{0, 0}}, "singleton geometry");
    invalid([] { (void) p::complete_expert_ranking({{0, 1}, {0, 1}}, 2, 3); }, "duplicate ranking pair");
    for (auto id : std::vector<p::ExpertId>{{-1, 0}, {0, -1}, {2, 0}, {0, 3}, {INT32_MAX, 0}})
        invalid([&] { (void) p::complete_expert_ranking({id}, 2, 3); }, "out-of-range ranking pair");
    for (auto shape : std::vector<std::pair<int64_t, int64_t>>{
             {0, 1}, {1, 0}, {-1, 1}, {1, -1}, {(int64_t) INT32_MAX + 1, 1},
             {1, (int64_t) INT32_MAX + 1}, {INT32_MAX, 2}, {2, INT32_MAX}, {INT64_MAX, INT64_MAX}})
        invalid([&] { (void) p::complete_expert_ranking({}, shape.first, shape.second); },
                "invalid/overflowing ranking geometry");

    // The shipped profile's 8,000 entries must not cap a larger multi-GPU tier.
    std::vector<p::ExpertId> short_profile;
    for (int i = 0; i < 8000; ++i) short_profile.emplace_back(i / 512, i % 512);
    const auto full = p::complete_expert_ranking(short_profile, 48, 512);
    check(full.size() == 48 * 512, "8,000-entry profile grows to the full 24,576 experts");
    check(std::equal(short_profile.begin(), short_profile.end(), full.begin()), "large profile prefix retained");
    check(std::set<p::ExpertId>(full.begin(), full.end()).size() == full.size(), "full ranking is duplicate-free");
    check(full.at(8000) == p::ExpertId{16, 0} && full.at(8031) == p::ExpertId{47, 0} &&
              full.at(8032) == p::ExpertId{16, 1}, "unranked late layers are interleaved, not starved");
    check(full.back() == p::ExpertId{47, 511}, "completion reaches final layer and expert");

    const auto coder = p::complete_expert_ranking({}, 48, 256);
    check(coder.size() == 48 * 256, "pruned Coder uses its own 256-expert geometry");
    std::vector<p::ExpertLocation> loc(coder.size());
    std::vector<uint64_t> sizes(48, 1024);
    const auto first = p::fit_experts(coder, loc, sizes, 256, 50 * 1024);
    for (size_t i = 0; i < first.size(); ++i)
        loc[(size_t) first[i].first * 256 + first[i].second] = {0, (int32_t) i};
    const auto second = p::fit_experts(coder, loc, sizes, 256, 70 * 1024);
    check(first.size() == 50 && second.size() == 70, "pruned model uses heterogeneous cache budgets");
    std::set<p::ExpertId> residents(first.begin(), first.end());
    for (auto id : second)
        check(residents.insert(id).second && id.first < 48 && id.second < 256,
              "pruned secondary residency is complementary and bounded");
}

void fitting() {
    check(p::expert_slot_bytes(1) == 256 && p::expert_slot_bytes(255) == 256 &&
              p::expert_slot_bytes(256) == 256 && p::expert_slot_bytes(257) == 512,
          "slot sizes rounded up, not down, to 256 bytes");
    check(p::expert_slot_bytes(UINT64_MAX - 255) == UINT64_MAX - 255, "largest aligned slot size");
    for (uint64_t n : {uint64_t(0), UINT64_MAX - 254, UINT64_MAX})
        invalid([&] { (void) p::expert_slot_bytes(n); }, "invalid/overflowing slot size");

    const std::vector<uint64_t> bytes{513, 1, 257}; // capacities 768, 256, 512, not a uniform average
    const std::vector<p::ExpertId> order{{0, 0}, {1, 0}, {2, 0}, {0, 1}, {2, 1}, {1, 1},
                                       {0, 2}, {2, 2}, {1, 2}, {0, 3}, {2, 3}, {1, 3}};
    std::vector<p::ExpertLocation> loc(12);
    loc[0] = {0, 0}; // primary
    loc[4] = {1, 0}; // already on another GPU
    loc[8] = {-2, -1}; // pending admission, not a CPU candidate
    check(p::fit_experts(order, loc, bytes, 4, 0).empty(), "zero placement budget");
    check(p::fit_experts(order, loc, bytes, 4, 255).empty(), "budget below smallest aligned slot");
    check(p::fit_experts(order, loc, bytes, 4, 511) == std::vector<p::ExpertId>{{1, 1}},
          "alignment charged, oversized earlier experts skipped");
    const auto first = p::fit_experts(order, loc, bytes, 4, 1024);
    check(first == std::vector<p::ExpertId>({{0, 1}, {1, 1}}), "variable slots fit exact first budget");
    for (size_t i = 0; i < first.size(); ++i) loc[(size_t) (first[i].first * 4 + first[i].second)] = {2, (int32_t) i};
    const auto second = p::fit_experts(order, loc, bytes, 4, 512);
    check(second == std::vector<p::ExpertId>{{2, 1}}, "smaller second GPU complements first, without duplicates");
    for (auto id : second) loc[(size_t) (id.first * 4 + id.second)] = {3, 0};
    check(p::fit_experts(order, loc, bytes, 4, 256) == std::vector<p::ExpertId>{{1, 2}},
          "unequal third budget skips large layers but still fills");
    const auto rest = p::fit_experts(order, loc, bytes, 4, UINT64_MAX);
    check(rest == std::vector<p::ExpertId>({{0, 2}, {2, 2}, {1, 2}, {0, 3}, {2, 3}, {1, 3}}),
          "assigned and in-flight experts excluded even with unlimited space");
    for (auto& entry : loc) entry = {-2, -1};
    check(p::fit_experts(order, loc, bytes, 4, UINT64_MAX).empty(), "entire in-flight table cannot be placed twice");
}

std::vector<size_t> incoming(const std::vector<p::ExpertSwap>& swaps) {
    std::vector<size_t> out;
    for (const auto& s : swaps) out.push_back(s.incoming);
    return out;
}

void swap_invariants(const std::vector<p::ExpertSwap>& swaps, const std::vector<p::ExpertLocation>& loc,
                     const std::vector<uint64_t>& bytes, int experts, int max_swaps, uint64_t budget) {
    std::set<size_t> in, out;
    std::set<std::pair<int, int32_t>> destinations;
    check(swaps.size() <= (size_t) max_swaps, "adaptive count budget obeyed");
    for (const auto& s : swaps) {
        check(s.incoming < loc.size() && s.outgoing < loc.size(), "adaptive indices within geometry");
        if (s.incoming >= loc.size() || s.outgoing >= loc.size()) continue;
        check(in.insert(s.incoming).second && out.insert(s.outgoing).second, "adaptive incoming/outgoing unique globally");
        check(destinations.emplace(s.destination.owner, s.destination.slot).second, "adaptive destination unique globally");
        check(loc[s.incoming].owner == -1 && loc[s.outgoing].owner >= 0, "adaptive excludes assigned/in-flight incoming");
        check(s.incoming / experts == s.outgoing / experts, "replacement stays in same layer");
        check(s.destination.owner == loc[s.outgoing].owner && s.destination.slot == loc[s.outgoing].slot &&
                  s.destination.slot >= 0, "replacement preserves exact GPU and slot");
        check(s.bytes == bytes[s.incoming / experts], "upload budget charges actual layer blob bytes");
        check(std::isfinite(s.gain) && s.gain >= 1.5f, "adaptive gain finite and above hysteresis");
        check(s.bytes <= budget, "adaptive remaining byte budget obeyed");
        if (s.bytes <= budget) budget -= s.bytes;
    }
}

void adaptive_budgets() {
    const std::vector<uint64_t> bytes{257, 1024};
    const std::vector<float> usage{0, 1, 1, 10, 8, 6, 0, 2, 4, 20, 12, 2};
    const std::vector<p::ExpertLocation> loc{{0, 3}, {1, 5}, {2, 7}, {}, {}, {},
                                          {2, 4}, {0, 6}, {1, 8}, {}, {}, {}};
    const auto all = p::choose_expert_swaps(usage, loc, bytes, 6, 20, UINT64_MAX);
    check(incoming(all) == std::vector<size_t>({3, 4, 9, 5, 10}), "global gain-per-byte ordering across unequal layers");
    swap_invariants(all, loc, bytes, 6, 20, UINT64_MAX);
    const auto counted = p::choose_expert_swaps(usage, loc, bytes, 6, 2, UINT64_MAX);
    check(incoming(counted) == std::vector<size_t>({3, 4}), "count limit truncates highest-value swaps");
    const auto bounded = p::choose_expert_swaps(usage, loc, bytes, 6, 20, 771);
    check(incoming(bounded) == std::vector<size_t>({3, 4, 5}), "byte limit skips costly candidate and admits later smaller one");
    swap_invariants(bounded, loc, bytes, 6, 20, 771);
    check(incoming(p::choose_expert_swaps(usage, loc, bytes, 6, 20, 257)) == std::vector<size_t>{3},
          "upload byte limit counts payload, not slot alignment");
    check(p::choose_expert_swaps(usage, loc, bytes, 6, 20, 256).empty(), "cannot exceed byte budget by even one byte");
    for (int n : {-1, 0})
        check(p::choose_expert_swaps(usage, loc, bytes, 6, n, UINT64_MAX).empty(), "nonpositive swap count disables adaptation");
    check(p::choose_expert_swaps(usage, loc, bytes, 6, 20, 0).empty(), "zero upload bytes disables adaptation");

    const std::vector<p::ExpertLocation> ties{{2, 4}, {0, 3}, {}, {}, {1, 7}, {2, 8}, {}, {}};
    const auto tied = p::choose_expert_swaps({0, 0, 4, 4, 0, 0, 4, 4}, ties, {256, 256}, 4, 8, 1024);
    check(incoming(tied) == std::vector<size_t>({2, 3, 6, 7}), "equal densities use incoming-index tie break");
    check(tied.size() == 4 && tied[0].outgoing == 0 && tied[1].outgoing == 1 &&
              tied[2].outgoing == 4 && tied[3].outgoing == 5, "equal cold scores use outgoing-index tie break");
    swap_invariants(tied, ties, {256, 256}, 4, 8, 1024);
}

void adaptive_filters() {
    const std::vector<p::ExpertLocation> pair{{0, 0}, {}};
    auto choose_pair = [&](float cold, float hot) {
        return p::choose_expert_swaps({cold, hot}, pair, {256}, 2, 1, 256);
    };
    check(choose_pair(0.5f, 2.0f).size() == 1, "exact popularity and 1.5 gain thresholds admitted");
    check(choose_pair(std::nextafter(0.5f, 1.0f), 2.0f).empty() == false,
          "rounded FP32 gain remains the policy's comparison value");
    check(choose_pair(0.501f, 2.0f).empty(), "gain just below hysteresis rejected");
    check(choose_pair(0, std::nextafter(2.0f, 0.0f)).empty(), "popularity just below two rejected even with gain");
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    for (float bad : {nan, inf, -inf}) {
        check(choose_pair(0, bad).empty(), "nonfinite hot usage cannot trigger admission");
        const auto swaps = choose_pair(bad, 2);
        check(swaps.size() == 1 && swaps[0].gain == 2, "nonfinite cold usage treated as zero");
    }
    const std::vector<p::ExpertLocation> loc{{0, 0}, {}, {-2, -1}, {-2, -1}, {1, -1}, {}, {2, 9}};
    const auto swaps = p::choose_expert_swaps({0, 4, 100, 0, 0, 3, 100}, loc, {100}, 7, 20, 2000);
    check(incoming(swaps) == std::vector<size_t>{1}, "in-flight hot/cold entries and unallocated slots excluded");
    swap_invariants(swaps, loc, {100}, 7, 20, 2000);
    check(p::choose_expert_swaps({0, 10}, {{0, 0}, {}}, {256, 128}, 1, 2, 512).empty(),
          "hot expert in another layer cannot take a cold slot");
    invalid([] { (void) p::choose_expert_swaps({}, {}, {}, 0, 1, 1); }, "zero adaptive expert geometry");
    invalid([] { (void) p::choose_expert_swaps({}, {}, {}, -1, 1, 1); }, "negative adaptive expert geometry");
    invalid([] { (void) p::choose_expert_swaps({1}, {}, {1}, 1, 1, 1); }, "usage/location dimensions differ");
    invalid([] { (void) p::choose_expert_swaps({1}, {{}}, {1}, 2, 1, 1); }, "layer/expert dimensions differ");
    check(p::choose_expert_swaps({}, {}, {}, 1, 1, 1).empty(), "empty consistent adaptive tables");
}

void adaptive_sweep() {
    // Many deterministic tables and simultaneous count/byte limits, with multiple GPUs and in-flight entries.
    const std::vector<uint64_t> bytes{129, 513, 2049};
    for (int seed = 0; seed < 20; ++seed) {
        std::vector<p::ExpertLocation> loc(36);
        std::vector<float> usage(36);
        for (size_t i = 0; i < loc.size(); ++i) {
            const int state = (int) ((i + seed) % 6);
            loc[i] = state < 3 ? p::ExpertLocation{state, (int32_t) i} : p::ExpertLocation{state == 5 ? -2 : -1, -1};
            usage[i] = (float) ((i * 17 + seed * 7) % 31);
        }
        for (int count : {1, 2, 5, 20})
            for (uint64_t budget : {uint64_t(128), uint64_t(129), uint64_t(1026), uint64_t(100000)})
                swap_invariants(p::choose_expert_swaps(usage, loc, bytes, 12, count, budget),
                                loc, bytes, 12, count, budget);
    }
}
} // namespace

int main() {
    try {
        expert_formats();
        devices();
        ranking();
        fitting();
        adaptive_budgets();
        adaptive_filters();
        adaptive_sweep();
    } catch (const std::exception& e) {
        check(false, std::string("unexpected exception: ") + e.what());
    }
    std::printf("expert_placement_test: %s (%d checks, %d failures)\n", failures ? "FAILED" : "OK", checks, failures);
    return failures ? 1 : 0;
}
