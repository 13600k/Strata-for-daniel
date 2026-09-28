// src/core/device.cu - P2.S1: the CUDA side of the runtime core.
#include "strata/core/device.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

namespace strata::core {

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw CudaError(std::string(what) + ": " + cudaGetErrorString(e), (int) e);
    }
}

// A NaN pattern, not zero.  Zeros read from uninitialised memory are indistinguishable from real zeros in a
// dequantized weight or a masked attention score, which is exactly the kind of wrong-but-plausible value the
// Phase 1 harnesses kept catching.
__global__ void poison_kernel(float* p, uint64_t n_floats) {
    const uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_floats) p[i] = __int_as_float(0x7fc00000);
}

}  // namespace

DeviceScope::DeviceScope(int ordinal) {
    check(cudaGetDevice(&previous_), "cudaGetDevice");
    check(cudaSetDevice(ordinal), "cudaSetDevice");
}

DeviceScope::~DeviceScope() {
    if (previous_ >= 0) (void) cudaSetDevice(previous_);
}

void device_check_kernels(int ordinal) {
    DeviceArena probe(256, ordinal, true);
}

DeviceInfo device_info(int ordinal) {
    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count == 0) {
        throw CudaError("no CUDA device is present; Strata requires an NVIDIA GPU with compute capability >= 8.0", -1);
    }
    if (ordinal < 0 || ordinal >= count) {
        throw CudaError("device ordinal " + std::to_string(ordinal) + " is out of range (have " +
                            std::to_string(count) + ")",
                        -1);
    }
    DeviceInfo d;
    d.ordinal = ordinal;
    DeviceScope scope(ordinal);

    cudaDeviceProp p{};
    check(cudaGetDeviceProperties(&p, ordinal), "cudaGetDeviceProperties");
    d.name = p.name;
    d.uuid = "GPU-";
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) d.uuid += '-';
        char hex[3];
        std::snprintf(hex, sizeof(hex), "%02x", (unsigned int) (unsigned char) p.uuid.bytes[i]);
        d.uuid += hex;
    }
    d.cc_major = p.major;
    d.cc_minor = p.minor;
    d.multi_processor_count = p.multiProcessorCount;

    size_t free_b = 0, total_b = 0;
    check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    d.free_bytes = free_b;
    d.total_bytes = total_b;

    check(cudaDriverGetVersion(&d.driver_version), "cudaDriverGetVersion");
    check(cudaRuntimeGetVersion(&d.runtime_version), "cudaRuntimeGetVersion");

    if (d.cc_major < 8) {
        throw CudaError("device " + d.name + " reports compute capability " + std::to_string(d.cc_major) +
                            "." + std::to_string(d.cc_minor) + "; Strata requires compute capability >= 8.0", -1);
    }
    return d;
}

DeviceArena::DeviceArena(uint64_t bytes, int ordinal, bool poison)
    : capacity_(bytes), ordinal_(ordinal), poison_(poison) {
    if (bytes == 0) throw CudaError("DeviceArena of 0 bytes", -1);
    DeviceScope scope(ordinal);
    // One allocation for the whole region.  cudaMalloc of a large block is the thing that can fail late, so it
    // happens once, here, before anything depends on it.
    check(cudaMalloc(&base_, (size_t) bytes), "cudaMalloc");
    try {
        if (poison_) {
            const int threads = 256;
            const uint64_t n = bytes / sizeof(float);
            const uint64_t blocks = (n + threads - 1) / threads;
            // gridDim.x is 32-bit; retain the chunking for unusually large regions.
            const uint64_t max_blocks = 0x7FFFFFFFull;
            for (uint64_t b = 0; b < blocks; b += max_blocks) {
                const uint64_t chunk = (blocks - b < max_blocks) ? (blocks - b) : max_blocks;
                poison_kernel<<<(unsigned) chunk, threads>>>((float*) base_ + b * threads, n - b * threads);
                check(cudaGetLastError(), "poison_kernel");
            }
            check(cudaDeviceSynchronize(), "poison sync");
        }
    } catch (...) {
        // A constructor that throws will not run ~DeviceArena. In particular, the
        // startup probe must not leak its allocation when this build lacks an SM image.
        (void) cudaFree(base_);
        base_ = nullptr;
        throw;
    }
}

DeviceArena::~DeviceArena() {
    if (base_) {
        int previous = -1;
        (void) cudaGetDevice(&previous);
        if (cudaSetDevice(ordinal_) == cudaSuccess) (void) cudaFree(base_);
        if (previous >= 0) (void) cudaSetDevice(previous);
    }
}

void* DeviceArena::alloc(uint64_t bytes, uint64_t align) {
    if (bytes == 0) return nullptr;
    if (align == 0 || (align & (align - 1)) != 0) {
        throw CudaError("DeviceArena::alloc alignment must be a power of two", -1);
    }
    const uint64_t start = (used_ + align - 1) & ~(align - 1);
    if (start + bytes > capacity_) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "DeviceArena out of memory: asked for %llu B at offset %llu (align %llu) in a %llu B "
                      "region - the plan from P1.S9 did not close",
                      (unsigned long long) bytes, (unsigned long long) start, (unsigned long long) align,
                      (unsigned long long) capacity_);
        throw CudaError(msg, -1);
    }
    used_ = start + bytes;
    return (char*) base_ + start;
}

}  // namespace strata::core
