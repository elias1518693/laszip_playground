#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstring>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// --- ZigZag Encoding Helper Functions ---
// Maps signed int32 to unsigned uint32.
// 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, etc.
__host__ __device__ inline uint32_t map_zigzag(int32_t v) {
    return (uint32_t)((v << 1) ^ (v >> 31));
}

// Inverse of ZigZag encoding
__host__ __device__ inline int32_t unmap_zigzag(uint32_t u) {
    return (int32_t)((u >> 1) ^ (-(int32_t)(u & 1)));
}

// --- Bit Width Calculation ---
// Returns the number of bits needed to store 'u'.
// 0 needs 0 bits (special case handled in kernel), 1 needs 1 bit.
__host__ __device__ inline uint8_t get_needed_bits(uint32_t u) {
    if (u == 0) return 0;
    // __clz counts leading zeros. For a 32-bit integer, 32 - clz is the width.
#ifdef __CUDA_ARCH__
    return 32 - __clz(u);
#else
    // Host fallback
    uint32_t bits = 0;
    while ((1u << bits) <= u && bits < 32) bits++;
    if ((1ull << bits) <= u) bits++; // Check for 32-bit edge case
    return bits;
#endif
}

__global__ void k_split_integers2(
    const int32_t* __restrict__ in_data,
    uint32_t* __restrict__ out_raw,
    uint32_t* __restrict__ max_bits_per_block,
    size_t n)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    int32_t val = in_data[idx];

    // 1. Map using ZigZag (Scale Invariant)
    uint32_t mapped = map_zigzag(val);

    // 2. Compute required bits
    uint8_t k = get_needed_bits(mapped);

    // 3. Track maximum k for this block
    atomicMax(&max_bits_per_block[blockIdx.x], (unsigned int)k);

    // 4. Store the mapped raw value
    out_raw[idx] = mapped;
}

__global__ void pack_integers(
    const uint32_t* __restrict__ in_data,      // Input is d_raw (ZigZagged uint32)
    uint32_t* __restrict__ compressed_buffer,
    const uint32_t* __restrict__ max_bits_per_block,
    const uint64_t* __restrict__ offsets,      // Offsets in BITS
    size_t n)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    // 1. Get the bit width for this block
    uint32_t bits = max_bits_per_block[blockIdx.x];

    // Optimization: If bits is 0, the values are 0, nothing to write.
    if (bits == 0) return;

    // 2. Calculate the global bit position
    uint64_t global_bit_pos = offsets[blockIdx.x] + (uint64_t)threadIdx.x * bits;

    // 3. Location in 32-bit buffer
    uint64_t word_idx = global_bit_pos / 32;
    uint32_t bit_offset = (uint32_t)(global_bit_pos % 32);

    // 4. Load value
    uint32_t val = in_data[idx];

    // 5. Write first part (Atomic OR to merge with neighbors)
    atomicOr(&compressed_buffer[word_idx], val << bit_offset);

    // 6. Handle split across word boundary
    if (bit_offset + bits > 32) {
        uint32_t bits_in_first = 32 - bit_offset;
        uint32_t val_overflow = val >> bits_in_first;
        atomicOr(&compressed_buffer[word_idx + 1], val_overflow);
    }
}

__global__ void decode_modern_optimized(
    const uint32_t* __restrict__ compressed_buffer,
    const uint32_t* __restrict__ max_bits_per_block,
    const uint64_t* __restrict__ offsets,
    int32_t* d_out,
    size_t n)
{
    size_t global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_idx >= n) return;

    // 1. Get Bit Width
    uint32_t bits = max_bits_per_block[blockIdx.x];

    // Optimization: Pure zero block
    if (bits == 0) {
        // ZigZag(0) is 0, so unmap(0) is 0.
        d_out[global_idx] = 0;
        return;
    }

    // 2. Calculate Global Bit Offsets
    uint64_t block_start_bit = offsets[blockIdx.x];
    uint64_t global_bit_pos = block_start_bit + (uint64_t)threadIdx.x * bits;

    uint64_t global_word_idx = global_bit_pos / 32;
    uint32_t bit_offset = (uint32_t)(global_bit_pos % 32);

    // 3. Read from Global Memory
    uint32_t val = compressed_buffer[global_word_idx];
    val >>= bit_offset;

    // 4. Handle "Split" words
    if (bit_offset + bits > 32) {
        uint32_t word2 = compressed_buffer[global_word_idx + 1];
        uint32_t bits_in_first = 32 - bit_offset;
        val |= (word2 << bits_in_first);
    }

    // 5. Mask out higher bits (garbage from next value)
    if (bits < 32) {
        val &= ((1u << bits) - 1);
    }

    // 6. Unmap ZigZag
    d_out[global_idx] = unmap_zigzag(val);
}

int compress_stream_aatrox(const std::vector<int32_t>& data, const char* name)
{
    if (data.empty()) return 0;
    printf("[%s] size: %.2f MB \n", name, double(data.size() * 4) / 1024 / 1024);
    size_t n = data.size();

    // 1. Buffers
    int32_t* d_in = nullptr;
    uint32_t* d_raw = nullptr;
    uint32_t* d_max_bits = nullptr;

    gpuErrchk(cudaMalloc(&d_in, n * sizeof(int32_t)));
    gpuErrchk(cudaMalloc(&d_raw, n * sizeof(uint32_t)));

    gpuErrchk(cudaMemcpy(d_in, data.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));

    int blockSize = 256;
    uint32_t numBlocks = static_cast<uint32_t>((n + blockSize - 1) / blockSize);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Allocate and zero d_max_bits
    gpuErrchk(cudaMalloc(&d_max_bits, numBlocks * sizeof(uint32_t)));
    gpuErrchk(cudaMemset(d_max_bits, 0, numBlocks * sizeof(uint32_t)));

    // Launch Split (ZigZag Map + Max k calculation)
    k_split_integers2 << <numBlocks, blockSize >> > (d_in, d_raw, d_max_bits, n);
    gpuErrchk(cudaGetLastError());

    // Copy metadata to host to compute offsets
    std::vector<uint32_t> h_max_bits(numBlocks);
    gpuErrchk(cudaMemcpy(h_max_bits.data(), d_max_bits, numBlocks * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Compute compressed offsets
    uint64_t total_bits = 0;
    std::vector<uint64_t> offsets;
    offsets.reserve(numBlocks);

    for (uint32_t block = 0; block < numBlocks; ++block) {
        uint32_t k = h_max_bits[block];
        offsets.push_back(total_bits);

        size_t block_start = size_t(block) * blockSize;
        size_t items_in_block = std::min<size_t>(blockSize, (n > block_start) ? (n - block_start) : 0);

        total_bits += (uint64_t)k * items_in_block;
    }

    // Allocate Compressed Buffer
    uint32_t* compressed_buffer;
    // +1 safety margin for the split-word logic
    size_t buffer_bytes = ((total_bits + 31) / 32) * sizeof(uint32_t) + sizeof(uint32_t);
    uint64_t* d_offsets;

    gpuErrchk(cudaMalloc(&compressed_buffer, buffer_bytes));
    gpuErrchk(cudaMemset(compressed_buffer, 0, buffer_bytes)); // Essential for atomicOr
    gpuErrchk(cudaMalloc(&d_offsets, offsets.size() * sizeof(uint64_t)));
    gpuErrchk(cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Launch Pack
    pack_integers << <numBlocks, blockSize >> > (d_raw, compressed_buffer, d_max_bits, d_offsets, n);
    gpuErrchk(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("[%s] Encode Time: %.3f ms. Throughput: %.2f MB/s\n", name, ms, (double)(n * sizeof(int32_t)) / (ms / 1000.0) / 1024 / 1024);
    printf("[%s] Encoded Size: %.2f MB (Ratio: %.2f:1)\n", name, (double)total_bits / 8 / 1024 / 1024, (double)(n * 4) / ((double)total_bits / 8));

    // --- Decode ---
    cudaEventRecord(start);
    // Shared memory not actually used in the "modern_optimized" kernel provided, 
    // but kept arg for compatibility if you add it back.
    decode_modern_optimized << <numBlocks, blockSize >> > (compressed_buffer, d_max_bits, d_offsets, d_in, n);
    gpuErrchk(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("[%s] Decode Time: %.3f ms. Throughput: %.2f MB/s\n", name, ms, (double)(n * sizeof(int32_t)) / (ms / 1000.0) / 1024 / 1024);

    // Verify
    std::vector<int32_t> h_verify(n);
    gpuErrchk(cudaMemcpy(h_verify.data(), d_in, n * 4, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (size_t i = 0; i < n; i++) {
        if (h_verify[i] != data[i]) {
            printf("FAIL at %zu. Exp %d Got %d\n", i, data[i], h_verify[i]);
            errors++;
            if (errors > 5) break;
        }
    }
    if (errors == 0) printf("Validation Success!\n");

    // cleanup
    cudaFree(d_in);
    cudaFree(d_raw);
    cudaFree(d_max_bits);
    cudaFree(compressed_buffer);
    cudaFree(d_offsets);
    return 0;
}