// ans.cu
// Block-Adaptive rANS for LiDAR Deltas
// Research Implementation: Single Context per Block, Shared Memory Decoding

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstring>

// Thrust is required for the prefix sum (scan) operation
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#define RANS_BYTE_L (1u << 15)
#define SCALE_BITS 12 
#define PROB_SCALE (1 << SCALE_BITS)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// --------------------------------------------------------------------------
// Helper Functions
// --------------------------------------------------------------------------

__host__ __device__ inline uint8_t get_k(int32_t v) {
    if (v == 0 || v == 1) return 0;
    if (v == INT32_MIN) return 32;
    uint32_t m = (v < 0) ? -v : v;
    uint32_t val = (v < 0) ? (m + 1) : m;
#ifdef __CUDA_ARCH__
    return 32 - __clz(val - 1);
#else
    uint32_t target = val - 1;
    int k = 0;
    while ((1u << k) <= target) k++;
    return k;
#endif
}

__host__ __device__ inline uint32_t map_delta(int32_t v, uint8_t k) {
    if (k == 0) return (uint32_t)v;
    if (k == 32) return 0;
    if (v < 0) return v + (1u << k) - 1;
    return v - 1;
}

struct DeviceTables {
    uint8_t* lookup_k;
    uint8_t* lookup_sym;
    uint32_t* enc_freq_k;
    uint32_t* enc_start_k;
    uint32_t* enc_freq_sym;
    uint32_t* enc_start_sym;
};

// --------------------------------------------------------------------------
// Kernels
// --------------------------------------------------------------------------

// 1. Split Integers
__global__ void k_split_integers(
    const int32_t* __restrict__ in_data,
    uint8_t* __restrict__ out_k,
    uint8_t* __restrict__ out_sym,
    uint32_t* __restrict__ out_raw,
    size_t n)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    int32_t val = in_data[idx];
    uint8_t k = get_k(val);
    uint32_t mapped = map_delta(val, k);

    out_k[idx] = k;

    if (k == 0) {
        out_sym[idx] = (uint8_t)mapped;
        out_raw[idx] = 0;
    }
    else if (k < 8) {
        out_sym[idx] = (uint8_t)mapped;
        out_raw[idx] = 0;
    }
    else if (k < 32) {
        out_sym[idx] = (uint8_t)(mapped >> (k - 8));
        out_raw[idx] = mapped & ((1u << (k - 8)) - 1);
    }
    else {
        out_sym[idx] = 0;
        out_raw[idx] = 0;
    }
}

// 2. Encoder
__global__ void k_encode_simple(
    uint8_t* out_buf,
    const uint8_t* in_k,
    const uint8_t* in_sym,
    const uint32_t* in_raw,
    uint32_t* out_sizes,
    size_t total_symbols,
    int symbols_per_thread,
    const uint32_t* freq_k, const uint32_t* start_k,
    const uint32_t* freq_sym, const uint32_t* start_sym,
    int num_chunks,
    int chunk_stride_bytes
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    size_t global_tid = bid * blockDim.x + tid;

    if (global_tid >= num_chunks) return;

    // RANS state
    uint32_t s = RANS_BYTE_L;

    // Encode writes BACKWARDS. 
    // We point to the END of the specific slot for this thread.
    uint8_t* end_ptr = out_buf + (global_tid * chunk_stride_bytes) + chunk_stride_bytes;
    uint8_t* ptr = end_ptr;

    uint64_t bit_buf = 0;
    int bit_cnt = 0;

    size_t start_idx = global_tid * symbols_per_thread;

    // Encode Backwards
    for (int i = symbols_per_thread - 1; i >= 0; --i) {
        size_t idx = start_idx + i;
        if (idx >= total_symbols) continue;

        uint8_t k = in_k[idx];
        uint8_t sym = in_sym[idx];
        uint32_t raw = in_raw[idx];

        if (k > 8 && k < 32) {
            int nbits = k - 8;
            bit_buf = (bit_buf << nbits) | (raw & ((1u << nbits) - 1));
            bit_cnt += nbits;
            while (bit_cnt >= 8) {
                *--ptr = (uint8_t)(bit_buf >> (bit_cnt - 8));
                bit_cnt -= 8;
                bit_buf &= ((1ull << bit_cnt) - 1);
            }
        }

        if (k != 32) {
            uint32_t f = freq_sym[sym];
            uint32_t st = start_sym[sym];
            uint32_t max = ((RANS_BYTE_L >> SCALE_BITS) << 8) * f;
            while (s >= max) {
                *--ptr = (uint8_t)(s & 0xff);
                s >>= 8;
            }
            s = ((s / f) << SCALE_BITS) + (s % f) + st;
        }

        {
            uint32_t f = freq_k[k];
            uint32_t st = start_k[k];
            uint32_t max = ((RANS_BYTE_L >> SCALE_BITS) << 8) * f;
            while (s >= max) {
                *--ptr = (uint8_t)(s & 0xff);
                s >>= 8;
            }
            s = ((s / f) << SCALE_BITS) + (s % f) + st;
        }
    }

    if (bit_cnt > 0) *--ptr = (uint8_t)bit_buf;
    *--ptr = (uint8_t)bit_cnt;

    *--ptr = (s >> 24); *--ptr = (s >> 16); *--ptr = (s >> 8); *--ptr = s;

    // Size is the distance from current ptr to the end of the slot
    out_sizes[global_tid] = (uint32_t)(end_ptr - ptr);
}


__global__ void encode_simple_byte(
    uint8_t* out_buf,
    const uint8_t* in_data, // Input is raw uint8_t symbols
    uint32_t* out_sizes,
    size_t total_symbols,
    int symbols_per_thread,
    const uint32_t* freq_sym,
    const uint32_t* start_sym,
    int num_chunks,
    int chunk_stride_bytes
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    size_t global_tid = bid * blockDim.x + tid;

    if (global_tid >= num_chunks) return;

    // RANS state
    uint32_t s = RANS_BYTE_L;

    // Encoder writes BACKWARDS from the END of the slot
    uint8_t* end_ptr = out_buf + (global_tid * chunk_stride_bytes) + chunk_stride_bytes;
    uint8_t* ptr = end_ptr;

    size_t start_idx = global_tid * symbols_per_thread;

    // Encode Backwards
    for (int i = symbols_per_thread - 1; i >= 0; --i) {
        size_t idx = start_idx + i;
        if (idx >= total_symbols) continue;

        uint8_t sym = in_data[idx];

        uint32_t f = freq_sym[sym];
        uint32_t st = start_sym[sym];

        // Renormalize (if s is too large, output bytes)
        // max = ((RANS_BYTE_L >> SCALE_BITS) << 8) * f;
        // Optimization: The comparison is equivalent to:
        // s >= (freq << (31 - SCALE_BITS)) ??? 
        // We stick to the standard readable version:
        uint32_t max = ((RANS_BYTE_L >> SCALE_BITS) << 8) * f;

        while (s >= max) {
            *--ptr = (uint8_t)(s & 0xff);
            s >>= 8;
        }

        // Update State
        s = ((s / f) << SCALE_BITS) + (s % f) + st;
    }

    // Write final state
    *--ptr = (s >> 24); *--ptr = (s >> 16); *--ptr = (s >> 8); *--ptr = s;

    // Size is the distance from current ptr to the end of the slot
    out_sizes[global_tid] = (uint32_t)(end_ptr - ptr);
}

// 3. Compactor Kernel (New!)
// Moves chunks from strided buffer to dense buffer based on offsets
__global__ void k_compact_chunks(
    const uint8_t* __restrict__ in_strided_buf,
    uint8_t* __restrict__ out_dense_buf,
    const uint32_t* __restrict__ chunk_sizes,
    const uint32_t* __restrict__ chunk_offsets,
    int num_chunks,
    int chunk_stride_bytes
) {
    int bid = blockIdx.x; // One block per chunk
    if (bid >= num_chunks) return;

    uint32_t size = chunk_sizes[bid];
    uint32_t offset = chunk_offsets[bid];

    // Source calculation:
    // The encoder wrote backwards from the END of the slot.
    // So the data starts at: SlotStart + Stride - ActualSize
    const uint8_t* src = in_strided_buf + (size_t)bid * chunk_stride_bytes + (chunk_stride_bytes - size);
    uint8_t* dst = out_dense_buf + offset;

    // Parallel Copy
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// 4. Decoder
__global__ void k_decode_shared(
    const uint8_t* __restrict__ in_buf,
    int32_t* __restrict__ out_data,
    const uint32_t* offsets,
    const uint8_t* __restrict__ global_lookup_k,
    const uint8_t* __restrict__ global_lookup_sym,
    const uint32_t* __restrict__ global_freq_k,
    const uint32_t* __restrict__ global_start_k,
    const uint32_t* __restrict__ global_freq_sym,
    const uint32_t* __restrict__ global_start_sym,
    size_t total_symbols,
    int symbols_per_thread,
    int num_chunks
) {
    __shared__ uint8_t s_lookup_k[PROB_SCALE];
    __shared__ uint8_t s_lookup_sym[PROB_SCALE];

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // Load lookups collaboratively 
    for (int i = tid; i < PROB_SCALE; i += bdim) {
        s_lookup_k[i] = global_lookup_k[i];
        s_lookup_sym[i] = global_lookup_sym[i];
    }
    __syncthreads();

    size_t global_tid = blockIdx.x * blockDim.x + tid;

    if (global_tid >= num_chunks) return;

    size_t block_start_idx = global_tid * symbols_per_thread;

    uint8_t* ptr = (uint8_t*)in_buf + offsets[global_tid];

    uint32_t s = ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);
    ptr += 4;

    uint64_t bit_buf = 0;
    int bit_cnt = 0;

    int valid_bits = *ptr++;
    if (valid_bits > 0) {
        bit_buf = *ptr++;
        bit_buf &= ((1ull << valid_bits) - 1);
        bit_cnt = valid_bits;
    }

    for (int i = 0; i < symbols_per_thread; ++i) {
        size_t out_idx = block_start_idx + i;
        if (out_idx >= total_symbols) break;

        uint32_t slot = s & (PROB_SCALE - 1);
        uint8_t k = s_lookup_k[slot];

        uint32_t freq = global_freq_k[k];
        uint32_t start = global_start_k[k];

        s = freq * (s >> SCALE_BITS) + (slot - start);
        if (s < RANS_BYTE_L) {
            s = (s << 8) | *ptr++;
            if (s < RANS_BYTE_L) s = (s << 8) | *ptr++;
        }

        uint8_t sym = 0;
        if (k != 32) {
            slot = s & (PROB_SCALE - 1);
            sym = s_lookup_sym[slot];

            freq = global_freq_sym[sym];
            start = global_start_sym[sym];

            s = freq * (s >> SCALE_BITS) + (slot - start);
            if (s < RANS_BYTE_L) {
                s = (s << 8) | *ptr++;
                if (s < RANS_BYTE_L) s = (s << 8) | *ptr++;
            }
        }

        uint32_t raw = 0;
        if (k > 8 && k < 32) {
            int nbits = k - 8;
            while (bit_cnt < nbits) {
                bit_buf |= (uint64_t)(*ptr++) << bit_cnt;
                bit_cnt += 8;
            }
            raw = (uint32_t)(bit_buf & ((1ull << nbits) - 1));
            bit_buf >>= nbits;
            bit_cnt -= nbits;
        }

        uint32_t mapped;
        if (k == 0) mapped = sym;
        else if (k < 8) mapped = sym;
        else if (k < 32) mapped = ((uint32_t)sym << (k - 8)) | raw;
        else mapped = 0;

        int32_t final_val;
        if (k == 0) final_val = (int32_t)mapped;
        else if (k == 32) final_val = INT32_MIN;
        else {
            if (mapped < (1u << (k - 1))) {
                final_val = (int32_t)mapped - (1 << k) + 1;
            }
            else {
                final_val = (int32_t)mapped + 1;
            }
        }

        out_data[out_idx] = final_val;
    }
}

__global__ void decode_shared_byte(
    const uint8_t* __restrict__ in_buf,
    int8_t* __restrict__ out_data, // Output is int8_t
    const uint32_t* offsets,
    const uint8_t* __restrict__ global_lookup_sym,
    const uint32_t* __restrict__ global_freq_sym,
    const uint32_t* __restrict__ global_start_sym,
    size_t total_symbols,
    int symbols_per_thread,
    int num_chunks
) {
    // Shared memory for fast lookup
    __shared__ uint8_t s_lookup_sym[PROB_SCALE];

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // Load lookups collaboratively 
    for (int i = tid; i < PROB_SCALE; i += bdim) {
        s_lookup_sym[i] = global_lookup_sym[i];
    }
    __syncthreads();

    size_t global_tid = blockIdx.x * blockDim.x + tid;
    if (global_tid >= num_chunks) return;

    size_t block_start_idx = global_tid * symbols_per_thread;

    // Setup pointer based on compact buffer offsets
    const uint8_t* ptr = in_buf + offsets[global_tid];

    // Read initial state
    uint32_t s = ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);
    ptr += 4;

    // Decode Loop
    for (int i = 0; i < symbols_per_thread; ++i) {
        size_t out_idx = block_start_idx + i;
        if (out_idx >= total_symbols) break;

        // 1. Get Symbol
        uint32_t slot = s & (PROB_SCALE - 1);
        uint8_t sym = s_lookup_sym[slot];

        // 2. Write Output
        out_data[out_idx] = (int8_t)sym;

        // 3. Advance rANS state
        uint32_t freq = global_freq_sym[sym];
        uint32_t start = global_start_sym[sym];

        // s = freq * (s / PROB_SCALE) + (slot - start)
        s = freq * (s >> SCALE_BITS) + (slot - start);

        // 4. Renormalize (Refill state from stream)
        while (s < RANS_BYTE_L) {
            s = (s << 8) | *ptr++;
        }
    }
}

// --------------------------------------------------------------------------
// Host Logic
// --------------------------------------------------------------------------

void build_lookup(uint8_t* lookup, const uint32_t* starts, const uint32_t* freqs, int alphabet_size) {
    for (int s = 0; s < alphabet_size; ++s) {
        for (uint32_t i = 0; i < freqs[s]; ++i) {
            lookup[starts[s] + i] = (uint8_t)s;
        }
    }
}

void normalize_freqs(std::vector<uint32_t>& freqs, std::vector<uint32_t>& starts) {
    uint64_t total = 0;
    for (auto f : freqs) total += f;

    uint64_t target = 1 << SCALE_BITS;
    uint64_t current_sum = 0;

    for (auto& f : freqs) {
        if (f > 0) {
            f = (uint32_t)((uint64_t)f * target / total);
            if (f == 0) f = 1;
        }
    }

    current_sum = 0;
    for (auto f : freqs) current_sum += f;

    if (current_sum != target) {
        int max_idx = 0;
        for (size_t i = 0; i < freqs.size(); i++) if (freqs[i] > freqs[max_idx]) max_idx = i;
        if (current_sum < target) freqs[max_idx] += (target - current_sum);
        else freqs[max_idx] -= (current_sum - target);
    }

    starts[0] = 0;
    for (size_t i = 0; i < freqs.size(); i++) {
        starts[i + 1] = starts[i] + freqs[i];
    }
}

int compress_stream_gpu(const std::vector<int32_t>& data, const char* name) {
    if (data.empty()) return 0;
    //printf("[%s] size: %.2f MB \n",name, double (data.size() * 4)/1024/1024);
    size_t n = data.size();

    // 1. Buffers for Splitting
    int32_t* d_in;
    uint8_t* d_k, * d_sym;
    uint32_t* d_raw;
    gpuErrchk(cudaMalloc(&d_in, n * 4));
    gpuErrchk(cudaMalloc(&d_k, n));
    gpuErrchk(cudaMalloc(&d_sym, n));
    gpuErrchk(cudaMalloc(&d_raw, n * 4));

    gpuErrchk(cudaMemcpy(d_in, data.data(), n * 4, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    k_split_integers << <numBlocks, blockSize >> > (d_in, d_k, d_sym, d_raw, n);
    gpuErrchk(cudaDeviceSynchronize());

    // 2. Frequency Analysis (Host)
    std::vector<uint8_t> h_k(n);
    std::vector<uint8_t> h_sym(n);
    gpuErrchk(cudaMemcpy(h_k.data(), d_k, n, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_sym.data(), d_sym, n, cudaMemcpyDeviceToHost));

    std::vector<uint32_t> freq_k(33, 0);
    std::vector<uint32_t> freq_sym(256, 0);
    for (auto x : h_k) freq_k[x]++;
    for (auto x : h_sym) freq_sym[x]++;

    std::vector<uint32_t> start_k(34), start_sym(257);
    normalize_freqs(freq_k, start_k);
    normalize_freqs(freq_sym, start_sym);

    std::vector<uint8_t> lut_k(PROB_SCALE), lut_sym(PROB_SCALE);
    build_lookup(lut_k.data(), start_k.data(), freq_k.data(), 33);
    build_lookup(lut_sym.data(), start_sym.data(), freq_sym.data(), 256);

    // 3. Tables to GPU
    DeviceTables tables;
    gpuErrchk(cudaMalloc(&tables.lookup_k, PROB_SCALE));
    gpuErrchk(cudaMalloc(&tables.lookup_sym, PROB_SCALE));
    gpuErrchk(cudaMalloc(&tables.enc_freq_k, 33 * 4));
    gpuErrchk(cudaMalloc(&tables.enc_start_k, 33 * 4));
    gpuErrchk(cudaMalloc(&tables.enc_freq_sym, 256 * 4));
    gpuErrchk(cudaMalloc(&tables.enc_start_sym, 256 * 4));

    gpuErrchk(cudaMemcpy(tables.lookup_k, lut_k.data(), PROB_SCALE, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tables.lookup_sym, lut_sym.data(), PROB_SCALE, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tables.enc_freq_k, freq_k.data(), 33 * 4, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tables.enc_start_k, start_k.data(), 33 * 4, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tables.enc_freq_sym, freq_sym.data(), 256 * 4, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tables.enc_start_sym, start_sym.data(), 256 * 4, cudaMemcpyHostToDevice));

    // 4. Encode
    int sym_per_thread = 2048;
    size_t total_syms = ((n + sym_per_thread - 1) / sym_per_thread) * sym_per_thread;
    int enc_threads = total_syms / sym_per_thread;
    int enc_blocks = (enc_threads + 255) / 256;

    int chunk_stride = sym_per_thread * 6;

    uint8_t* d_comp_buf;
    uint32_t* d_comp_sizes;
    gpuErrchk(cudaMalloc(&d_comp_buf, enc_threads * chunk_stride));
    gpuErrchk(cudaMalloc(&d_comp_sizes, enc_threads * 4));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    k_encode_simple << <enc_blocks, 256 >> > (
        d_comp_buf, d_k, d_sym, d_raw, d_comp_sizes,
        n, sym_per_thread,
        tables.enc_freq_k, tables.enc_start_k, tables.enc_freq_sym, tables.enc_start_sym,
        enc_threads, chunk_stride
        );
    gpuErrchk(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    //printf("[%s] Encode Time: %.3f ms. Throughput: %.2f MB/s\n", name, ms, (double)(n * 4) / (ms / 1000.0) / 1024 / 1024);
    // 5. Compaction (Scan + Gather)

    // A. Prefix Sum (Scan) on Sizes
    uint32_t* d_dec_offsets;
    gpuErrchk(cudaMalloc(&d_dec_offsets, enc_threads * 4));

    // Exclusive scan: [10, 20, 30] -> [0, 10, 30]
    thrust::exclusive_scan(thrust::device, d_comp_sizes, d_comp_sizes + enc_threads, d_dec_offsets);

    // B. Calculate Total Size
    uint32_t last_size, last_offset;
    gpuErrchk(cudaMemcpy(&last_size, d_comp_sizes + enc_threads - 1, 4, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&last_offset, d_dec_offsets + enc_threads - 1, 4, cudaMemcpyDeviceToHost));
    size_t total_compressed = last_offset + last_size;

    uint8_t* d_coherent_buf;
    gpuErrchk(cudaMalloc(&d_coherent_buf, total_compressed));

    // C. GPU Compaction Kernel
    // Launch one block per chunk. 256 threads per block is enough to copy chunks (typ. < 12KB)
    k_compact_chunks << <enc_threads, 256 >> > (
        d_comp_buf, d_coherent_buf,
        d_comp_sizes, d_dec_offsets,
        enc_threads, chunk_stride
        );
    gpuErrchk(cudaDeviceSynchronize());

    //printf("[%s] Encoded Size: %.2f MB\n", name, (double)total_compressed / 1024 / 1024);

    // 6. Decode
    int32_t* d_decoded;
    gpuErrchk(cudaMalloc(&d_decoded, n * 4));

 
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    k_decode_shared << <enc_blocks, 256 >> > (
        d_coherent_buf, d_decoded, d_dec_offsets,
        tables.lookup_k, tables.lookup_sym,
        tables.enc_freq_k, tables.enc_start_k,
        tables.enc_freq_sym, tables.enc_start_sym,
        n, sym_per_thread,
        enc_threads
        );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    //printf("[%s] Decode Time: %.3f ms. Throughput: %.2f MB/s\n", name, ms, (double)(n * 4) / (ms / 1000.0) / 1024 / 1024);

    // Verify
    std::vector<int32_t> h_verify(n);
    gpuErrchk(cudaMemcpy(h_verify.data(), d_decoded, n * 4, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; i++) {
        if (h_verify[i] != data[i]) {
            printf("FAIL at %zu. Exp %d Got %d\n", i, data[i], h_verify[i]);
            break;
        }
    }

    // Free
    cudaFree(d_in); cudaFree(d_k); cudaFree(d_sym); cudaFree(d_raw);
    cudaFree(d_comp_buf); cudaFree(d_comp_sizes);
    cudaFree(d_coherent_buf); cudaFree(d_dec_offsets); cudaFree(d_decoded);
    cudaFree(tables.lookup_k); cudaFree(tables.lookup_sym);
    cudaFree(tables.enc_freq_k); cudaFree(tables.enc_start_k);
    cudaFree(tables.enc_freq_sym); cudaFree(tables.enc_start_sym);

    return total_compressed;
}




int compress_stream_gpu(const std::vector<int8_t>& data, const char* name) {
    if (data.empty()) return 0;

    // Correct size calculation for int8_t
    //printf("[%s] Input Size: %.2f MB\n", name, double(data.size()) / 1024.0 / 1024.0);
    size_t n = data.size();

    // 1. Device Setup (Input is uint8_t equivalent)
    uint8_t* d_in;
    gpuErrchk(cudaMalloc(&d_in, n)); // Allocate exactly n bytes

    // Cast int8_t data to uint8_t for internal processing
    gpuErrchk(cudaMemcpy(d_in, data.data(), n, cudaMemcpyHostToDevice));

    // 2. Frequency Analysis (Host)
    // We can do this on CPU for simplicity as per original code
    std::vector<uint32_t> freq_sym(256, 0);

    // Treat int8_t as uint8_t (0-255) for frequency counting
    const uint8_t* raw_ptr = (const uint8_t*)data.data();
    for (size_t i = 0; i < n; ++i) {
        freq_sym[raw_ptr[i]]++;
    }

    std::vector<uint32_t> start_sym(257);
    normalize_freqs(freq_sym, start_sym); // Reuse existing helper

    std::vector<uint8_t> lut_sym(PROB_SCALE);
    // Use freq_sym data size 256
    build_lookup(lut_sym.data(), start_sym.data(), freq_sym.data(), 256);

    // 3. Tables to GPU
    DeviceTables tables;
    gpuErrchk(cudaMalloc(&tables.lookup_sym, PROB_SCALE));
    gpuErrchk(cudaMalloc(&tables.enc_freq_sym, 256 * 4));
    gpuErrchk(cudaMalloc(&tables.enc_start_sym, 256 * 4));

    gpuErrchk(cudaMemcpy(tables.lookup_sym, lut_sym.data(), PROB_SCALE, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tables.enc_freq_sym, freq_sym.data(), 256 * 4, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tables.enc_start_sym, start_sym.data(), 256 * 4, cudaMemcpyHostToDevice));

    // 4. Encode
    int sym_per_thread = 2048;
    size_t total_syms = ((n + sym_per_thread - 1) / sym_per_thread) * sym_per_thread;
    int enc_threads = total_syms / sym_per_thread;
    int enc_blocks = (enc_threads + 255) / 256;

    // Stride estimation: 1 byte input -> max ~1.01 bytes output usually, 
    // but rANS can expand slightly if entropy is max. 
    // sym_per_thread * 2 is plenty safe for int8.
    int chunk_stride = sym_per_thread * 2 + 16;

    uint8_t* d_comp_buf;
    uint32_t* d_comp_sizes;
    gpuErrchk(cudaMalloc(&d_comp_buf, enc_threads * chunk_stride));
    gpuErrchk(cudaMalloc(&d_comp_sizes, enc_threads * 4));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    encode_simple_byte << <enc_blocks, 256 >> > (
        d_comp_buf, d_in, d_comp_sizes,
        n, sym_per_thread,
        tables.enc_freq_sym, tables.enc_start_sym,
        enc_threads, chunk_stride
        );
    gpuErrchk(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    //printf("[%s] Encode Time: %.3f ms. Throughput: %.2f MB/s\n", name, ms, (double)(n) / (ms / 1000.0) / 1024 / 1024);

    // 5. Compaction (Scan + Gather)
    uint32_t* d_dec_offsets;
    gpuErrchk(cudaMalloc(&d_dec_offsets, enc_threads * 4));

    // Exclusive scan
    thrust::exclusive_scan(thrust::device, d_comp_sizes, d_comp_sizes + enc_threads, d_dec_offsets);

    // Calculate Total Size
    uint32_t last_size, last_offset;
    gpuErrchk(cudaMemcpy(&last_size, d_comp_sizes + enc_threads - 1, 4, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&last_offset, d_dec_offsets + enc_threads - 1, 4, cudaMemcpyDeviceToHost));
    size_t total_compressed = last_offset + last_size;

    uint8_t* d_coherent_buf;
    gpuErrchk(cudaMalloc(&d_coherent_buf, total_compressed));

    // Compact
    k_compact_chunks << <enc_threads, 256 >> > (
        d_comp_buf, d_coherent_buf,
        d_comp_sizes, d_dec_offsets,
        enc_threads, chunk_stride
        );
    gpuErrchk(cudaDeviceSynchronize());

    //printf("[%s] Encoded Size: %.2f MB (Ratio: %.2f:1)\n", name, (double)total_compressed / 1024 / 1024, (double)n / total_compressed);

    // 6. Decode
    int8_t* d_decoded;
    gpuErrchk(cudaMalloc(&d_decoded, n)); // Allocate exactly n bytes

    cudaEventRecord(start);
    decode_shared_byte << <enc_blocks, 256 >> > (
        d_coherent_buf, d_decoded, d_dec_offsets,
        tables.lookup_sym, tables.enc_freq_sym, tables.enc_start_sym,
        n, sym_per_thread,
        enc_threads
        );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    //printf("[%s] Decode Time: %.3f ms. Throughput: %.2f MB/s\n", name, ms, (double)(n) / (ms / 1000.0) / 1024 / 1024);

    // Verify
    std::vector<int8_t> h_verify(n);
    gpuErrchk(cudaMemcpy(h_verify.data(), d_decoded, n, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (size_t i = 0; i < n; i++) {
        if (h_verify[i] != data[i]) {
            printf("FAIL at %zu. Exp %d Got %d\n", i, data[i], h_verify[i]);
            errors++;
            if (errors > 10) break;
        }
    }
    //if (errors == 0) printf("[%s] Verification Passed.\n", name);

    // Free
    cudaFree(d_in);
    cudaFree(d_comp_buf); cudaFree(d_comp_sizes);
    cudaFree(d_coherent_buf); cudaFree(d_dec_offsets);
    cudaFree(d_decoded);
    cudaFree(tables.lookup_sym);
    cudaFree(tables.enc_freq_sym); cudaFree(tables.enc_start_sym);

    return total_compressed;
}