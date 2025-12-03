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
    int chunk_stride_bytes // *** FIX: Passed stride
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    size_t global_tid = bid * blockDim.x + tid;

    if (global_tid >= num_chunks) return;

    // RANS state
    uint32_t s = RANS_BYTE_L;

    // *** FIX: Point to the END of the allocated slot, not the start ***
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

    out_sizes[global_tid] = (uint32_t)(end_ptr - ptr);
}

// 3. Decoder
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
    size_t n = data.size();

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

    // Encode
    int sym_per_thread = 2048;
    size_t total_syms = ((n + sym_per_thread - 1) / sym_per_thread) * sym_per_thread;
    int enc_threads = total_syms / sym_per_thread;
    int enc_blocks = (enc_threads + 255) / 256;

    // *** FIX: Increase buffer size to prevent overflow ***
    // 2048 symbols * 6 bytes = 12KB per thread (Safe for 32-bit worst case)
    int chunk_stride = sym_per_thread * 6;

    uint8_t* d_comp_buf;
    uint32_t* d_comp_sizes;
    gpuErrchk(cudaMalloc(&d_comp_buf, enc_threads * chunk_stride));
    gpuErrchk(cudaMalloc(&d_comp_sizes, enc_threads * 4));

    k_encode_simple << <enc_blocks, 256 >> > (
        d_comp_buf, d_k, d_sym, d_raw, d_comp_sizes,
        n, sym_per_thread,
        tables.enc_freq_k, tables.enc_start_k, tables.enc_freq_sym, tables.enc_start_sym,
        enc_threads, chunk_stride // *** FIX: Passed
        );
    gpuErrchk(cudaDeviceSynchronize());

    std::vector<uint32_t> h_sizes(enc_threads);
    gpuErrchk(cudaMemcpy(h_sizes.data(), d_comp_sizes, enc_threads * 4, cudaMemcpyDeviceToHost));

    // Pack buffers
    size_t total_compressed = 0;
    for (auto s : h_sizes) total_compressed += s;

    uint8_t* d_coherent_buf;
    uint32_t* d_dec_offsets;
    gpuErrchk(cudaMalloc(&d_coherent_buf, total_compressed));
    gpuErrchk(cudaMalloc(&d_dec_offsets, enc_threads * 4));

    std::vector<uint8_t> h_temp_all(total_compressed);
    std::vector<uint32_t> h_dec_offsets(enc_threads);
    uint32_t offset = 0;

    // Note: We copy from host for simplicity here (avoiding complex device scatter/gather)
    // Production code would do this on GPU
    std::vector<uint8_t> h_gpu_src(enc_threads * chunk_stride);
    gpuErrchk(cudaMemcpy(h_gpu_src.data(), d_comp_buf, enc_threads * chunk_stride, cudaMemcpyDeviceToHost));

    for (int i = 0; i < enc_threads; ++i) {
        h_dec_offsets[i] = offset;
        uint32_t sz = h_sizes[i];
        if (sz > 0) {
            // Src logic: Start + Stride - Size
            // Because pointer moves BACKWARDS from (Start+Stride)
            size_t src_idx = (size_t)i * chunk_stride + (chunk_stride - sz);
            memcpy(h_temp_all.data() + offset, h_gpu_src.data() + src_idx, sz);
            offset += sz;
        }
    }

    gpuErrchk(cudaMemcpy(d_coherent_buf, h_temp_all.data(), total_compressed, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_dec_offsets, h_dec_offsets.data(), enc_threads * 4, cudaMemcpyHostToDevice));

    printf("[%s] Encoded Size: %.2f MB\n", name, (double)total_compressed / 1024 / 1024);

    // Decode
    int32_t* d_decoded;
    gpuErrchk(cudaMalloc(&d_decoded, n * 4));

    cudaEvent_t start, stop;
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
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("[%s] Decode Time: %.3f ms. Throughput: %.2f MB/s\n", name, ms, (double)(n * 4) / (ms / 1000.0) / 1024 / 1024);

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

    return 0;
}