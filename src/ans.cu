// rans_byte_cuda.cu
// CUDA translation of "Simple byte-aligned rANS encoder/decoder" (Fabian 'ryg' Giesen)
// Single-file example with per-thread rANS encode/decode demo.
//
// Cleaned-up version:
// - Removed unused device functions (RansEncPut, RansDecAdvanceSymbol, etc.)
// - Inlined single-use helper functions (RansEncFlush, RansDecInit, etc.)
//   directly into the kernels for a more compact file.
//
// Compile: nvcc rans_byte_cuda_cleaned.cu -o rans_test
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <ostream>
#include <cassert>
#include <iostream>

#ifdef assert
#define RansAssert assert
#else
#define RansAssert(x)
#endif

#define RANS_BYTE_L (1u << 23) // lower bound of normalization interval

#define SCALE_BITS 11             // 4096 total probability space
#define ALPHABET_SIZE 256

__device__ __constant__ uint16_t d_freq[ALPHABET_SIZE];
__device__ __constant__ uint32_t d_cum[ALPHABET_SIZE];
__device__ __constant__ uint8_t d_lookup[1 << SCALE_BITS];

// --------------------------------------------------------------------------
// Fast encoder/decoder symbol structs and helpers (same as original).
typedef struct {
    uint32_t x_max;
    uint32_t rcp_freq;
    uint32_t bias;
    uint16_t cmpl_freq;
    uint16_t rcp_shift;
} RansEncSymbol;

// Fast encode using symbol.
// This is the core encoding function, kept from the original.
__device__ static inline void RansEncPutSymbol(uint32_t* r, uint8_t** pptr, RansEncSymbol const* sym)
{
    RansAssert(sym->x_max != 0);

    uint32_t x = *r;
    uint32_t x_max = sym->x_max;
    if (x >= x_max) {
        uint8_t* ptr = *pptr;
        do {
            *--ptr = (uint8_t)(x & 0xff);
            x >>= 8;
        } while (x >= x_max);
        *pptr = ptr;
    }

    // x = freq * floor(x / freq) + (x % freq)
    // This is the fast alternative using precomputed rcp_freq
    uint32_t q = (uint32_t)(((uint64_t)x * sym->rcp_freq) >> 32) >> sym->rcp_shift;
    *r = x + sym->bias + q * sym->cmpl_freq;
}


__global__ void encode_kernel(uint8_t* outbuf, const uint8_t* in_data,
    size_t buf_stride, int nthreads, int symbols_per_thread)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint8_t* mybuf = outbuf + tid * buf_stride;
    uint8_t* ptr = mybuf + buf_stride;
    uint32_t s = RANS_BYTE_L;
    const uint8_t* seq = in_data + tid * symbols_per_thread;

    for (int i = symbols_per_thread - 1; i >= 0; --i) {
        uint8_t sym = seq[i];
        uint32_t start = d_cum[sym];
        uint32_t freq = d_freq[sym];

        RansEncSymbol symb;

        // --- Inlined RansEncSymbolInit ---
        {
            symb.x_max = ((RANS_BYTE_L >> SCALE_BITS) << 8) * freq;
            symb.cmpl_freq = (uint16_t)((1 << SCALE_BITS) - freq);
            if (freq < 2) {
                symb.rcp_freq = ~0u;
                symb.rcp_shift = 0;
                symb.bias = start + (1 << SCALE_BITS) - 1;
            }
            else {
                uint32_t shift = 0;
                while (freq > (1u << shift))
                    shift++;

                symb.rcp_freq = (uint32_t)(((1ull << (shift + 31)) + freq - 1) / freq);
                symb.rcp_shift = shift - 1;
                symb.bias = start;
            }
        }
        // --- End Inlined RansEncSymbolInit ---

        RansEncPutSymbol(&s, &ptr, &symb);
    }

    // --- Inlined RansEncFlush ---
    {
        uint32_t x = s;
        ptr -= 4;
        ptr[0] = (uint8_t)(x >> 0);
        ptr[1] = (uint8_t)(x >> 8);
        ptr[2] = (uint8_t)(x >> 16);
        ptr[3] = (uint8_t)(x >> 24);
        // *pptr = ptr; // Not needed, ptr is local
    }
    // --- End Inlined RansEncFlush ---

    // Store size at start of buffer
    uint32_t bytes_used = (uint32_t)(buf_stride - (ptr - mybuf));
    mybuf[0] = (uint8_t)(bytes_used);
    mybuf[1] = (uint8_t)(bytes_used >> 8);
    mybuf[2] = (uint8_t)(bytes_used >> 16);
    mybuf[3] = (uint8_t)(bytes_used >> 24);

    // Copy encoded data to the start (after the 4-byte size)
    uint8_t* emitted_start = ptr;
    uint8_t* dst =
        mybuf + 4;
    for (uint32_t i = 0; i < bytes_used; ++i)
        dst[i] = emitted_start[i];
}


__global__ void decode_kernel(uint8_t* inbuf, size_t buf_stride,
    int nthreads, int symbols_per_thread, uint8_t* out_symbols)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load freq tables to shared memory
    __shared__ uint16_t s_freq[ALPHABET_SIZE];
    __shared__ uint32_t s_cum[ALPHABET_SIZE];
    for (int i = 0; i < 8; i++) {
        int idx = threadIdx.x + i * 32; // 8 * 32 = 256
        if (idx < ALPHABET_SIZE) {
            s_freq[idx] = d_freq[idx];
            s_cum[idx] = d_cum[idx];
        }
    }
    __syncthreads();

    uint8_t* ptr = inbuf + tid * buf_stride + 4;
    uint32_t st;

    // --- Inlined RansDecInit ---
    {
        st = ptr[0] << 0;
        st |= ptr[1] << 8;
        st |= ptr[2] << 16;
        st |= ptr[3] << 24;
        ptr += 4;
    }
    // --- End Inlined RansDecInit ---

    for (int i = 0; i < symbols_per_thread; ++i) {

        // --- Inlined RansDecGet ---
        uint32_t cum_val = st & ((1u << SCALE_BITS) - 1);

        // --- Inlined find_symbol ---
        uint8_t sym = d_lookup[cum_val]; // O(1) lookup
        out_symbols[tid * symbols_per_thread + i] = sym;

        uint32_t start = s_cum[sym];
        uint32_t freq = s_freq[sym];

        // --- Inlined RansDecAdvance ---
        {
            uint32_t mask = (1u << SCALE_BITS) - 1;
            uint32_t x = st;

            x = freq * (x >> SCALE_BITS) + (x & mask) - start;

            // Renormalize
            if (x < RANS_BYTE_L) {
                do x = (x << 8) | *ptr++; while (x < RANS_BYTE_L);
            }
            st = x;
        }
        // --- End Inlined RansDecAdvance ---
    }
}


// --------------------------------------------------------------------------
// Host-side code
// --------------------------------------------------------------------------

struct SymbolStats
{
    uint32_t freqs[256];
    uint32_t cum_freqs[257];

    void count_freqs(uint8_t const* in, size_t nbytes);
    void calc_cum_freqs();
    void normalize_freqs(uint32_t target_total);
};

void SymbolStats::count_freqs(uint8_t const* in, size_t nbytes)
{
    for (int i = 0; i < 256; i++)
        freqs[i] = 0;

    for (size_t i = 0; i < nbytes; i++)
        freqs[in[i]]++;
}

void SymbolStats::calc_cum_freqs()
{
    cum_freqs[0] = 0;
    for (int i = 0; i < 256; i++)
        cum_freqs[i + 1] = cum_freqs[i] + freqs[i];
}

void SymbolStats::normalize_freqs(uint32_t target_total)
{
    assert(target_total >= 256);

    calc_cum_freqs();
    uint32_t cur_total = cum_freqs[256];

    // resample distribution based on cumulative freqs
    for (int i = 1; i <= 256; i++)
        cum_freqs[i] = ((uint64_t)target_total * cum_freqs[i]) / cur_total;

    // if we nuked any non-0 frequency symbol to 0, we need to steal
    // the range to make the frequency nonzero from elsewhere.
    for (int i = 0; i < 256; i++) {
        if (freqs[i] && cum_freqs[i + 1] == cum_freqs[i]) {
            // symbol i was set to zero freq
            // find best symbol to steal frequency from
            uint32_t best_freq = ~0u;
            int best_steal = -1;
            for (int j = 0; j < 256; j++) {
                uint32_t freq = cum_freqs[j + 1] - cum_freqs[j];
                if (freq > 1 && freq < best_freq) {
                    best_freq = freq;
                    best_steal = j;
                }
            }
            assert(best_steal != -1);

            // and steal from it!
            if (best_steal < i) {
                for (int j = best_steal + 1; j <= i; j++)
                    cum_freqs[j]--;
            }
            else {
                assert(best_steal > i);
                for (int j = i + 1; j <= best_steal; j++)
                    cum_freqs[j]++;
            }
        }
    }

    // calculate updated freqs and make sure we didn't screw anything up
    assert(cum_freqs[0] == 0 && cum_freqs[256] == target_total);
    for (int i = 0; i < 256; i++) {
        if (freqs[i] == 0)
            assert(cum_freqs[i + 1] == cum_freqs[i]);
        else
            assert(cum_freqs[i + 1] > cum_freqs[i]);

        // calc updated freq
        freqs[i] = cum_freqs[i + 1] - cum_freqs[i];
    }
}


int test(uint8_t* data, size_t length)
{
    const int nthreads = 32; // blockDim.x
    int nblocks = 4048;      // gridDim.x

    // Calculate initial total_threads
    int total_threads = nthreads * nblocks;

    // Check if length is divisible
    if (length % total_threads != 0) {
        printf("Error: Data length %zu is not divisible by total threads %d\n", length, total_threads);
        // Adjust nblocks
        nblocks = (int)(length / (nthreads * 256)); // Example: 256 symbols per thread
        if (nblocks == 0) nblocks = 1;

        printf("Adjusting nblocks to %d\n", nblocks);

        // *** FIX: Recalculate total_threads with the new nblocks ***
        total_threads = nthreads * nblocks;
    }

    // Ensure length is divisible
    length = (length / total_threads) * total_threads;
    if (length == 0) {
        printf("Error: Data length too small for this thread configuration.\n");
        return 1;
    }

    const int symbols_per_thread = (int)(length / total_threads);
    const size_t buf_stride = symbols_per_thread; // This is small, but it's per-thread *compressed* size

    // Total symbols processed by all threads
    const size_t total_symbols = (size_t)total_threads * symbols_per_thread;

    // Total buffer size for compressed data (one buffer per thread)
    const size_t total_buf = buf_stride * total_threads;

    printf("Running test with %d blocks * %d threads = %d total threads.\n", nblocks, nthreads, total_threads);
    printf("Symbols per thread: %d. Total symbols: %zu.\n", symbols_per_thread, total_symbols);

    // --- Build adaptive model
    SymbolStats stats;
    stats.count_freqs(data, total_symbols); // Use total_symbols
    stats.normalize_freqs(1 << SCALE_BITS);

    uint16_t host_freq[ALPHABET_SIZE];
    uint32_t host_cum[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i) {
        host_freq[i] = (uint16_t)stats.freqs[i];
        host_cum[i] = stats.cum_freqs[i];
    }

    cudaMemcpyToSymbol(d_freq, host_freq, sizeof(host_freq));
    cudaMemcpyToSymbol(d_cum, host_cum, sizeof(host_cum));

    // Build host-side lookup table
    uint8_t* h_lookup = (uint8_t*)malloc(sizeof(uint8_t) * (1 << SCALE_BITS));
    if (!h_lookup) {
        printf("Failed to alloc h_lookup\n");
        return 1;
    }
    for (int s = 0; s < 256; ++s) {
        for (uint32_t i = stats.cum_freqs[s]; i < stats.cum_freqs[s + 1]; ++i) {
            h_lookup[i] = (uint8_t)s;
        }
    }

    cudaMemcpyToSymbol(d_lookup, h_lookup, sizeof(uint8_t) * (1 << SCALE_BITS));
    free(h_lookup); // Free host table after copy

    // --- Allocate buffers
    uint8_t* d_buf, * d_in, * d_out;
    cudaMalloc(&d_buf, total_buf);
    cudaMemset(d_buf, 0, total_buf);
    cudaMalloc(&d_in, total_symbols);
    cudaMemcpy(d_in, data, total_symbols, cudaMemcpyHostToDevice);
    cudaMalloc(&d_out, total_symbols);


    // --- Encode timing
    cudaEvent_t start_encode, stop_encode;
    cudaEventCreate(&start_encode);
    cudaEventCreate(&stop_encode);
    cudaEventRecord(start_encode);

    encode_kernel << <nblocks, nthreads >> > (d_buf, d_in, buf_stride, nthreads, symbols_per_thread);

    cudaEventRecord(stop_encode);
    cudaEventSynchronize(stop_encode);

    float encode_ms = 0.0f;
    cudaEventElapsedTime(&encode_ms, start_encode, stop_encode);

    // --- Decode timing
    cudaEvent_t start_decode, stop_decode;
    cudaEventCreate(&start_decode);
    cudaEventCreate(&stop_decode);
    cudaEventRecord(start_decode);

    decode_kernel << <nblocks, nthreads >> > (d_buf, buf_stride, nthreads, symbols_per_thread, d_out);

    cudaEventRecord(stop_decode);
    cudaEventSynchronize(stop_decode);

    float decode_ms = 0.0f;
    cudaEventElapsedTime(&decode_ms, start_decode, stop_decode);

    // --- Verify
    uint8_t* h_out = (uint8_t*)malloc(total_symbols);
    if (!h_out) {
        printf("Failed to alloc h_out\n");
        return 1;
    }
    cudaMemcpy(h_out, d_out, total_symbols, cudaMemcpyDeviceToHost);

    bool ok = true;
    for (size_t i = 0; i < total_symbols; ++i) {
        if (h_out[i] != data[i]) {
            ok = false;
            printf("Mismatch at index %zu: expected %d, got %d\n", i, data[i], h_out[i]);
            break;
        }
    }

    // --- Compute throughput (MB/s)
    double data_mb = static_cast<double>(total_symbols) / (1024.0 * 1024.0);
    double encode_throughput = data_mb / (encode_ms / 1000.0);
    double decode_throughput = data_mb / (decode_ms / 1000.0);

    // --- Print results
    printf("\nVerification: %s\n", ok ? "PASS" : "FAIL");
    printf("Data size: %.2f MB\n", data_mb);
    printf("Encode time: %.3f ms (%.2f MB/s)\n", encode_ms, encode_throughput);
    printf("Decode time: %.3f ms (%.2f MB/s)\n", decode_ms, decode_throughput);

    // --- Cleanup
    cudaFree(d_buf);
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_out);

    return ok ? 0 : 1;
}