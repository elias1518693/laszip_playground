// rans_byte_cuda.cu
// CUDA translation of "Simple byte-aligned rANS encoder/decoder" (Fabian 'ryg' Giesen)
// Single-file example with per-thread rANS encode/decode demo.
// Compile: nvcc rans_byte_cuda.cu -o rans_test
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

#define RANS_BYTE_L (1u << 23)  // lower bound of normalization interval


#define SCALE_BITS 14                  // 4096 total probability space
#define ALPHABET_SIZE 256

__device__ __constant__ uint16_t d_freq[ALPHABET_SIZE];
__device__ __constant__ uint32_t d_cum[ALPHABET_SIZE];


__device__ __host__ static inline void symbol_to_range(uint8_t sym, uint32_t* start, uint32_t* freq)
{
    *start = d_cum[sym];
    *freq = d_freq[sym];
}

// Renormalize encoder (internal).
__device__ static inline uint32_t RansEncRenorm(uint32_t x, uint8_t** pptr, uint32_t freq, uint32_t scale_bits)
{
    uint32_t x_max = ((RANS_BYTE_L >> scale_bits) << 8) * freq;
    if (x >= x_max) {
        uint8_t* ptr = *pptr;
        do {
            *--ptr = (uint8_t)(x & 0xff);
            x >>= 8;
        } while (x >= x_max);
        *pptr = ptr;
    }
    return x;
}

// Put symbol (generic).
__device__ static inline void RansEncPut(uint32_t* r, uint8_t** pptr, uint32_t start, uint32_t freq, uint32_t scale_bits)
{
    uint32_t x = RansEncRenorm(*r, pptr, freq, scale_bits);
    *r = ((x / freq) << scale_bits) + (x % freq) + start;
}

// Flush encoder state to stream (writes 4 bytes little-endian).
__device__ static inline void RansEncFlush(uint32_t* r, uint8_t** pptr)
{
    uint32_t x = *r;
    uint8_t* ptr = *pptr;

    ptr -= 4;
    ptr[0] = (uint8_t)(x >> 0);
    ptr[1] = (uint8_t)(x >> 8);
    ptr[2] = (uint8_t)(x >> 16);
    ptr[3] = (uint8_t)(x >> 24);

    *pptr = ptr;
}

// Decoder init: read 4 bytes to form initial state. Note: decoder reads forward in stream.
__device__ static inline void RansDecInit(uint32_t* r, uint8_t** pptr)
{
    uint32_t x;
    uint8_t* ptr = *pptr;

    x = ptr[0] << 0;
    x |= ptr[1] << 8;
    x |= ptr[2] << 16;
    x |= ptr[3] << 24;
    ptr += 4;

    *pptr = ptr;
    *r = x;
}

// Return current cumulative frequency (low bits).
__device__ static inline uint32_t RansDecGet(uint32_t* r, uint32_t scale_bits)
{
    return *r & ((1u << scale_bits) - 1);
}

// Decoder advance (pops symbol).
__device__ static inline void RansDecAdvance(uint32_t* r, uint8_t** pptr, uint32_t start, uint32_t freq, uint32_t scale_bits)
{
    uint32_t mask = (1u << scale_bits) - 1;

    uint32_t x = *r;
    x = freq * (x >> scale_bits) + (x & mask) - start;

    if (x < RANS_BYTE_L) {
        uint8_t* ptr = *pptr;
        do x = (x << 8) | *ptr++; while (x < RANS_BYTE_L);
        *pptr = ptr;
    }

    *r = x;
}

// --------------------------------------------------------------------------
// Fast encoder/decoder symbol structs and helpers (same as original).
typedef struct {
    uint32_t x_max;
    uint32_t rcp_freq;
    uint32_t bias;
    uint16_t cmpl_freq;
    uint16_t rcp_shift;
} RansEncSymbol;

typedef struct {
    uint16_t start;
    uint16_t freq;
} RansDecSymbol;

__device__ static inline void RansEncSymbolInit(RansEncSymbol* s, uint32_t start, uint32_t freq, uint32_t scale_bits)
{   s->x_max = ((RANS_BYTE_L >> scale_bits) << 8) * freq;
    s->cmpl_freq = (uint16_t)((1 << scale_bits) - freq);
    if (freq < 2) {
        s->rcp_freq = ~0u;
        s->rcp_shift = 0;
        s->bias = start + (1 << scale_bits) - 1;
    }
    else {
        uint32_t shift = 0;
        while (freq > (1u << shift))
            shift++;

        s->rcp_freq = (uint32_t)(((1ull << (shift + 31)) + freq - 1) / freq);
        s->rcp_shift = shift - 1;
        s->bias = start;
    }
}

// Fast encode using symbol.
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
    //x = freq * floor(x / freq) + (x % freq)
    uint32_t q = (uint32_t)(((uint64_t)x * sym->rcp_freq) >> 32) >> sym->rcp_shift;
    *r = x + sym->bias + q * sym->cmpl_freq;
}

__device__ static inline void RansDecAdvanceSymbol(uint32_t* r, uint8_t** pptr, RansDecSymbol const* sym, uint32_t scale_bits)
{
    RansDecAdvance(r, pptr, sym->start, sym->freq, scale_bits);
}

__device__ static inline void RansDecAdvanceStep(uint32_t* r, uint32_t start, uint32_t freq, uint32_t scale_bits)
{
    uint32_t mask = (1u << scale_bits) - 1;
    uint32_t x = *r;
    *r = freq * (x >> scale_bits) + (x & mask) - start;
}

__device__ static inline void RansDecAdvanceSymbolStep(uint32_t* r, RansDecSymbol const* sym, uint32_t scale_bits)
{
    RansDecAdvanceStep(r, sym->start, sym->freq, scale_bits);
}

__device__ static inline void RansDecRenorm(uint32_t* r, uint8_t** pptr)
{
    uint32_t x = *r;
    if (x < RANS_BYTE_L) {
        uint8_t* ptr = *pptr;
        do x = (x << 8) | *ptr++; while (x < RANS_BYTE_L);
        *pptr = ptr;
    }
    *r = x;
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
        RansEncSymbolInit(&symb, start, freq, SCALE_BITS);
        RansEncPutSymbol(&s, &ptr, &symb);
    }

    RansEncFlush(&s, &ptr);
	//store size at start of buffer
    uint32_t bytes_used = (uint32_t)(buf_stride - (ptr - mybuf));
    mybuf[0] = (uint8_t)(bytes_used);
    mybuf[1] = (uint8_t)(bytes_used >> 8);
    mybuf[2] = (uint8_t)(bytes_used >> 16);
    mybuf[3] = (uint8_t)(bytes_used >> 24);

    uint8_t* emitted_start = ptr;
    uint8_t* dst = 
        mybuf + 4;
    for (uint32_t i = 0; i < bytes_used; ++i)
        dst[i] = emitted_start[i];
}

__device__ __constant__ uint8_t d_lookup[1 << SCALE_BITS];
__device__ __forceinline__ uint8_t find_symbol(uint32_t cum)
{
    return d_lookup[cum];
}

__global__ void decode_kernel(uint8_t* inbuf, size_t buf_stride,
    int nthreads, int symbols_per_thread, uint8_t* out_symbols)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load freq tables to shared memory
    __shared__ uint16_t s_freq[ALPHABET_SIZE];
    __shared__ uint32_t s_cum[ALPHABET_SIZE];
    for(int i = 0; i < 8; i++) {
        int idx = threadIdx.x + i * 32;
        if (idx < ALPHABET_SIZE) {
            s_freq[idx] = d_freq[idx];
            s_cum[idx] = d_cum[idx];
        }
	}
    __syncthreads();

    uint8_t* ptr = inbuf + tid * buf_stride + 4;
    uint32_t st;
    RansDecInit(&st, &ptr);

//#pragma unroll
    for (int i = 0; i < symbols_per_thread; ++i) {
        uint32_t cum_val = RansDecGet(&st, SCALE_BITS);
        uint8_t sym = d_lookup[cum_val];    // O(1) lookup
        out_symbols[tid * symbols_per_thread + i] = sym;

        uint32_t start = s_cum[sym];
        uint32_t freq = s_freq[sym];
        RansDecAdvance(&st, &ptr, start, freq, SCALE_BITS);
    }
}



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
    //
    // this is not at all optimal, i'm just doing the first thing that comes to mind.
    for (int i = 0; i < 256; i++) {
        if (freqs[i] && cum_freqs[i + 1] == cum_freqs[i]) {
            // symbol i was set to zero freq

            // find best symbol to steal frequency from (try to steal from low-freq ones)
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
    const int nthreads = 32;
    int nblocks = 4048;
    const int symbols_per_thread = length / (nthreads*nblocks);
    const size_t buf_stride = symbols_per_thread;
    const size_t total_buf = buf_stride * nthreads * nblocks;

    // --- Build adaptive model
    SymbolStats stats;
    stats.count_freqs(data, length);
    stats.normalize_freqs(1 << SCALE_BITS);

    uint16_t host_freq[ALPHABET_SIZE];
    uint32_t host_cum[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i) {
        host_freq[i] = (uint16_t)stats.freqs[i];
        host_cum[i] = stats.cum_freqs[i];
    }

    cudaMemcpyToSymbol(d_freq, host_freq, sizeof(host_freq));
    cudaMemcpyToSymbol(d_cum, host_cum, sizeof(host_cum));
    uint8_t h_lookup[1 << SCALE_BITS];
    for (int s = 0; s < 256; ++s) {
        for (uint32_t i = stats.cum_freqs[s]; i < stats.cum_freqs[s + 1]; ++i) {
            h_lookup[i] = (uint8_t)s;
        }
    }

    cudaMemcpyToSymbol(d_lookup, h_lookup, sizeof(h_lookup));
    // --- Allocate buffers
    uint8_t* d_buf, * d_in, * d_out;
    cudaMalloc(&d_buf, total_buf);
    cudaMemset(d_buf, 0, total_buf);
    cudaMalloc(&d_in, nthreads * symbols_per_thread * nblocks);
    cudaMemcpy(d_in, data, nthreads * symbols_per_thread * nblocks, cudaMemcpyHostToDevice);
    cudaMalloc(&d_out, nthreads * symbols_per_thread * nblocks);

 

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
    uint8_t* h_out = (uint8_t*)malloc(nthreads * symbols_per_thread * nblocks);
    cudaMemcpy(h_out, d_out, nthreads * symbols_per_thread * nblocks, cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < nthreads * symbols_per_thread * nblocks; ++i)
        if (h_out[i] != data[i]) { ok = false; break; }

    // --- Compute throughput (MB/s)
    double data_mb = static_cast<double>(length) / (1024.0 * 1024.0);
    double encode_throughput = data_mb / (encode_ms / 1000.0);
    double decode_throughput = data_mb / (decode_ms / 1000.0);

    // --- Print results
    printf("\nVerification: %s\n", ok ? "PASS" : "FAIL");
    printf("Encode time: %.3f ms (%.2f MB/s)\n", encode_ms, encode_throughput);
    printf("Decode time: %.3f ms (%.2f MB/s)\n", decode_ms, decode_throughput);

    // --- Cleanup
    cudaFree(d_buf);
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_out);

    return ok ? 0 : 1;
}
