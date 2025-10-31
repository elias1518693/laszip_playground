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

#ifdef assert
#define RansAssert assert
#else
#define RansAssert(x)
#endif

#define RANS_BYTE_L (1u << 23)  // lower bound of normalization interval

typedef uint32_t RansState;

// Mark functions to be available on host and device
#define HOST_DEVICE __host__ __device__


#define SCALE_BITS 14                  // 4096 total probability space
#define ALPHABET_SIZE 256

__device__ __constant__ uint16_t d_freq[ALPHABET_SIZE];
__device__ __constant__ uint32_t d_cum[ALPHABET_SIZE];
// Initialize encoder.
HOST_DEVICE static inline void RansEncInit(RansState* r)
{
    *r = RANS_BYTE_L;
}

__device__ __host__ static inline void symbol_to_range(uint8_t sym, uint32_t* start, uint32_t* freq)
{
    *start = d_cum[sym];
    *freq = d_freq[sym];
}

// Renormalize encoder (internal).
HOST_DEVICE static inline RansState RansEncRenorm(RansState x, uint8_t** pptr, uint32_t freq, uint32_t scale_bits)
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
HOST_DEVICE static inline void RansEncPut(RansState* r, uint8_t** pptr, uint32_t start, uint32_t freq, uint32_t scale_bits)
{
    RansState x = RansEncRenorm(*r, pptr, freq, scale_bits);
    *r = ((x / freq) << scale_bits) + (x % freq) + start;
}

// Flush encoder state to stream (writes 4 bytes little-endian).
HOST_DEVICE static inline void RansEncFlush(RansState* r, uint8_t** pptr)
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
HOST_DEVICE static inline void RansDecInit(RansState* r, uint8_t** pptr)
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
HOST_DEVICE static inline uint32_t RansDecGet(RansState* r, uint32_t scale_bits)
{
    return *r & ((1u << scale_bits) - 1);
}

// Decoder advance (pops symbol).
HOST_DEVICE static inline void RansDecAdvance(RansState* r, uint8_t** pptr, uint32_t start, uint32_t freq, uint32_t scale_bits)
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

HOST_DEVICE static inline void RansEncSymbolInit(RansEncSymbol* s, uint32_t start, uint32_t freq, uint32_t scale_bits)
{
    RansAssert(scale_bits <= 16);
    RansAssert(start <= (1u << scale_bits));
    RansAssert(freq <= (1u << scale_bits) - start);

    s->x_max = ((RANS_BYTE_L >> scale_bits) << 8) * freq;
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

HOST_DEVICE static inline void RansDecSymbolInit(RansDecSymbol* s, uint32_t start, uint32_t freq)
{
    RansAssert(start <= (1 << 16));
    RansAssert(freq <= (1 << 16) - start);
    s->start = (uint16_t)start;
    s->freq = (uint16_t)freq;
}

// Fast encode using symbol.
HOST_DEVICE static inline void RansEncPutSymbol(RansState* r, uint8_t** pptr, RansEncSymbol const* sym)
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

    uint32_t q = (uint32_t)(((uint64_t)x * sym->rcp_freq) >> 32) >> sym->rcp_shift;
    *r = x + sym->bias + q * sym->cmpl_freq;
}

HOST_DEVICE static inline void RansDecAdvanceSymbol(RansState* r, uint8_t** pptr, RansDecSymbol const* sym, uint32_t scale_bits)
{
    RansDecAdvance(r, pptr, sym->start, sym->freq, scale_bits);
}

HOST_DEVICE static inline void RansDecAdvanceStep(RansState* r, uint32_t start, uint32_t freq, uint32_t scale_bits)
{
    uint32_t mask = (1u << scale_bits) - 1;
    uint32_t x = *r;
    *r = freq * (x >> scale_bits) + (x & mask) - start;
}

HOST_DEVICE static inline void RansDecAdvanceSymbolStep(RansState* r, RansDecSymbol const* sym, uint32_t scale_bits)
{
    RansDecAdvanceStep(r, sym->start, sym->freq, scale_bits);
}

HOST_DEVICE static inline void RansDecRenorm(RansState* r, uint8_t** pptr)
{
    uint32_t x = *r;
    if (x < RANS_BYTE_L) {
        uint8_t* ptr = *pptr;
        do x = (x << 8) | *ptr++; while (x < RANS_BYTE_L);
        *pptr = ptr;
    }
    *r = x;
}

// --------------------------------------------------------------------------
// Example usage in CUDA: each thread encodes a tiny sequence of symbols
// using per-thread buffer region to avoid races. This is an illustrative demo
// and not a high-performance production encoder pipeline.

// Simple toy alphabet: 'A' (freq 3), 'B' (freq 1) with scale_bits = 2  (total 4)

// We'll map symbol -> (start, freq):
__device__ HOST_DEVICE static inline void symbol_to_range(char sym, uint32_t* start, uint32_t* freq)
{
    if (sym == 'A') { *start = 0; *freq = 3; }
    else if (sym == 'B') { *start = 3; *freq = 1; }
    else { *start = 0; *freq = 1; } // fallback
}

__global__ void encode_kernel(uint8_t* outbuf, const uint8_t* in_data,
    size_t buf_stride, int nthreads, int symbols_per_thread)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads) return;

    uint8_t* mybuf = outbuf + tid * buf_stride;
    uint8_t* ptr = mybuf + buf_stride;
    RansState s;
    RansEncInit(&s);

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

    uint32_t bytes_used = (uint32_t)(buf_stride - (ptr - mybuf));
    mybuf[0] = (uint8_t)(bytes_used);
    mybuf[1] = (uint8_t)(bytes_used >> 8);
    mybuf[2] = (uint8_t)(bytes_used >> 16);
    mybuf[3] = (uint8_t)(bytes_used >> 24);

    uint8_t* emitted_start = ptr;
    uint8_t* dst = mybuf + 4;
    for (uint32_t i = 0; i < bytes_used; ++i)
        dst[i] = emitted_start[i];
}


__device__ __forceinline__ uint8_t find_symbol(uint32_t cum)
{
    for (int s = 0; s < ALPHABET_SIZE; ++s)
        if (cum < d_cum[s] + d_freq[s])
            return (uint8_t)s;
    return 0;
}

__global__ void decode_kernel(uint8_t* inbuf, size_t buf_stride,
    int nthreads, int symbols_per_thread,
    uint8_t* out_symbols)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads) return;

    uint8_t* mybuf = inbuf + tid * buf_stride;
    uint32_t bytes_used = (uint32_t)mybuf[0]
        | ((uint32_t)mybuf[1] << 8)
        | ((uint32_t)mybuf[2] << 16)
        | ((uint32_t)mybuf[3] << 24);
    uint8_t* ptr = mybuf + 4;
    RansState st;
    RansDecInit(&st, &ptr);

    for (int i = 0; i < symbols_per_thread; ++i) {
        uint32_t cum_val = RansDecGet(&st, SCALE_BITS);
        uint8_t sym = find_symbol(cum_val);
        out_symbols[tid * symbols_per_thread + i] = sym;

        uint32_t start = d_cum[sym];
        uint32_t freq = d_freq[sym];
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
    const int symbols_per_thread = 3;
    const size_t buf_stride = 256;
    const size_t total_buf = buf_stride * nthreads;

    // --- Build adaptive model
    static const uint32_t prob_bits = 14;
    static const uint32_t prob_scale = 1 << prob_bits;
    uint16_t h_freq[256];
    uint32_t h_cum[257];
    SymbolStats stats;
    stats.count_freqs(data, length);
    stats.normalize_freqs(prob_scale);




    uint16_t host_freq[ALPHABET_SIZE];
    uint32_t host_cum[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i) {
        host_freq[i] = (uint16_t)stats.freqs[i];          // safe if target_total <= 65535
        host_cum[i] = stats.cum_freqs[i];               // start for symbol i
    }
    cudaMemcpyToSymbol(d_freq, host_freq, sizeof(host_freq));
    cudaMemcpyToSymbol(d_cum, host_cum, sizeof(host_cum));

    // --- Allocate buffers
    uint8_t* d_buf, * d_in;
    cudaMalloc(&d_buf, total_buf);
    cudaMemset(d_buf, 0, total_buf);
    cudaMalloc(&d_in, nthreads * symbols_per_thread);
    cudaMemcpy(d_in, data, nthreads * symbols_per_thread, cudaMemcpyHostToDevice);

    // --- Encode
    int threadsPerBlock = 32;
    int blocks = (nthreads + threadsPerBlock - 1) / threadsPerBlock;
    encode_kernel << <blocks, threadsPerBlock >> > (d_buf, d_in, buf_stride, nthreads, symbols_per_thread);
    cudaDeviceSynchronize();

    // --- Decode
    uint8_t* d_out;
    cudaMalloc(&d_out, nthreads * symbols_per_thread);
    decode_kernel << <blocks, threadsPerBlock >> > (d_buf, buf_stride, nthreads, symbols_per_thread, d_out);
    cudaDeviceSynchronize();

    // --- Read back decoded symbols
    uint8_t* h_out = (uint8_t*)malloc(nthreads * symbols_per_thread);
    cudaMemcpy(h_out, d_out, nthreads * symbols_per_thread, cudaMemcpyDeviceToHost);

    printf("\nDecoded symbols (per-thread):\n");
    for (int t = 0; t < nthreads; ++t) {
        printf("Thread %d: ", t);
        for (int i = 0; i < symbols_per_thread; ++i)
            printf("%02X ", h_out[t * symbols_per_thread + i]);
        printf("\n");
    }

    // --- Verify correctness
    bool ok = true;
    for (int i = 0; i < nthreads * symbols_per_thread; ++i)
        if (h_out[i] != data[i]) { ok = false; break; }

    printf("\nVerification: %s\n", ok ? "PASS" : "FAIL");

    // Cleanup
    cudaFree(d_buf);
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_out);
    return ok ? 0 : 1;
}