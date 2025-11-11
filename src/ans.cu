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
#include <vector>
#ifdef assert
#define RansAssert assert
#else
#define RansAssert(x)
#endif

#define RANS_BYTE_L (1u << 23) // lower bound of normalization interval

#define SCALE_BITS 11             // 4096 total probability space
#define ALPHABET_SIZE 256

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

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
    size_t buf_stride, size_t block_size, int symbols_per_thread, size_t total_symbols, uint32_t* bytes_used)
{
    // --- Warp-Parallel Coalesced Indexing ---
    int tid_in_block = threadIdx.x;
    int total_threads_in_block = blockDim.x;
    size_t global_tid = (size_t)blockIdx.x * total_threads_in_block + tid_in_block;
    size_t block_start_addr = (size_t)blockIdx.x * block_size;
    // --- End Indexing ---

    uint8_t* mybuf = outbuf + global_tid * buf_stride;
    uint8_t* ptr = mybuf + buf_stride;
    uint32_t s = RANS_BYTE_L;
    const uint8_t* seq = in_data; // Use base pointer

    for (int i = symbols_per_thread - 1; i >= 0; --i) {
        // --- Coalesced memory access ---
        // 'i' is the stripe index, 'tid_in_block' is the offset in the stripe
        size_t addr = block_start_addr + (size_t)i * total_threads_in_block + tid_in_block;

        // Handle padding on the last block
        if (addr >= total_symbols) {
            continue;
        }
        uint8_t sym = seq[addr];
        // --- End Coalesced Access ---

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

    size_t length = (size_t)((mybuf + buf_stride) - ptr);

    // 2. Store this length as the 'bytes_used' for this thread
    bytes_used[global_tid] = (uint32_t)length;



    // Copy encoded data to the start
}


__global__ void decode_kernel(uint8_t* inbuf,
    size_t block_size, int symbols_per_thread, uint8_t* out_symbols, size_t total_symbols, uint32_t* offsets)
{
    // --- Warp-Parallel Coalesced Indexing ---
    int tid_in_block = threadIdx.x;
    int total_threads_in_block = blockDim.x;
    size_t global_tid = (size_t)blockIdx.x * total_threads_in_block + tid_in_block;
    size_t block_start_addr = (size_t)blockIdx.x * block_size;
    // --- End Indexing ---

    // Load freq tables to shared memory
    __shared__ uint16_t s_freq[ALPHABET_SIZE];
    __shared__ uint32_t s_cum[ALPHABET_SIZE];

            s_freq[tid_in_block] = d_freq[tid_in_block];
            s_cum[tid_in_block] = d_cum[tid_in_block];

    __syncthreads();

    uint8_t* ptr = inbuf + offsets[global_tid]; // Use global_tid

    uint32_t st;

    {
        st = (uint32_t)ptr[0] << 0;
        st |= (uint32_t)ptr[1] << 8;
        st |= (uint32_t)ptr[2] << 16;
        st |= (uint32_t)ptr[3] << 24;

        ptr += 4;
    }

    for (int i = 0; i < symbols_per_thread; ++i) {

        // --- Inlined RansDecGet ---
        uint32_t cum_val = st & ((1u << SCALE_BITS) - 1);

        // --- Inlined find_symbol ---
        uint8_t sym = d_lookup[cum_val]; // O(1) lookup

        // --- Coalesced memory access ---
        size_t addr = block_start_addr + (size_t)i * total_threads_in_block + tid_in_block;

        // Handle padding on the last block
        if (addr >= total_symbols) {
            continue;
        }
        out_symbols[addr] = sym;
        // --- End Coalesced Access ---

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


int test(std::vector<std::vector<uint8_t>>& data, size_t length)
{
    // --- New Warp-Parallel Configuration ---
    const int nthreads = 256;
    const size_t LOG_BLOCK_SIZE = 20;
    const size_t BLOCK_SIZE = 1 << LOG_BLOCK_SIZE;
    const int symbols_per_thread = BLOCK_SIZE / nthreads; 
    const size_t buf_stride = symbols_per_thread * 8; 
    // --- End Configuration ---

    // Pad total_symbols to be a multiple of BLOCK_SIZE for full blocks
    size_t total_symbols = (length + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    int nblocks = (int)(total_symbols / BLOCK_SIZE);
    std::vector<uint32_t> bytes_used(nthreads * nblocks);
    // Total threads based on new block grid
    int total_threads = nblocks * nthreads;

    // Total buffer size for compressed data (one buffer per thread)
    const size_t total_buf = (size_t)total_threads * buf_stride;

    printf("Running test with %d blocks * %d threads = %d total threads.\n", nblocks, nthreads, total_threads);
    printf("Data Block Size: %zu symbols. Symbols per thread: %d.\n", BLOCK_SIZE, symbols_per_thread);
    printf("Total Symbols (padded): %zu. Original length: %zu.\n", total_symbols, length);

    // --- Build adaptive model
    SymbolStats stats;
    stats.count_freqs(data[0].data(), length); // Use original length for stats
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
	uint32_t* d_bytes_used;
    cudaMalloc(&d_buf, total_buf);
    cudaMemset(d_buf, 0, total_buf);

    cudaMalloc(&d_bytes_used, bytes_used.size() * sizeof(uint32_t));
    cudaMemset(d_bytes_used, 0, bytes_used.size() * sizeof(uint32_t));

    // Allocate for padded size and clear to 0
    cudaMalloc(&d_in, total_symbols);
    cudaMemset(d_in, 0, total_symbols);
    cudaMemcpy(d_in, data[0].data(), length, cudaMemcpyHostToDevice); // Copy only valid data

    gpuErrchk(cudaMalloc(&d_out, total_symbols * sizeof(uint8_t)));
    gpuErrchk(cudaMemset(d_out, 0, total_symbols * sizeof(uint8_t)));
    // --- Encode timing
    cudaEvent_t start_encode, stop_encode;
    cudaEventCreate(&start_encode);
    cudaEventCreate(&stop_encode);
    cudaEventRecord(start_encode);

    // --- Launch with new parallel strategy ---
    encode_kernel << <nblocks, nthreads >> > (

        d_buf, d_in, buf_stride, BLOCK_SIZE, symbols_per_thread, total_symbols, d_bytes_used);

    cudaEventRecord(stop_encode);
    cudaEventSynchronize(stop_encode);
    cudaMemcpy(bytes_used.data(),          
        d_bytes_used,                
        bytes_used.size() * sizeof(uint32_t),
        cudaMemcpyDeviceToHost);
    size_t total_compressed_size = 0;
    for (uint32_t size : bytes_used) {
        total_compressed_size += size;
    }

    // 2. Allocate a single, contiguous buffer on the host (CPU)
    std::vector<uint8_t> h_final_output(total_compressed_size);

    uint8_t* host_ptr = h_final_output.data(); // Current position in host buffer
    size_t num_chunks = bytes_used.size(); // e.g., total_threads

    for (size_t i = 0; i < num_chunks; ++i) {
        if (bytes_used[i] > 0) {

            // --- THIS IS THE FIX ---
            // Calculate the correct source pointer on the GPU.
            // It's at the end of the buffer, minus its own length.
            uint8_t* gpu_source_ptr = d_buf + (i * buf_stride) + buf_stride - bytes_used[i];
            // --- END OF FIX ---

            // Destination: host_ptr
            // Size: bytes_used[i]

            // This cudaMemcpy is now grabbing the *correct* data
            gpuErrchk(cudaMemcpy(host_ptr,
                gpu_source_ptr,
                bytes_used[i],
                cudaMemcpyDeviceToHost));

            // Move the host pointer forward for the next chunk
            host_ptr += bytes_used[i];
        }
    }


    std::vector<uint32_t> h_offsets(num_chunks);
    uint32_t current_offset = 0;
    for (size_t i = 0; i < num_chunks; ++i) {
        h_offsets[i] = current_offset;
        current_offset += bytes_used[i];
    }

    printf("Encoded length: %zu.\n", current_offset);
    float encode_ms = 0.0f;
    cudaEventElapsedTime(&encode_ms, start_encode, stop_encode);

    uint8_t* coherrent_byte;
    cudaMalloc(&coherrent_byte, total_compressed_size);
    cudaMemcpy(coherrent_byte,
        h_final_output.data(),
        total_compressed_size,
        cudaMemcpyHostToDevice);

    uint32_t* d_offsets;
    gpuErrchk(cudaMalloc(&d_offsets, h_offsets.size() * sizeof(uint32_t)));
    gpuErrchk(cudaMemcpy(d_offsets,
        h_offsets.data(),
        h_offsets.size() * sizeof(uint32_t),
        cudaMemcpyHostToDevice));

    // --- Decode timing
    cudaEvent_t start_decode, stop_decode;
    gpuErrchk(cudaEventCreate(&start_decode));
    gpuErrchk(cudaEventCreate(&stop_decode));
    gpuErrchk(cudaEventRecord(start_decode));

    // --- Launch with new parallel strategy ---
    decode_kernel << <nblocks, nthreads >> > (
        coherrent_byte, BLOCK_SIZE, symbols_per_thread, d_out, total_symbols, d_offsets);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    cudaEventRecord(stop_decode);
    cudaEventSynchronize(stop_decode);

    float decode_ms = 0.0f;
    cudaEventElapsedTime(&decode_ms, start_decode, stop_decode);

    // --- Verify
    uint8_t* h_out = (uint8_t*)malloc(total_symbols); // Alloc for padded size
    if (!h_out) {
        printf("Failed to alloc h_out\n");
        return 1;
    }
    cudaMemcpy(h_out, d_out, total_symbols, cudaMemcpyDeviceToHost);

    bool ok = true;
    for (size_t i = 0; i < length; ++i) { // Verify only original length
        if (h_out[i] != data[0][i]) {
            ok = false;
            printf("Mismatch at index %zu: expected %d, got %d\n", i, data[0][i], h_out[i]);
            break;
        }
    }

    // --- Compute throughput (MB/s)
    double data_mb = static_cast<double>(length) / (1024.0 * 1024.0); // Use original length
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