// ans.cu - Adapted for DietGPU Warp-Synchronous Logic
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include "ans.cuh"
// Configuration Constants matching DietGPU defaults
#define PROB_BITS 12 
#define PROB_SCALE (1 << PROB_BITS)
#define STATE_BITS 31
#define ANS_SIGNATURE 0xA75 

// Types from DietGPU context
typedef uint32_t ANSStateT;
typedef uint32_t ANSDecodedT;
typedef uint16_t ANSEncodedT; // Output is usually u16 or u8. Let's use u16 (short) to match 'encodeOneWarp' mask if needed, or u8. 
// Note: In DietGPU ANSEncodedT is usually uint8_t or uint16_t depending on config. 
// For this example, we will output BYTES (uint8_t) to be standard.
typedef uint8_t EncodedT;

#define kANSStateBits 31
#define kANSEncodedBits 8 
#define kANSEncodedMask 0xFF

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// --------------------------------------------------------------------------
// 1. Host Helper: Compute Magic Numbers for Fast Division
// --------------------------------------------------------------------------
// This matches how DietGPU precalculates division optimization
void compute_magic_u32(uint32_t d, uint32_t* m, uint32_t* s) {
    if (d == 0) { *m = 0; *s = 0; return; }
    // We want to compute (n / d) using __umulhi(n, m) >> s
    // For 31-bit state and 12-bit prob, standard recipes apply:
    uint32_t shift = 32 + PROB_BITS;
    *m = (uint32_t)((1ULL << shift) / d);
    *s = shift - 32;
}
__device__ __forceinline__ uint32_t getLaneMaskLt() {
    uint32_t laneId = threadIdx.x & 0x1f;
    return (1 << laneId) - 1;
}
// --------------------------------------------------------------------------
// 2. Device Helpers (From GpuANSEncode.cuh)
// --------------------------------------------------------------------------
__device__ __forceinline__ uint32_t encodeOnePartialWarp(
    bool valid,                  // <--- NEW: Is this lane active?
    ANSStateT& state,
    ANSDecodedT sym,
    uint32_t outOffset,
    EncodedT* __restrict__ out,
    const uint4* __restrict__ smemLookup) {

    // 1. Load Statistics (All threads load, even invalid ones, to avoid divergence issues)
    // We use a safe symbol (0) if invalid to prevent out-of-bounds shared mem access
    auto lookup = smemLookup[valid ? sym : 0];

    uint32_t pdf = lookup.x;
    uint32_t cdf = lookup.y;
    uint32_t div_m1 = lookup.z;
    uint32_t div_shift = lookup.w;

    // 2. Check for Renormalization (Write)
    constexpr ANSStateT kStateCheckMul = 1 << (kANSStateBits - PROB_BITS);
    ANSStateT maxStateCheck = pdf * kStateCheckMul;

    // Crucial: Invalid lanes NEVER write
    bool write = valid && (state >= maxStateCheck);

    // 3. Ballot & Prefix Sum
    // Even invalid threads participate in the ballot (contributing 0)
    auto vote = __ballot_sync(0xffffffff, write);
    auto prefix = __popc(vote & getLaneMaskLt());

    if (write) {
        out[outOffset + prefix] = (EncodedT)(state & kANSEncodedMask);
        state >>= kANSEncodedBits;
    }

    // 4. Update State (Fast Division)
    constexpr uint32_t kProbBitsMul = 1 << PROB_BITS;

    uint32_t t = __umulhi(state, div_m1);
    uint32_t div = (t + state) >> div_shift;
    auto mod = state - (div * pdf);

    // Crucial: Invalid lanes NEVER update state
    state = valid ? (div * kProbBitsMul + mod + cdf) : state;

    return __popc(vote);
}


// Reference: GpuANSEncode.cuh
// We use uint8_t for output (ANSEncodedT)
__device__ __forceinline__ uint32_t encodeOneWarp(
    ANSStateT& state,
    ANSDecodedT sym,
    uint32_t outOffset,
    EncodedT* __restrict__ out,
    const uint4* __restrict__ smemLookup) {

    // lookup.x = pdf, .y = cdf, .z = div_m1, .w = div_shift
    auto lookup = smemLookup[sym];

    uint32_t pdf = lookup.x;
    uint32_t cdf = lookup.y;
    uint32_t div_m1 = lookup.z;
    uint32_t div_shift = lookup.w;

    constexpr ANSStateT kStateCheckMul = 1 << (kANSStateBits - PROB_BITS);

    ANSStateT maxStateCheck = pdf * kStateCheckMul;
    bool write = (state >= maxStateCheck);

    auto vote = __ballot_sync(0xffffffff, write);
    auto prefix = __popc(vote & getLaneMaskLt());

    // Some lanes wish to write out their data
    if (write) {
        out[outOffset + prefix] = (EncodedT)(state & kANSEncodedMask);
        state >>= kANSEncodedBits;
    }

    constexpr uint32_t kProbBitsMul = 1 << PROB_BITS;

    // Optimized division using the magic numbers
    uint32_t t = __umulhi(state, div_m1);
    uint32_t div = (t + state) >> div_shift;
    auto mod = state - (div * pdf);

    // calculating ((state / pdf) << ProbBits) + (state % pdf) + cdf
    state = div * kProbBitsMul + mod + cdf;

    // how many values we actually write to the compressed output
    return __popc(vote);
}

// --------------------------------------------------------------------------
// 3. Encoder Kernel
// --------------------------------------------------------------------------
__global__ void k_encode_warp_interleaved(
    EncodedT* __restrict__ out_buf,
    const uint8_t* __restrict__ in_data,
    uint32_t* __restrict__ out_states,
    uint32_t* __restrict__ out_offsets,
    const uint4* __restrict__ d_lookup,
    int total_symbols,
    int symbols_per_warp
) {
    // Shared Mem & Warp Setup ...
    __shared__ uint4 s_lookup[256];
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    for (int i = tid; i < 256; i += blockDim.x) s_lookup[i] = d_lookup[i];
    __syncthreads();

    int global_warp_id = (blockIdx.x * (blockDim.x / 32)) + warp_id;
    int warp_start_idx = global_warp_id * symbols_per_warp;

    // Bounds check for the whole warp
    if (warp_start_idx >= total_symbols) return;

    // Output Setup
    const int MAX_WARP_OUT = symbols_per_warp * 2;
    EncodedT* my_out_buf = out_buf + (global_warp_id * MAX_WARP_OUT);

    ANSStateT state = (1 << (kANSStateBits - PROB_BITS));

    uint32_t outOffset = 0;
    int syms_per_thread = symbols_per_warp / 32;

    // ENCODE BACKWARDS
    for (int i = syms_per_thread - 1; i >= 0; --i) {
        int sym_idx = warp_start_idx + (i * 32) + lane;

        // --- UPDATED LOGIC ---
        bool valid = (sym_idx < total_symbols);
        uint8_t sym = valid ? in_data[sym_idx] : 0;

        // Use the Partial function. 
        // If valid=false, state won't change, nothing written.
        outOffset += encodeOnePartialWarp(valid, state, sym, outOffset, my_out_buf, s_lookup);
    }

    out_states[global_warp_id * 32 + lane] = state;
    if (lane == 0) out_offsets[global_warp_id] = outOffset;
}

// --------------------------------------------------------------------------
// 4. Decoder Kernel
// --------------------------------------------------------------------------
__global__ void k_decode_warp_interleaved(
    int8_t* __restrict__ out_data,
    const EncodedT* __restrict__ in_buf,
    const uint32_t* __restrict__ in_states,
    const uint32_t* __restrict__ in_offsets,
    const uint8_t* __restrict__ d_map,
    const uint32_t* __restrict__ d_freq,
    const uint32_t* __restrict__ d_start,
    int total_symbols,
    int symbols_per_warp
) {
    // Shared Mem Setup ...
    __shared__ uint8_t s_map[PROB_SCALE];
    __shared__ uint32_t s_freq[256];
    __shared__ uint32_t s_start[256];

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    for (int i = tid; i < PROB_SCALE; i += blockDim.x) s_map[i] = d_map[i];
    for (int i = tid; i < 256; i += blockDim.x) {
        s_freq[i] = d_freq[i];
        s_start[i] = d_start[i];
    }
    __syncthreads();

    int global_warp_id = (blockIdx.x * (blockDim.x / 32)) + warp_id;
    int warp_start_idx = global_warp_id * symbols_per_warp;

    if (warp_start_idx >= total_symbols) return;

    // 1. Initialize State
    ANSStateT state = in_states[global_warp_id * 32 + lane];

    // 2. Setup Input Pointer
    const int MAX_WARP_OUT = symbols_per_warp * 2;
    const EncodedT* my_in_buf = in_buf + (global_warp_id * MAX_WARP_OUT);
    int read_idx = in_offsets[global_warp_id];

    int syms_per_thread = symbols_per_warp / 32;

    // DECODE FORWARDS
    for (int i = 0; i < syms_per_thread; ++i) {
        int out_idx = warp_start_idx + (i * 32) + lane;

        // --- UPDATED LOGIC ---
        bool valid = (out_idx < total_symbols);

        // A. Decode Symbol
        uint32_t slot = state & (PROB_SCALE - 1);
        uint8_t sym = s_map[slot];

        // Only write if valid
        if (valid) {
            out_data[out_idx] = (int8_t)sym;
        }

        // B. Update State
        // Only update state if valid. 
        // If invalid, state remains frozen (matches Encoder behavior)
        uint32_t freq = s_freq[sym];
        uint32_t start = s_start[sym];

        if (valid) {
            state = freq * (state >> PROB_BITS) + (slot - start);
        }

        // C. Renormalize (Read / Pop)
        // Invalid lanes must NOT trigger a read
        uint32_t bound = freq * (1 << (kANSStateBits - PROB_BITS));

        // Crucial: read is false if valid is false
        bool read = valid && (state < bound);

        uint32_t vote = __ballot_sync(0xffffffff, read);

        if (read) {
            uint32_t prefix = __popc(vote & getLaneMaskLt());
            uint32_t total_reads = __popc(vote);
            int my_read_offset = (read_idx - total_reads) + prefix;

            EncodedT new_byte = my_in_buf[my_read_offset];
            state = (state << kANSEncodedBits) | new_byte;
        }

        // Decrement stack pointer by total reads (even if I didn't read, someone else might have)
        read_idx -= __popc(vote);
    }
}
// --------------------------------------------------------------------------
// Host Functions
// --------------------------------------------------------------------------



int compress_stream_dietgpu_style(const std::vector<int8_t>& data, const char* name) {
    size_t n = data.size();
    printf("[%s] Input: %.2f MB\n", name, (double)n / 1024 / 1024);

    // 1. Prepare Tables (Host)
    std::vector<uint32_t> freq(256, 0);
    for (auto x : data) freq[(uint8_t)x]++;

    std::vector<uint32_t> start(257);
    normalize_freqs(freq, start);

    // Create the uint4 lookup table for the Encoder
    std::vector<uint4> h_enc_lookup(256);
    // Create Inverse maps for Decoder
    std::vector<uint8_t> h_dec_map(PROB_SCALE);

    for (int s = 0; s < 256; ++s) {
        uint32_t f = freq[s];
        uint32_t st = start[s];

        // Magic Numbers for __umulhi
        uint32_t m, shift;
        compute_magic_u32(f > 0 ? f : 1, &m, &shift);

        // Pack into uint4 (.x=pdf, .y=cdf, .z=div_m1, .w=div_shift)
        h_enc_lookup[s] = make_uint4(f, st, m, shift);

        // Inverse Map
        for (uint32_t i = 0; i < f; ++i) h_dec_map[st + i] = (uint8_t)s;
    }

    // 2. Upload to Device
    uint8_t* d_in;
    uint4* d_enc_lookup;
    uint8_t* d_dec_map;
    uint32_t* d_freq, * d_start;

    gpuErrchk(cudaMalloc(&d_in, n));
    gpuErrchk(cudaMemcpy(d_in, data.data(), n, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_enc_lookup, 256 * sizeof(uint4)));
    gpuErrchk(cudaMemcpy(d_enc_lookup, h_enc_lookup.data(), 256 * sizeof(uint4), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_dec_map, PROB_SCALE));
    gpuErrchk(cudaMemcpy(d_dec_map, h_dec_map.data(), PROB_SCALE, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_freq, 256 * 4));
    gpuErrchk(cudaMemcpy(d_freq, freq.data(), 256 * 4, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&d_start, 256 * 4));
    gpuErrchk(cudaMemcpy(d_start, start.data(), 256 * 4, cudaMemcpyHostToDevice));

    // 3. Execution Params
    int sym_per_warp = 1024; // 32 threads * 32 items
    int num_warps = (n + sym_per_warp - 1) / sym_per_warp;
    int num_blocks = (num_warps * 32 + 255) / 256;

    EncodedT* d_out_buf;
    uint32_t* d_out_states, * d_out_offsets;
    int chunk_stride = sym_per_warp * 2; // Safe upper bound

    gpuErrchk(cudaMalloc(&d_out_buf, num_warps * chunk_stride));
    gpuErrchk(cudaMalloc(&d_out_states, num_warps * 32 * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_out_offsets, num_warps * sizeof(uint32_t)));

    // 4. Encode
    k_encode_warp_interleaved << <num_blocks, 256 >> > (
        d_out_buf, d_in, d_out_states, d_out_offsets,
        d_enc_lookup, n, sym_per_warp
        );
    gpuErrchk(cudaDeviceSynchronize());

    // 5. Decode
    int8_t* d_out_decoded;
    gpuErrchk(cudaMalloc(&d_out_decoded, n));

    k_decode_warp_interleaved << <num_blocks, 256 >> > (
        d_out_decoded, d_out_buf, d_out_states, d_out_offsets,
        d_dec_map, d_freq, d_start, n, sym_per_warp
        );
    gpuErrchk(cudaDeviceSynchronize());

    // 6. Verify
    std::vector<int8_t> verify(n);
    gpuErrchk(cudaMemcpy(verify.data(), d_out_decoded, n, cudaMemcpyDeviceToHost));

    int errs = 0;
    for (size_t i = 0; i < n; ++i) {
        if (verify[i] != data[i]) {
            printf("Error @ %zu: Exp %d Got %d\n", i, data[i], verify[i]);
            errs++;
            if (errs > 5) break;
        }
    }
    if (errs == 0) printf("Verification PASSED.\n");

    // Cleanup omitted for brevity...
    return 0;
}