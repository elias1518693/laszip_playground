// rans_byte_cuda.cu
// CUDA translation of "Simple byte-aligned rANS encoder/decoder" (Fabian 'ryg' Giesen)
//
// Fixes applied:
// 1. Corrected bit-packing logic in write_raw_bits_backward to match forward-read/LSB-consume decoder.
// 2. Fixed host/device table memory copies and type mismatches.
// 3. Implemented robust 9-context model + k-context model.
// 4. Fixed Decoder Unmapping logic (Low=Negative, High=Positive).
// 5. Reduced RANS_BYTE_L to 1<<15 to prevent 32-bit state overflow.
// 6. Added explicit synchronization for symbol tables.
// 7. Added Bit-Count Header to handle partial byte flushing correctly.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <ostream>
#include <cassert>
#include <iostream>
#include <vector>
#include <tuple>
#include <string.h> // for memset
#include <algorithm> // for std::fill
#include <limits> // for numeric_limits
#include <cmath> // for log2
#include <climits> // for CHAR_BIT
#include <inttypes.h>

#ifdef _WIN32
#include <intrin.h>
#include <map>
#include <string>
#else
// __clz for GCC/Clang
static inline int __clz(uint32_t x) {
    return x == 0 ? 32 : __builtin_clz(x);
}
#endif


#ifdef assert
#define RansAssert assert
#else
#define RansAssert(x)
#endif

// *** FIX 5: Reduced L to prevent overflow in 32-bit registers ***
#define RANS_BYTE_L (1u << 15) // lower bound of normalization interval (Standard for 32-bit rANS)

#define SCALE_BITS 11           // 4096 total probability space
#define ALPHABET_SIZE 256
#define K_ALPHABET_SIZE 33      // k can be 0-32

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// --------------------------------------------------------------------------
// Constant Memory
// --------------------------------------------------------------------------
// We have 9 contexts for symbols (0-8) + 1 context for k (0-32)

// [DECODER] O(1) Lookup Tables
__device__ __constant__ uint8_t d_lookup[9][1 << SCALE_BITS];
__device__ __constant__ uint8_t d_lookup_k[1 << SCALE_BITS]; // k uses 8-bit symbols (0-32)

// [ENCODER + DECODER-ADVANCE] Frequency/Cumulative Tables
// Contexts 0-8 for symbols (uint8_t)
__device__ __constant__ uint16_t d_sym_freq[9][ALPHABET_SIZE];
__device__ __constant__ uint32_t d_sym_cum[9][ALPHABET_SIZE + 1];

// Context for k (0-32)
__device__ __constant__ uint16_t d_k_freq[K_ALPHABET_SIZE];
__device__ __constant__ uint32_t d_k_cum[K_ALPHABET_SIZE + 1];

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
__device__ static inline void RansEncPutSymbol(uint32_t* r, uint8_t** pptr, RansEncSymbol const* sym)
{
    uint32_t x = *r;
    uint32_t x_max = sym->x_max;

    // Renormalize
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

// --- RANS requires a *backward* bit writer ---
// CRITICAL FIX: The bits must be packed such that the *last* symbol encoded (Symbol 0)
// sits at the LSB of the byte at the lowest memory address, because the decoder
// reads forward and extracts LSBs.
//
// To achieve this when writing backwards:
// 1. Insert new bits at the LSB of the buffer (Shift buffer up).
// 2. Flush bits from the MSB of the buffer (Shift buffer down).
__device__ __forceinline__ void write_raw_bits_backward(
    uint8_t** pptr,     // RANS stream pointer (moves backward)
    uint64_t* bit_buf,  // Bit buffer
    int* bit_count,     // Number of bits in buffer
    uint32_t data,      // Data to write
    int num_bits)       // Number of bits
{
    // Shift existing bits UP, insert new bits at LSB
    *bit_buf = (*bit_buf << num_bits) | ((uint64_t)data & ((1ull << num_bits) - 1));
    *bit_count += num_bits;

    // Flush MSB bytes from buffer to memory
    while (*bit_count >= 8) {
        int shift = *bit_count - 8;
        // Extract top 8 bits
        uint8_t val = (uint8_t)(*bit_buf >> shift);
        *--(*pptr) = val;
        *bit_count -= 8;
        // Mask to keep clean for next iteration (prevent garbage in upper bits)
        *bit_buf &= ((1ull << *bit_count) - 1);
    }
}

// --- Decoder needs a *forward* bit reader ---
// Standard LSB reader: Reads byte, appends to MSB of buffer, consumes LSB.
__device__ __forceinline__ uint32_t read_raw_bits_forward(
    uint8_t** pptr,     // RANS stream pointer (moves forward)
    uint64_t* bit_buf,  // Bit buffer (consumed from LSB)
    int* bit_count,     // Number of bits in buffer
    int num_bits)       // Number of bits to read
{
    // Fill buffer from memory if needed
    while (*bit_count < num_bits) {
        *bit_buf |= (uint64_t)(*(*pptr)++) << (*bit_count);
        *bit_count += 8;
    }

    // Get bits from LSB
    uint32_t data = (uint32_t)(*bit_buf & ((1ull << num_bits) - 1));

    // Consume bits
    *bit_buf >>= num_bits;
    *bit_count -= num_bits;

    return data;
}

// --------------------------------------------------------------------------
// KERNELS
// --------------------------------------------------------------------------

__global__ void encode_kernel(uint8_t* outbuf, const int32_t* in_data,
    size_t buf_stride, size_t block_size, int symbols_per_thread, size_t total_symbols, uint32_t* bytes_used)
{
    // --- Indexing ---
    int tid_in_block = threadIdx.x;
    int total_threads_in_block = blockDim.x;
    size_t global_tid = (size_t)blockIdx.x * total_threads_in_block + tid_in_block;
    size_t block_start_addr = (size_t)blockIdx.x * block_size;

    // --- Per-thread state ---
    uint8_t* mybuf = outbuf + global_tid * buf_stride;
    uint8_t* ptr = mybuf + buf_stride; // RANS encodes backwards
    uint32_t s = RANS_BYTE_L;
    const int32_t* seq = in_data;

    // --- Raw bit stream state ---
    uint64_t raw_bit_buf = 0;
    int raw_bit_count = 0;

    // Process symbols backwards
    for (int i = symbols_per_thread - 1; i >= 0; --i) {
        size_t addr = block_start_addr + (size_t)i * total_threads_in_block + tid_in_block;

        if (addr >= total_symbols) {
            continue;
        }

        int32_t c = seq[addr];
        uint32_t k;
        uint32_t mapped_c;
        uint8_t high_byte; // This is the symbol we RANS encode
        int context_k;     // This is the *row* in the freq table (0-8)

        // --- 1. Find k (must match host find_k_9_context) ---
        if (c == 0 || c == 1) {
            k = 0;
        }
        else if (c == INT32_MIN) { // Handle INT32_MIN specifically
            k = 32;
        }
        else {
            uint32_t m = (c < 0) ? (-c) : c;
            uint32_t val = (c < 0) ? (m + 1) : m;
            k = (val == 1) ? 1 : (32 - __clz(val - 1));
        }

        // --- 2. Map c to positive value (must match host map_delta_to_positive) ---
        if (k == 0) mapped_c = (uint32_t)c;
        else if (k == 32) mapped_c = 0; // No value stored
        else if (c < 0) mapped_c = c + (1u << k) - 1;
        else mapped_c = c - 1;


        // --- 3. Determine high_byte (symbol) and context ---
        if (k == 0) {
            high_byte = (uint8_t)mapped_c; // 0 or 1
            context_k = 0; // Use context 0
            // No raw bits
        }
        else if (k < 8) { // 1 <= k <= 7
            high_byte = (uint8_t)mapped_c; // Full value
            context_k = k; // Use context k (1-7)
            // No raw bits
        }
        else if (k < 32) { // 8 <= k < 32
            high_byte = (uint8_t)(mapped_c >> (k - 8)); // "Corrector" (upper 8 bits)
            uint32_t lower = mapped_c & ((1u << (k - 8)) - 1); // "lower bits"
            context_k = 8; // Use context 8 for all k >= 8

            // Write 'k-8' raw bits *backward*
            write_raw_bits_backward(&ptr, &raw_bit_buf, &raw_bit_count, lower, k - 8);
        }
        else { // k == 32
            // Only 'k' is stored. No symbol, no raw bits.
            context_k = -1; // Flag to skip symbol encoding
        }


        // --- 4. Encode Symbol (if it exists) ---
        if (context_k != -1) {
            uint32_t freq = d_sym_freq[context_k][high_byte];
            uint32_t start = d_sym_cum[context_k][high_byte];

            RansEncSymbol symb;
            symb.x_max = ((RANS_BYTE_L >> SCALE_BITS) << 8) * freq;
            symb.cmpl_freq = (uint16_t)((1 << SCALE_BITS) - freq);
            symb.bias = start;

            if (freq < 2) {
                symb.rcp_freq = ~0u;
                symb.rcp_shift = 0;
                symb.bias = start + (1 << SCALE_BITS) - 1;
            }
            else {
                uint32_t shift = 0;
                while (freq > (1u << shift)) shift++;
                symb.rcp_freq = (uint32_t)(((1ull << (shift + 31)) + freq - 1) / freq);
                symb.rcp_shift = shift - 1;
            }

            RansEncPutSymbol(&s, &ptr, &symb);
        }

        // --- 5. Encode 'k' ---
        {
            uint32_t freq = d_k_freq[k];
            uint32_t start = d_k_cum[k];

            RansEncSymbol symb_k;
            symb_k.x_max = ((RANS_BYTE_L >> SCALE_BITS) << 8) * freq;
            symb_k.cmpl_freq = (uint16_t)((1 << SCALE_BITS) - freq);
            symb_k.bias = start;

            if (freq < 2) {
                symb_k.rcp_freq = ~0u;
                symb_k.rcp_shift = 0;
                symb_k.bias = start + (1 << SCALE_BITS) - 1;
            }
            else {
                uint32_t shift = 0;
                while (freq > (1u << shift)) shift++;
                symb_k.rcp_freq = (uint32_t)(((1ull << (shift + 31)) + freq - 1) / freq);
                symb_k.rcp_shift = shift - 1;
            }

            RansEncPutSymbol(&s, &ptr, &symb_k);
        }
    }

    // --- 6. Flush remaining bits from raw buffer ---
    if (raw_bit_count > 0) {
        // With Insert-LSB, the bits we want are at the bottom of raw_bit_buf.
        *--ptr = (uint8_t)raw_bit_buf;
    }

    // *** FIX 7: Write Header for Partial Byte Count ***
    // This tells the decoder how many bits in the *next* byte (the one we just wrote) are valid.
    // If raw_bit_count == 0, we write 0.
    *--ptr = (uint8_t)raw_bit_count;

    // --- 7. Flush RANS State ---
    {
        uint32_t x = s;
        ptr -= 4;
        ptr[0] = (uint8_t)(x >> 0);
        ptr[1] = (uint8_t)(x >> 8);
        ptr[2] = (uint8_t)(x >> 16);
        ptr[3] = (uint8_t)(x >> 24);
    }

    // --- 8. Calculate total length ---
    size_t length = (size_t)((mybuf + buf_stride) - ptr);
    bytes_used[global_tid] = (uint32_t)length;
}


__global__ void decode_kernel(uint8_t* inbuf,
    size_t block_size, int symbols_per_thread, int32_t* out_data, size_t total_symbols, uint32_t* offsets)
{
    // --- Indexing ---
    int tid_in_block = threadIdx.x;
    int total_threads_in_block = blockDim.x;
    size_t global_tid = (size_t)blockIdx.x * total_threads_in_block + tid_in_block;
    size_t block_start_addr = (size_t)blockIdx.x * block_size;

    // --- Per-thread state ---
    uint8_t* ptr = inbuf + offsets[global_tid];
    uint32_t s; // RANS state

    // --- Raw bit stream state ---
    uint64_t raw_bit_buf = 0;
    int raw_bit_count = 0;

    // --- Init RANS State ---
    {
        s = (uint32_t)ptr[0] << 0;
        s |= (uint32_t)ptr[1] << 8;
        s |= (uint32_t)ptr[2] << 16;
        s |= (uint32_t)ptr[3] << 24;
        ptr += 4;
    }

    // *** FIX 7: Read Header for Partial Byte Count ***
    int valid_bits = (int)*ptr++;
    if (valid_bits > 0) {
        raw_bit_buf = (uint64_t)*ptr++;
        // Mask out the garbage upper bits
        raw_bit_buf &= ((1ull << valid_bits) - 1);
        raw_bit_count = valid_bits;
    }
    else {
        // No partial byte, stream is aligned or empty
        raw_bit_buf = 0;
        raw_bit_count = 0;
    }

    for (int i = 0; i < symbols_per_thread; ++i) {

        // --- 1. Decode 'k' ---
        uint32_t cum_val = s & ((1u << SCALE_BITS) - 1);
        uint8_t k = d_lookup_k[cum_val]; // k is 0-32

        // --- Get freq/start for 'k' ---
        uint32_t k_start = d_k_cum[k];
        uint32_t k_freq = d_k_freq[k];

        // --- Advance RANS state for 'k' ---
        {
            uint32_t mask = (1u << SCALE_BITS) - 1;
            s = k_freq * (s >> SCALE_BITS) + (s & mask) - k_start;
            // Renormalize
            if (s < RANS_BYTE_L) {
                do s = (s << 8) | *ptr++; while (s < RANS_BYTE_L);
            }
        }

        // --- 2. Determine context and decode symbol ---
        uint8_t high_byte;
        int context_k;

        if (k == 0) context_k = 0;
        else if (k < 8) context_k = k; // 1-7
        else context_k = 8; // 8-31
        // k=32 has no symbol

        if (k != 32) {
            cum_val = s & ((1u << SCALE_BITS) - 1);
            high_byte = d_lookup[context_k][cum_val];

            uint32_t sym_start = d_sym_cum[context_k][high_byte];
            uint32_t sym_freq = d_sym_freq[context_k][high_byte];

            // --- Advance RANS state for symbol ---
            {
                uint32_t mask = (1u << SCALE_BITS) - 1;
                s = sym_freq * (s >> SCALE_BITS) + (s & mask) - sym_start;
                // Renormalize
                if (s < RANS_BYTE_L) {
                    do s = (s << 8) | *ptr++; while (s < RANS_BYTE_L);
                }
            }
        }

        // --- 3. Read raw bits and reconstruct ---
        uint32_t mapped_c;
        int32_t c;

        if (k == 0) {
            mapped_c = high_byte; // 0 or 1
            c = (int32_t)mapped_c;
        }
        else if (k < 8) { // 1-7
            mapped_c = high_byte;
            // Unmap Logic Corrected:
            if (mapped_c < (1u << (k - 1))) {
                // Negative Case
                c = (int32_t)mapped_c - (int32_t)(1u << k) + 1;
            }
            else {
                // Positive Case
                c = mapped_c + 1;
            }
        }
        else if (k < 32) { // 8-31
            uint32_t lower = read_raw_bits_forward(&ptr, &raw_bit_buf, &raw_bit_count, k - 8);
            mapped_c = ((uint32_t)high_byte << (k - 8)) | lower;

            if (mapped_c < (1u << (k - 1))) {
                c = (int32_t)mapped_c - (int32_t)(1u << k) + 1;
            }
            else {
                c = mapped_c + 1;
            }
        }
        else { // k == 32
            c = INT32_MIN;
        }

        // --- 4. Write output ---
        size_t addr = block_start_addr + (size_t)i * total_threads_in_block + tid_in_block;
        if (addr >= total_symbols) {
            continue;
        }
        out_data[addr] = c;
    }
}


// --------------------------------------------------------------------------
// Host-side code
// --------------------------------------------------------------------------

// Standard SymbolStats for 8-bit symbols
template <int BITS>
struct SymbolStats {
    static_assert(BITS >= 1 && BITS <= 8, "BITS must be between 1 and 8");
    static constexpr int SYMBOL_COUNT = 1 << BITS;

    uint32_t freqs[SYMBOL_COUNT];
    uint32_t cum_freqs[SYMBOL_COUNT + 1];

    SymbolStats() {
        std::fill(freqs, freqs + SYMBOL_COUNT, 0u);
        std::fill(cum_freqs, cum_freqs + SYMBOL_COUNT + 1, 0u);
    }

    // This count_freqs is for 8-bit symbols (uint8_t)
    void count_freqs(const uint8_t* in, size_t nbytes)
    {
        if (nbytes == 0) return;
        std::fill(freqs, freqs + SYMBOL_COUNT, 0u);
        for (size_t i = 0; i < nbytes; i++) {
            freqs[in[i]]++;
        }
    }

    void calc_cum_freqs()
    {
        cum_freqs[0] = 0;
        for (int i = 0; i < SYMBOL_COUNT; i++)
            cum_freqs[i + 1] = cum_freqs[i] + freqs[i];
    }

    void normalize_freqs(uint32_t target_total)
    {
        assert(target_total >= SYMBOL_COUNT);

        calc_cum_freqs();
        uint32_t cur_total = cum_freqs[SYMBOL_COUNT];
        if (cur_total == 0)
        {
            // No symbols, create flat distribution
            for (int i = 0; i < SYMBOL_COUNT; i++) freqs[i] = 1;
            freqs[0] = target_total - (SYMBOL_COUNT - 1);
            calc_cum_freqs();
            cur_total = cum_freqs[SYMBOL_COUNT];
            assert(cur_total == target_total);
            return;
        }

        // Resample distribution
        for (int i = 1; i <= SYMBOL_COUNT; i++)
            cum_freqs[i] = (uint64_t)target_total * cum_freqs[i] / cur_total;

        // Fix zeroed nonzero frequencies
        for (int i = 0; i < SYMBOL_COUNT; i++) {
            if (freqs[i] && cum_freqs[i + 1] == cum_freqs[i]) {
                uint32_t best_freq = ~0u;
                int best_steal = -1;
                for (int j = 0; j < SYMBOL_COUNT; j++) {
                    uint32_t freq = cum_freqs[j + 1] - cum_freqs[j];
                    if (freq > 1 && freq < best_freq) {
                        best_freq = freq;
                        best_steal = j;
                    }
                }

                if (best_steal == -1) {
                    best_steal = i;
                    if (cum_freqs[i + 1] - cum_freqs[i] <= 1) {
                        for (int j = 0; j < SYMBOL_COUNT; ++j) {
                            if (cum_freqs[j + 1] - cum_freqs[j] > 1) {
                                best_steal = j;
                                break;
                            }
                        }
                    }
                }

                if (best_steal < i) {
                    for (int j = best_steal + 1; j <= i; j++)
                        cum_freqs[j]--;
                }
                else {
                    for (int j = i + 1; j <= best_steal; j++)
                        cum_freqs[j]++;
                }
            }
        }

        assert(cum_freqs[0] == 0 && cum_freqs[SYMBOL_COUNT] == target_total);

        for (int i = 0; i < SYMBOL_COUNT; i++) {
            if (freqs[i] == 0)
                assert(cum_freqs[i + 1] == cum_freqs[i]);
            else
                assert(cum_freqs[i + 1] > cum_freqs[i]);

            freqs[i] = cum_freqs[i + 1] - cum_freqs[i];
        }
    }
};

// Custom Stats for 'k' (33 symbols, 0-32)
struct SymbolStatsK {
    static constexpr int SYMBOL_COUNT = K_ALPHABET_SIZE; // 33

    uint32_t freqs[SYMBOL_COUNT];
    uint32_t cum_freqs[SYMBOL_COUNT + 1];

    SymbolStatsK() {
        std::fill(freqs, freqs + SYMBOL_COUNT, 0u);
        std::fill(cum_freqs, cum_freqs + SYMBOL_COUNT + 1, 0u);
    }

    // This count_freqs is for 'k' values (0-32)
    void count_freqs(const std::vector<uint8_t>& in)
    {
        if (in.empty()) return;
        std::fill(freqs, freqs + SYMBOL_COUNT, 0u);
        for (uint8_t k : in) {
            assert(k < SYMBOL_COUNT);
            freqs[k]++;
        }
    }

    // Identical normalize_freqs logic
    void calc_cum_freqs()
    {
        cum_freqs[0] = 0;
        for (int i = 0; i < SYMBOL_COUNT; i++)
            cum_freqs[i + 1] = cum_freqs[i] + freqs[i];
    }

    void normalize_freqs(uint32_t target_total)
    {
        assert(target_total >= SYMBOL_COUNT);

        calc_cum_freqs();
        uint32_t cur_total = cum_freqs[SYMBOL_COUNT];
        if (cur_total == 0)
        {
            for (int i = 0; i < SYMBOL_COUNT; i++) freqs[i] = 1;
            freqs[0] = target_total - (SYMBOL_COUNT - 1);
            calc_cum_freqs();
            cur_total = cum_freqs[SYMBOL_COUNT];
            assert(cur_total == target_total);
            return;
        }

        for (int i = 1; i <= SYMBOL_COUNT; i++)
            cum_freqs[i] = (uint64_t)target_total * cum_freqs[i] / cur_total;

        for (int i = 0; i < SYMBOL_COUNT; i++) {
            if (freqs[i] && cum_freqs[i + 1] == cum_freqs[i]) {
                uint32_t best_freq = ~0u;
                int best_steal = -1;
                for (int j = 0; j < SYMBOL_COUNT; j++) {
                    uint32_t freq = cum_freqs[j + 1] - cum_freqs[j];
                    if (freq > 1 && freq < best_freq) {
                        best_freq = freq;
                        best_steal = j;
                    }
                }
                if (best_steal == -1) {
                    for (int j = 0; j < SYMBOL_COUNT; ++j) {
                        if (cum_freqs[j + 1] - cum_freqs[j] > 1) {
                            best_steal = j;
                            break;
                        }
                    }
                }
                assert(best_steal != -1);

                if (best_steal < i) {
                    for (int j = best_steal + 1; j <= i; j++)
                        cum_freqs[j]--;
                }
                else {
                    for (int j = i + 1; j <= best_steal; j++)
                        cum_freqs[j]++;
                }
            }
        }

        assert(cum_freqs[0] == 0 && cum_freqs[SYMBOL_COUNT] == target_total);

        for (int i = 0; i < SYMBOL_COUNT; i++) {
            if (freqs[i] == 0)
                assert(cum_freqs[i + 1] == cum_freqs[i]);
            else
                assert(cum_freqs[i + 1] > cum_freqs[i]);
            freqs[i] = cum_freqs[i + 1] - cum_freqs[i];
        }
    }
};


struct k_code {
    std::vector<std::vector<uint8_t>> upper_byte; // [9]
    std::vector<uint8_t> k;
    std::vector <uint32_t> raw_bits;
};

// Cross-platform clz32
int clz32(uint32_t x) {
    if (x == 0) return 32;

#ifdef _WIN32
    unsigned long idx;
    _BitScanReverse(&idx, x);
    return 31 - idx;
#else
    // __builtin_clz is for GCC/Clang
    return __builtin_clz(x);
#endif
}

// Host-side find_k (must match device)
uint8_t find_k(int32_t c) {
    if (c == 0 || c == 1) return 0;
    if (c == std::numeric_limits<int32_t>::min()) return 32;

    uint32_t m = (c < 0) ? (-c) : c;
    uint32_t val = (c < 0) ? (m + 1) : m;
    return (val == 1) ? 1 : (32 - clz32(val - 1));
}

// Host-side map_delta (must match device)
uint32_t map_delta_to_positive(int32_t c, uint8_t k) {
    if (k == 0) return (uint32_t)c;
    if (k == 32) return 0;
    if (c < 0) return c + (1ULL << k) - 1;
    else return c - 1;
}

/**
 * Converts signed deltas into the k_code struct for the 9-Context Model
 */
k_code int32_to_k_codes_9_context(const std::vector<int32_t>& input) {
    k_code code;
    code.upper_byte.resize(9); // Contexts 0-8

    for (int32_t val : input) {
        uint8_t k = find_k(val);
        code.k.push_back(k);

        uint32_t mapped_val = map_delta_to_positive(val, k);

        if (k == 0) {
            // k=0: Store 0 or 1 in context 0
            code.upper_byte[0].push_back((uint8_t)mapped_val);
        }
        else if (k < 8) { // 1 <= k <= 7
            // Store full value in context k
            code.upper_byte[k].push_back((uint8_t)mapped_val);
        }
        else if (k < 32) { // 8 <= k < 32
            // Store "Corrector" (upper 8 bits) in context 8
            uint8_t upper = (uint8_t)(mapped_val >> (k - 8));
            code.upper_byte[8].push_back(upper);

            // "lower bits" (remaining k-8 bits)
            uint32_t lower = mapped_val & ((1u << (k - 8)) - 1);
            code.raw_bits.push_back(lower);
        }
        else {
            // k == 32: Only k is stored. No symbol, no raw bits.
        }
    }
    return code;
}


// Host-side build lookup table (for 8-bit symbols)
void build_host_lookup(uint8_t* h_lookup, uint32_t* cum_freqs, int num_symbols, size_t lookup_size) {
    for (int s = 0; s < num_symbols; ++s) {
        for (uint32_t i = cum_freqs[s]; i < cum_freqs[s + 1]; ++i) {
            if (i < lookup_size) {
                h_lookup[i] = (uint8_t)s;
            }
        }
    }
}

double calculate_shannon_entropy(const std::vector<int32_t>& data) {
    std::map<int32_t, uint64_t> counts;
    for (int32_t val : data) {
        counts[val]++;
    }

    double entropy = 0.0;
    double total = (double)data.size();
    for (auto const& [val, count] : counts) {
        double p = (double)count / total;
        entropy -= p * std::log2(p);
    }

    // Returns total entropy in MB
    return (entropy * total) / (8.0 * 1024.0 * 1024.0);
}

// Helper to print a simple text histogram (C++14 compatible)
template <typename IntType>
void print_histogram(const std::vector<IntType>& data, const char* title, int num_bins = 20)
{
    if (data.empty()) {
        printf("Histogram: '%s' (No data)\n", title);
        return;
    }

    // 1. Find min and max
    IntType min_val = data[0];
    IntType max_val = data[0];
    for (auto val : data) {
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    printf("\n--- Histogram: %s ---\n", title);
    printf("Total points: %zu\n", data.size());
    printf("Min: %lld, Max: %lld\n", (long long)min_val, (long long)max_val);

    if (min_val == max_val) {
        printf("[%lld] count: %zu\n", (long long)min_val, data.size());
        return;
    }

    // 2. Determine bin size
    double range = (double)max_val - (double)min_val;
    double bin_size = range / num_bins;
    if (bin_size < 1.0) {
        bin_size = 1.0;
        num_bins = (int)range + 1;
        if (num_bins > 50) { // Cap bins if range is small but large enough to spam
            num_bins = 50;
            bin_size = range / num_bins;
        }
    }

    // 3. Populate bins
    std::vector<long long> bin_counts(num_bins, 0);
    long long max_count = 0;

    for (auto val : data) {
        int bin_index = (int)(((double)val - (double)min_val) / bin_size);
        if (bin_index >= num_bins) bin_index = num_bins - 1;
        if (bin_index < 0) bin_index = 0;
        bin_counts[bin_index]++;
        if (bin_counts[bin_index] > max_count)
            max_count = bin_counts[bin_index];
    }

    // 4. Print histogram
    const int max_bar_width = 40;
    printf("%-25s | %-10s | %s\n", "Bin Range", "Count", "Bar");
    for (int k = 0; k < 25 + 10 + 3 + max_bar_width; k++) printf("-");
    printf("\n");

    for (int i = 0; i < num_bins; ++i) {
        long long bin_start = (long long)(min_val + i * bin_size);
        long long bin_end = (long long)(min_val + (i + 1) * bin_size);

        // Format range string manually
        char range_buf[32];
        if (i == num_bins - 1)
            snprintf(range_buf, sizeof(range_buf), "[%lld, %lld]", bin_start, (long long)max_val);
        else
            snprintf(range_buf, sizeof(range_buf), "[%lld, %lld)", bin_start, bin_end);

        long long count = bin_counts[i];
        int bar_width = (max_count > 0)
            ? (int)(((double)count / max_count) * max_bar_width)
            : 0;

        std::string bar(bar_width, '#');
        printf("%-25s | %-10lld | %s\n", range_buf, count, bar.c_str());
    }
    printf("---------------------------------\n");
}

int test(std::vector<int32_t>& data, size_t length)
{
    if (length == 0) return 0;

    // --- Configuration ---
    const int nthreads = 256;
    const size_t LOG_BLOCK_SIZE = 12;
    const size_t BLOCK_SIZE = 1 << LOG_BLOCK_SIZE;
    const int symbols_per_thread = BLOCK_SIZE / nthreads;
    const size_t buf_stride = symbols_per_thread * 32;
    // --- End Configuration ---

    // Pad total_symbols to be a multiple of BLOCK_SIZE for full blocks
    size_t total_symbols = (length + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    int nblocks = (int)(total_symbols / BLOCK_SIZE);

    int total_threads = nblocks * nthreads;
    std::vector<uint32_t> bytes_used(total_threads);

    const size_t total_buf = (size_t)total_threads * buf_stride;

    printf("Running test with %d blocks * %d threads = %d total threads.\n", nblocks, nthreads, total_threads);
    printf("Data Block Size: %zu symbols. Symbols per thread: %d.\n", BLOCK_SIZE, symbols_per_thread);
    printf("Total Symbols (padded): %zu. Original length: %.2f MB.\n", total_symbols, (double)(length * sizeof(int32_t)) / (1024.0 * 1024.0));


    double shannon_limit_mb = calculate_shannon_entropy(data);
    printf("Theoretical Shannon Limit: %.2f MB (Absolute limit for lossless compression).\n", shannon_limit_mb);

    // --- 1. Process data on host ---
    k_code k_codes = int32_to_k_codes_9_context(data);

	print_histogram<uint8_t>(k_codes.k, "'k' Values Histogram", 33);
    for (int i = 0; i < 9; ++i) {
        if (!k_codes.upper_byte[i].empty()) {
            std::string title = "Context " + std::to_string(i) + " Symbols";
            // Use 256 bins if we have enough data to see the full byte range details
            int bins = (k_codes.upper_byte[i].size() > 500) ? 64 : 20;
            print_histogram(k_codes.upper_byte[i], title.c_str(), bins);
        }
    }
    // Calculate Ideal Bit-Packed Size (Theoretical)
    // This sums up the minimal bits needed for each integer (magnitude 'k')
    // For k=0 (values 0 and 1), we count 1 bit because 0 bits cannot store information.
    size_t total_packed_bits = 0;
    for (uint8_t k : k_codes.k) {
        total_packed_bits += (k == 0) ? 1 : k;
    }
    printf("Ideal Bit-Packed length: %.2f MB (Theoretical limit without entropy coding overhead).\n", (double)total_packed_bits / (8.0 * 1024.0 * 1024.0));


    // --- 2. Build Freq/Cum/Lookup tables for Symbols (Contexts 0-8) ---
    const size_t kLookupSize = (1 << SCALE_BITS);

    // Create all 9 stats objects (all use 8-bit symbols)
    auto stats_tuple = std::make_tuple(
        SymbolStats<8>{}, SymbolStats<8>{}, SymbolStats<8>{},
        SymbolStats<8>{}, SymbolStats<8>{}, SymbolStats<8>{},
        SymbolStats<8>{}, SymbolStats<8>{}, SymbolStats<8>{}
    );

    // Host-side tables
    uint8_t* h_lookup_table_2d = (uint8_t*)malloc(sizeof(uint8_t) * 9 * kLookupSize);
    uint16_t* h_sym_freq_2d = (uint16_t*)malloc(sizeof(uint16_t) * 9 * ALPHABET_SIZE);
    uint32_t* h_sym_cum_2d = (uint32_t*)malloc(sizeof(uint32_t) * 9 * (ALPHABET_SIZE + 1));

    if (!h_lookup_table_2d || !h_sym_freq_2d || !h_sym_cum_2d) {
        printf("Failed to alloc host tables\n"); return 1;
    }
    memset(h_lookup_table_2d, 0, sizeof(uint8_t) * 9 * kLookupSize);

    // Process all 9 contexts (0-8)
    std::apply([&](auto&... stats) {
        size_t idx = 0; // start from upper_byte[0]
        ((
            // A: Count freqs
            stats.count_freqs(k_codes.upper_byte[idx].data(), k_codes.upper_byte[idx].size()),
            // B: Normalize
            stats.normalize_freqs(1 << SCALE_BITS),
            // C: Build host lookup
            build_host_lookup(h_lookup_table_2d + (idx * kLookupSize), stats.cum_freqs, 256, kLookupSize),

            // D: Copy freqs/cums for encoder (with correct casting)
            [&]() {
                // Manually copy and cast from uint32_t (stats) to uint16_t (host buffer)
                uint16_t* freq_ptr = h_sym_freq_2d + (idx * ALPHABET_SIZE);
                for (int i = 0; i < ALPHABET_SIZE; ++i) {
                    freq_ptr[i] = (uint16_t)stats.freqs[i];
                }
            }(),
                memcpy(h_sym_cum_2d + (idx * (ALPHABET_SIZE + 1)), stats.cum_freqs, sizeof(uint32_t) * (ALPHABET_SIZE + 1)),

                // E: Increment index
                ++idx
                ), ...);
        }, stats_tuple);

    // --- 3. Build Freq/Cum/Lookup for 'k' (Context 33) ---
    SymbolStatsK statsk;
    statsk.count_freqs(k_codes.k);
    statsk.normalize_freqs(1 << SCALE_BITS);

    uint8_t* h_lookup_k = (uint8_t*)malloc(sizeof(uint8_t) * kLookupSize);
    if (!h_lookup_k) {
        printf("Failed to alloc h_lookup_k\n"); return 1;
    }
    build_host_lookup(h_lookup_k, statsk.cum_freqs, K_ALPHABET_SIZE, kLookupSize);

    // Create host buffer for k_freqs
    uint16_t* h_k_freq = (uint16_t*)malloc(sizeof(uint16_t) * K_ALPHABET_SIZE);
    if (!h_k_freq) { printf("Failed to alloc h_k_freq\n"); return 1; }
    for (int i = 0; i < K_ALPHABET_SIZE; ++i) {
        h_k_freq[i] = (uint16_t)statsk.freqs[i];
    }

    // --- 4. Upload ALL tables to Constant Memory ---
    // Lookups (Decoder)
    gpuErrchk(cudaMemcpyToSymbol(d_lookup_k, h_lookup_k, sizeof(uint8_t) * kLookupSize));
    gpuErrchk(cudaMemcpyToSymbol(d_lookup, h_lookup_table_2d, sizeof(uint8_t) * 9 * kLookupSize));

    // Freqs/Cums (Encoder + Decoder)
    gpuErrchk(cudaMemcpyToSymbol(d_sym_freq, h_sym_freq_2d, sizeof(uint16_t) * 9 * ALPHABET_SIZE));
    gpuErrchk(cudaMemcpyToSymbol(d_sym_cum, h_sym_cum_2d, sizeof(uint32_t) * 9 * (ALPHABET_SIZE + 1)));

    gpuErrchk(cudaMemcpyToSymbol(d_k_freq, h_k_freq, sizeof(uint16_t) * K_ALPHABET_SIZE));
    gpuErrchk(cudaMemcpyToSymbol(d_k_cum, statsk.cum_freqs, sizeof(uint32_t) * (K_ALPHABET_SIZE + 1)));

    // *** FIX 6: Explicit sync to ensure tables are visible ***
    gpuErrchk(cudaDeviceSynchronize());

    // Free host tables
    free(h_lookup_table_2d);
    free(h_sym_freq_2d);
    free(h_sym_cum_2d);
    free(h_lookup_k);
    free(h_k_freq);

    // --- 5. Allocate GPU Buffers ---
    uint8_t* d_buf;
    int32_t* d_out;
    uint32_t* d_bytes_used;
    int32_t* d_in;

    gpuErrchk(cudaMalloc(&d_buf, total_buf));
    gpuErrchk(cudaMemset(d_buf, 0, total_buf));

    gpuErrchk(cudaMalloc(&d_bytes_used, bytes_used.size() * sizeof(uint32_t)));
    gpuErrchk(cudaMemset(d_bytes_used, 0, bytes_used.size() * sizeof(uint32_t)));

    gpuErrchk(cudaMalloc(&d_in, total_symbols * sizeof(int32_t)));
    gpuErrchk(cudaMemset(d_in, 0, total_symbols * sizeof(int32_t)));
    gpuErrchk(cudaMemcpy(d_in, data.data(), length * sizeof(int32_t), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_out, total_symbols * sizeof(int32_t))); // Output is int32_t
    gpuErrchk(cudaMemset(d_out, 0, total_symbols * sizeof(int32_t)));

    // --- 6. Encode ---
    cudaEvent_t start_encode, stop_encode;
    cudaEventCreate(&start_encode);
    cudaEventCreate(&stop_encode);
    cudaEventRecord(start_encode);

    encode_kernel << < nblocks, nthreads >> > (
        d_buf, d_in, buf_stride, BLOCK_SIZE, symbols_per_thread, total_symbols, d_bytes_used);

    cudaEventRecord(stop_encode);
    cudaEventSynchronize(stop_encode);

    gpuErrchk(cudaMemcpy(bytes_used.data(), d_bytes_used, bytes_used.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    size_t total_compressed_size = 0;
    for (uint32_t size : bytes_used) {
        total_compressed_size += size;
    }

    // --- 7. Consolidate Encoded Data ---
    std::vector<uint8_t> h_final_output(total_compressed_size);
    uint8_t* host_ptr = h_final_output.data();
    size_t num_chunks = bytes_used.size();

    for (size_t i = 0; i < num_chunks; ++i) {
        if (bytes_used[i] > 0) {
            uint8_t* gpu_source_ptr = d_buf + (i * buf_stride) + buf_stride - bytes_used[i];
            gpuErrchk(cudaMemcpy(host_ptr, gpu_source_ptr, bytes_used[i], cudaMemcpyDeviceToHost));
            host_ptr += bytes_used[i];
        }
    }

    std::vector<uint32_t> h_offsets(num_chunks);
    uint32_t current_offset = 0;
    for (size_t i = 0; i < num_chunks; ++i) {
        h_offsets[i] = current_offset;
        current_offset += bytes_used[i];
    }

    printf("Encoded length: %.2f MB.\n", (double)total_compressed_size / (1024.0 * 1024.0));
    float encode_ms = 0.0f;
    cudaEventElapsedTime(&encode_ms, start_encode, stop_encode);

    uint8_t* d_coherent_buf;
    gpuErrchk(cudaMalloc(&d_coherent_buf, total_compressed_size));
    gpuErrchk(cudaMemcpy(d_coherent_buf, h_final_output.data(), total_compressed_size, cudaMemcpyHostToDevice));

    uint32_t* d_offsets;
    gpuErrchk(cudaMalloc(&d_offsets, h_offsets.size() * sizeof(uint32_t)));
    gpuErrchk(cudaMemcpy(d_offsets, h_offsets.data(), h_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // --- 8. Decode ---
    cudaEvent_t start_decode, stop_decode;
    gpuErrchk(cudaEventCreate(&start_decode));
    gpuErrchk(cudaEventCreate(&stop_decode));
    gpuErrchk(cudaEventRecord(start_decode));

    decode_kernel << < nblocks, nthreads >> > (
        d_coherent_buf, BLOCK_SIZE, symbols_per_thread, d_out, total_symbols, d_offsets);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    cudaEventRecord(stop_decode);
    cudaEventSynchronize(stop_decode);

    float decode_ms = 0.0f;
    cudaEventElapsedTime(&decode_ms, start_decode, stop_decode);

    // --- 9. Verify ---
    std::vector<int32_t> h_out(total_symbols);
    gpuErrchk(cudaMemcpy(h_out.data(), d_out, total_symbols * sizeof(int32_t), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (size_t i = 0; i < length; ++i) { // Verify only original length
        if (h_out[i] != data[i]) {
            ok = false;
            printf("Mismatch at index %zu: expected %d, got %d\n", i, data[i], h_out[i]);
            break;
        }
    }

    // --- Compute throughput (MB/s)
    double data_mb = static_cast<double>(length * sizeof(int32_t)) / (1024.0 * 1024.0); // Use original length
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
    cudaFree(d_coherent_buf);
    cudaFree(d_offsets);
    cudaFree(d_bytes_used);

    return ok ? 0 : 1;
}