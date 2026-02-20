#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#define RENORM_THRESHOLD (1u << 24)

struct alignas(4) PointFormat2 {
    int32_t  X;
    int32_t  Y;
    int32_t  Z;
    uint16_t Intensity;
    uint8_t  ReturnMask;
    uint8_t  Classification;
    int8_t   ScanAngleRank;
    uint8_t  UserData;
    uint16_t PointSourceID;
    uint16_t Red;
    uint16_t Green;
    uint16_t Blue;
};

// ---------------------------------------------------------------
// Range Decoder (LASzip style)
// ---------------------------------------------------------------

#define AC__MinLength 0x01000000U



// ---------------------------------------------------------------
// 1. FastAC Arithmetic Decoder (Compliant with LAZ Spec 1.4)
// ---------------------------------------------------------------

struct ArithmeticDecoder {
    uint32_t length;
    uint32_t value;
    const uint8_t* buffer;
    uint32_t pos;

    __device__ void init(const uint8_t* data, uint32_t offset) {
        buffer = data;
        pos = offset;
        length = 0xFFFFFFFFU;
        value = ((uint32_t)buffer[pos] << 24) | ((uint32_t)buffer[pos + 1] << 16) |
            ((uint32_t)buffer[pos + 2] << 8) | ((uint32_t)buffer[pos + 3]);
        pos += 4;
    }

    __device__ inline void renorm() {
        while (length < AC__MinLength) {
            value = (value << 8) | buffer[pos++];
            length <<= 8;
        }
    }

    __device__ inline uint32_t readBits(int bit_count) {
        // LAZ allows max 19 bits per block to prevent overflow.
        // If more than 19 bits are needed, read the lower 16 bits first.
        if (bit_count > 19) {
            uint32_t lower = readBits(16);
            uint32_t upper = readBits(bit_count - 16);
            return lower | (upper << 16);
        }

        // Read block of bits using division as exactly defined in the LAZ Raw Encoder
        uint32_t ltmp = length >> bit_count;
        uint32_t raw = value / ltmp;

        length = ltmp;
        value -= length * raw;

        renorm();

        return raw;
    }
};

// ---------------------------------------------------------------
// 2. Symbol Model (FastAC)
// ---------------------------------------------------------------
struct SymbolModel {
    static const uint32_t SCALE = 1u << 15; // 32768
    uint32_t symbol_count[256];
    uint16_t cum[257];        // cumulative, size num_symbols + 1
    uint32_t num_symbols;

    uint32_t update_cycle;
    uint32_t symbols_until_update;

    __device__ void init(uint32_t symbols) {
        num_symbols = symbols;
        for (uint32_t i = 0; i < num_symbols; ++i) symbol_count[i] = 1;
        update_cycle = (num_symbols + 6) / 2;
        symbols_until_update = update_cycle;
        build_cum();
    }

    __device__ void rescale_counts_if_needed() {
        // Keep counts in a reasonable range
        uint32_t total = 0;
        for (uint32_t i = 0; i < num_symbols; ++i) total += symbol_count[i];
        if (total > 32768) {
            total = 0;
            for (uint32_t i = 0; i < num_symbols; ++i) {
                symbol_count[i] = (symbol_count[i] >> 1) + 1;
                total += symbol_count[i];
            }
        }
    }

    __device__ void build_cum() {
        // Build cumulative from counts and scale to SCALE
        uint32_t total = 0;
        for (uint32_t i = 0; i < num_symbols; ++i) total += symbol_count[i];

        // Guard against division by zero (should not happen with counts>=1)
        if (total == 0) total = 1;

        cum[0] = 0;
        uint32_t run = 0;
        for (uint32_t i = 0; i < num_symbols; ++i) {
            uint32_t next = run + symbol_count[i];
            // Scale cumulative to [0, SCALE]
            uint32_t scaled = (next * SCALE) / total;
            cum[i + 1] = (uint16_t)scaled;
            run = next;
        }

        // Enforce strictly increasing CDF and terminal == SCALE
        cum[num_symbols] = (uint16_t)SCALE;
        for (uint32_t i = 1; i <= num_symbols; ++i) {
            if (cum[i] <= cum[i - 1]) cum[i] = cum[i - 1] + 1;
        }
        // If we overshot, clamp terminal and fix backwards (rare)
        if (cum[num_symbols] > SCALE) cum[num_symbols] = (uint16_t)SCALE;
        for (int i = (int)num_symbols - 1; i >= 0; --i) {
            if (cum[i] >= cum[i + 1]) cum[i] = cum[i + 1] - 1;
        }
        cum[0] = 0;
    }

    __device__ void update_distribution() {
        rescale_counts_if_needed();
        build_cum();

        update_cycle += (update_cycle >> 2);
        if (update_cycle > 8 * (num_symbols + 6)) update_cycle = 8 * (num_symbols + 6);
        symbols_until_update = update_cycle;
    }

    __device__ uint32_t decode(ArithmeticDecoder& dec) {
        // Map value to [0, SCALE)
        uint32_t ltmp = dec.length >> 15;              // length / SCALE
        uint32_t scaled = dec.value / ltmp;            // in [0, SCALE)

        // Find s s.t. cum[s] <= scaled < cum[s+1]
        // Linear scan is fine for 128/256 symbols; replace with binary search if needed.
        uint32_t s = 0;
        // Optional: binary search for speed
        uint32_t lo = 0, hi = num_symbols;
        while (lo + 1 < hi) {
            uint32_t mid = (lo + hi) >> 1;
            if (cum[mid] <= scaled) lo = mid;
            else hi = mid;
        }
        s = lo;

        // Interval bounds
        uint32_t lower_c = cum[s];
        uint32_t upper_c = cum[s + 1];
        uint32_t lower = lower_c * ltmp;
        uint32_t upper = upper_c * ltmp;

        // Renormalize decoder state
        dec.value -= lower;
        dec.length = upper - lower;
        dec.renorm();

        // Adapt model
        symbol_count[s]++;
        if (--symbols_until_update == 0) update_distribution();

        return s;
    }
};



struct BitModel {
    uint32_t bit_0_count;
    uint32_t bit_count;
    uint32_t bit_0_prob;
    uint32_t update_cycle;
    uint32_t bits_until_update;

    __device__ void init() {
        bit_0_count = 1;
        bit_count = 2;
        bit_0_prob = 4096;
        update_cycle = 4;
        bits_until_update = 4;
    }

    __device__ void update() {
        bit_count += update_cycle;
        if (bit_count > 8192) {
            bit_count = (bit_count + 1) >> 1;
            bit_0_count = (bit_0_count + 1) >> 1;
            if (bit_0_count == bit_count) bit_count++;
        }
        // Bit probability scaled to 13 bits (8192)
        bit_0_prob = (bit_0_count * 8192) / bit_count;

        update_cycle += (update_cycle >> 2);
        if (update_cycle > 64) update_cycle = 64;
        bits_until_update = update_cycle;
    }

    __device__ uint32_t decode(ArithmeticDecoder& dec) {
        uint32_t ltmp = dec.length >> 13;
        uint32_t lower = bit_0_prob * ltmp;
        uint32_t bit;

        if (dec.value >= lower) {
            bit = 1;
            dec.value -= lower;
            dec.length -= lower;
        }
        else {
            bit = 0;
            dec.length = lower;
        }
        dec.renorm();

        if (bit == 0) bit_0_count++;
        if (--bits_until_update == 0) update();

        return bit;
    }
};
// ---------------------------------------------------------------
// 3. Integer Compressor
// ---------------------------------------------------------------
template<int BITS, int BITS_HIGH>
struct IntegerCompressor {
    SymbolModel model_k;
    BitModel model_bit_0; // Added for k=0
    SymbolModel model_corrector[BITS + 1];

    __device__ void init() {
        model_k.init(BITS + 1);
        model_bit_0.init();

        for (int i = 1; i <= BITS_HIGH; i++) {
            model_corrector[i].init(1 << i);
        }
        for (int i = BITS_HIGH + 1; i <= BITS; i++) {
            model_corrector[i].init(1 << BITS_HIGH);
        }
    }

    __device__ int32_t decode(ArithmeticDecoder& dec, int& out_k) {
        int k = model_k.decode(dec);
        out_k = k;

        // CRITICAL FIX: Decode the 1-bit corrector!
        if (k == 0) return model_bit_0.decode(dec);

        if (k == 32) return -(1 << 31);

        int32_t c;
        if (k <= BITS_HIGH) {
            c = model_corrector[k].decode(dec);
        }
        else {
            int k1 = k - BITS_HIGH;
            c = model_corrector[k].decode(dec);
            int c1 = dec.readBits(k1);
            c = (c << k1) | c1;
        }

        if (c >= (1 << (k - 1))) c += 1;
        else c -= ((1 << k) - 1);

        return c;
    }

    __device__ inline int32_t decode(ArithmeticDecoder& dec) {
        int dummy_k;
        return decode(dec, dummy_k);
    }
};

// ---------------------------------------------------------------
// Chunk State: Contains All Models
// ---------------------------------------------------------------

// ---------------------------------------------------------------
// 4. Chunk State (Complete context models for Point + RGB)
// ---------------------------------------------------------------
struct ChunkState {
    // -----------------------------------------------------------
    // Core Point Attributes Models
    // -----------------------------------------------------------
    // Tracks which fields have changed relative to the previous point
    SymbolModel model_changed_values;

    // Return Number / Number of Returns (Context: Previous Return Mask)
    SymbolModel model_bit_byte[256];

    // Intensity (Context: Return index 'm', up to 4 contexts)
    IntegerCompressor<16, 8> model_intensity[4];

    // Classification (Context: Previous Classification)
    SymbolModel model_classification[256];

    // Scan Angle Rank (Context: Flight line direction / edge of flight line)
    SymbolModel model_scan_angle[2];

    // User Data (Context: Previous User Data)
    SymbolModel model_user_data[256];

    // Point Source ID (Single integer compressor context)
    IntegerCompressor<16, 8> model_point_source;

    // -----------------------------------------------------------
    // Spatial Coordinate Models (X, Y, Z)
    // -----------------------------------------------------------
    // X context depends on the return index 'm'
    IntegerCompressor<32, 8> model_X[4];

    // Y context depends on the number of bits (k) used to encode X (22 contexts)
    IntegerCompressor<32, 8> model_Y[22];

    // Z context depends on the bits (k) used to encode X and Y (20 contexts)
    IntegerCompressor<32, 8> model_Z[20];

    // -----------------------------------------------------------
    // RGB12 Color Models
    // -----------------------------------------------------------
    // Tracks which color channels changed
    SymbolModel model_rgb_changed;

    // 6 byte-models: [0]=Red Low, [1]=Red High, [2]=Green Low, [3]=Green High, [4]=Blue Low, [5]=Blue High
    SymbolModel rgb_models[6];

    __device__ void init() {
        // Initialize Single Symbol Models
        model_changed_values.init(64);
        model_point_source.init();
        model_rgb_changed.init(128);

        // Initialize Context Arrays of size 256
        for (int i = 0; i < 256; i++) {
            model_bit_byte[i].init(256);
            model_classification[i].init(256);
            model_user_data[i].init(256);
        }

        // Initialize Intensity & X Coordinates (4 contexts based on return 'm')
        for (int i = 0; i < 4; i++) {
            model_intensity[i].init();
            model_X[i].init();
        }

        // Initialize Scan Angle (2 contexts based on scan direction bit)
        for (int i = 0; i < 2; i++) {
            model_scan_angle[i].init(256);
        }

        // Initialize Y Coordinates (22 contexts based on dX bit-length)
        for (int i = 0; i < 22; i++) {
            model_Y[i].init();
        }

        // Initialize Z Coordinates (20 contexts based on dX and dY bit-lengths)
        for (int i = 0; i < 20; i++) {
            model_Z[i].init();
        }

        // Initialize the 6 byte-based models for RGB channels
        for (int i = 0; i < 6; i++) {
            rgb_models[i].init(256);
        }
    }
};


struct StreamingMedian5 {
    int32_t values[5];
    int index;

    __device__ void init() {
        for (int i = 0; i < 5; i++) values[i] = 0;
        index = 0;
    }

    __device__ void add(int32_t v) {
        values[index] = v;
        index = (index + 1) % 5; // ring buffer
    }

    __device__ int32_t get() const {
        // Quick 5-element median sorting network
        int32_t v[5];
        for (int i = 0; i < 5; i++) v[i] = values[i];

        // Bubble sort (safe for N=5 in CUDA registers)
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4 - i; ++j) {
                if (v[j] > v[j + 1]) {
                    int32_t tmp = v[j];
                    v[j] = v[j + 1];
                    v[j + 1] = tmp;
                }
            }
        }
        return v[2]; // Return the middle (median) value
    }
};
// ---------------------------------------------------------------
// Kernel: Decode Points Per Chunk
// ---------------------------------------------------------------
__device__ inline int32_t read_i32(const uint8_t* ptr)
{
    // Reconstruct 32-bit int from 4 bytes (Little Endian)
    return (int32_t)(
        (uint32_t)ptr[0] |
        ((uint32_t)ptr[1] << 8) |
        ((uint32_t)ptr[2] << 16) |
        ((uint32_t)ptr[3] << 24));
}

__device__ inline uint16_t read_u16(const uint8_t* ptr)
{
    // Reconstruct 16-bit int from 2 bytes (Little Endian)
    return (uint16_t)(ptr[0] | (ptr[1] << 8));
}
__device__ inline int32_t ISum32(int32_t pred, int32_t diff) {
    return pred + diff; // Rely on two's complement wrap-around
}

__constant__ uint8_t number_return_map[8][8] = {
    {15, 14, 13, 12, 11, 10,  9,  8},
    {14,  0,  1,  3,  6, 10, 10,  9},
    {13,  1,  2,  4,  7, 11, 11, 10},
    {12,  3,  4,  5,  8, 12, 12, 11},
    {11,  6,  7,  8,  9, 13, 13, 12},
    {10, 10, 11, 12, 13, 14, 14, 13},
    { 9, 10, 11, 12, 13, 14, 15, 14},
    { 8,  9, 10, 11, 12, 13, 14, 15}
};

__constant__ uint8_t number_return_level[8][8] = {
    {0, 1, 2, 3, 4, 5, 6, 7},
    {1, 0, 1, 2, 3, 4, 5, 6},
    {2, 1, 0, 1, 2, 3, 4, 5},
    {3, 2, 1, 0, 1, 2, 3, 4},
    {4, 3, 2, 1, 0, 1, 2, 3},
    {5, 4, 3, 2, 1, 0, 1, 2},
    {6, 5, 4, 3, 2, 1, 0, 1},
    {7, 6, 5, 4, 3, 2, 1, 0}
};


__device__ inline int clamp255(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return v;
}

__global__ void laszip_format2_kernel(
    const uint8_t* __restrict__ compressed,
    const uint64_t* __restrict__ chunk_offsets,
    PointFormat2* __restrict__ out_points,
    int points_per_chunk,
    int total_chunks,
    ChunkState* states)
{
    int chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk_id >= total_chunks) return;

    ChunkState& state = states[chunk_id];
    state.init();

    int base_idx = chunk_id * points_per_chunk;
    uint64_t offset = chunk_offsets[chunk_id];

    PointFormat2& first = out_points[base_idx];

    // Read First Point Raw (SAFE UNALIGNED READS)
    first.X = read_i32(compressed + offset); offset += 4;
    first.Y = read_i32(compressed + offset); offset += 4;
    first.Z = read_i32(compressed + offset); offset += 4;

    first.Intensity = read_u16(compressed + offset); offset += 2;
    first.ReturnMask = compressed[offset++];
    first.Classification = compressed[offset++];
    first.ScanAngleRank = (int8_t)compressed[offset++];
    first.UserData = compressed[offset++];

    first.PointSourceID = read_u16(compressed + offset); offset += 2;
    first.Red = read_u16(compressed + offset); offset += 2;
    first.Green = read_u16(compressed + offset); offset += 2;
    first.Blue = read_u16(compressed + offset); offset += 2;

    ArithmeticDecoder dec;
    // Note: Ensure dec.init() also uses safe byte reading if it reads a 32-bit value internally!
    dec.init(compressed, offset);

    PointFormat2 prev = first;

    StreamingMedian5 median_X[16], median_Y[16];
    for (int i = 0; i < 16; i++) {
        median_X[i].init();
        median_Y[i].init();
    }

    int32_t last_Z[8] = { 0 };          // Z does NOT use a median filter!
    uint16_t last_Intensity[16] = { 0 };

    for (int i = 1; i < points_per_chunk; i++) {
        PointFormat2 p = prev; // Start delta off previous

        // 1. Decode Changed Values (Point10)
        uint32_t changed = state.model_changed_values.decode(dec);

        // 2. Decode Point10 attributes
        if (changed & (1 << 5)) {
            // Bit-Byte Decode
            p.ReturnMask = state.model_bit_byte[prev.ReturnMask].decode(dec);
        }
        int r = p.ReturnMask & 7;
        int n = (p.ReturnMask >> 3) & 7;
        int m = number_return_map[n][r];
        int I = number_return_level[n][r];

        if (changed & (1 << 4)) {
            int ctx_int = (m < 3) ? m : 3;
            int32_t dInt = state.model_intensity[ctx_int].decode(dec);
            p.Intensity = (uint16_t)(last_Intensity[m] + dInt);
        }
        else {
            p.Intensity = last_Intensity[m];
        }
        last_Intensity[m] = p.Intensity;
        if (changed & (1 << 3)) p.Classification = state.model_classification[prev.Classification].decode(dec);
        if (changed & (1 << 2)) p.ScanAngleRank = (p.ScanAngleRank + state.model_scan_angle[(p.ReturnMask >> 6) & 1].decode(dec)) % 256;
        if (changed & (1 << 1)) p.UserData = state.model_user_data[prev.UserData].decode(dec);
        if (changed & (1 << 0)) p.PointSourceID = ISum32(prev.PointSourceID, state.model_point_source.decode(dec));

        // 3. Decode Coordinates
        int k_x, k_y, k_z;
        int m_x = (m == 1) ? 1 : 0; // Context grouping based on return number

        // X Coordinate
        int ctx_x = (n == 1) ? 1 : 0;
        int32_t dx = state.model_X[ctx_x].decode(dec, k_x);
        p.X = prev.X + median_X[m].get() + dx;
        median_X[m].add(dx);

        // Y Coordinate
        int ctx_y = (k_x < 20) ? (k_x & ~1) : 20;
        if (n == 1) ctx_y += 1;
        int32_t dy = state.model_Y[ctx_y].decode(dec, k_y);
        p.Y = prev.Y + median_Y[m].get() + dy;
        median_Y[m].add(dy);

        // Z Coordinate (NO MEDIAN)
        int kXY = (k_x + k_y) / 2;
        int ctx_z = (kXY < 18) ? (kXY & ~1) : 18;
        if (n == 1) ctx_z += 1;
        int32_t dz = state.model_Z[ctx_z].decode(dec, k_z);
        p.Z = last_Z[I] + dz;
        last_Z[I] = p.Z;

        // 4. Decode RGB12 (Table 42)
// 4. Decode RGB12 (Table 42)
        uint32_t rgb_changed = state.model_rgb_changed.decode(dec);

        // Previous bytes
        uint8_t prev_r_low = prev.Red & 0xFF;
        uint8_t prev_r_high = (prev.Red >> 8) & 0xFF;
        uint8_t prev_g_low = prev.Green & 0xFF;
        uint8_t prev_g_high = (prev.Green >> 8) & 0xFF;
        uint8_t prev_b_low = prev.Blue & 0xFF;
        uint8_t prev_b_high = (prev.Blue >> 8) & 0xFF;

        // Red
        uint8_t r_low = prev_r_low;
        uint8_t r_high = prev_r_high;
        if (rgb_changed & (1u << 0)) {
            int dR_L = state.rgb_models[0].decode(dec);
            r_low = (prev_r_low + dR_L + 256) % 256;
        }
        if (rgb_changed & (1u << 1)) {
            int dR_H = state.rgb_models[1].decode(dec);
            r_high = (prev_r_high + dR_H + 256) % 256;
        }
        int diff_r_low = (int)r_low - (int)prev_r_low;
        int diff_r_high = (int)r_high - (int)prev_r_high;

        uint8_t g_low = prev_g_low;
        uint8_t g_high = prev_g_high;
        uint8_t b_low = prev_b_low;
        uint8_t b_high = prev_b_high;

        if (rgb_changed & (1u << 6)) {
            // G and B identical to the newly computed R
            g_low = r_low; g_high = r_high;
            b_low = r_low; b_high = r_high;
        }
        else {
            // Green low
            if (rgb_changed & (1u << 2)) {
                int dG_L = state.rgb_models[2].decode(dec);
                int base = clamp255(prev_g_low + diff_r_low);
                g_low = (dG_L + base + 256) % 256;
            }
            // Green high
            if (rgb_changed & (1u << 3)) {
                int dG_H = state.rgb_models[3].decode(dec);
                int base = clamp255(prev_g_high + diff_r_high);
                g_high = (dG_H + base + 256) % 256;
            }

            // Compute actual Green diffs (needed for Blue prediction)
            int diff_g_low = (int)g_low - (int)prev_g_low;
            int diff_g_high = (int)g_high - (int)prev_g_high;

            // Blue low
            if (rgb_changed & (1u << 4)) {
                int dB_L = state.rgb_models[4].decode(dec);
                int diff_b_L_pred = (diff_r_low + diff_g_low) / 2; // round toward 0 in C/C++
                int base = clamp255(prev_b_low + diff_b_L_pred);
                b_low = (dB_L + base + 256) % 256;
            }
            // Blue high
            if (rgb_changed & (1u << 5)) {
                int dB_H = state.rgb_models[5].decode(dec);
                int diff_b_H_pred = (diff_r_high + diff_g_high) / 2; // round toward 0
                int base = clamp255(prev_b_high + diff_b_H_pred);
                b_high = (dB_H + base + 256) % 256;
            }
        }

        p.Red = (uint16_t(r_high) << 8) | r_low;
        p.Green = (uint16_t(g_high) << 8) | g_low;
        p.Blue = (uint16_t(b_high) << 8) | b_low;
        if (chunk_id == 0 && i < 5) {
            // Build a small bit string for changed bits 0..6
            unsigned c = rgb_changed;
            printf("RGB12 DBG pt=%d rgb_changed=0x%02x bits=[%d%d%d%d%d%d%d]\n",
                i, (unsigned)(c & 0x7F),
                (c >> 6) & 1, (c >> 5) & 1, (c >> 4) & 1, (c >> 3) & 1, (c >> 2) & 1, (c >> 1) & 1, (c >> 0) & 1);

            printf("  prev  R(l,h)=(%3d,%3d)  G(l,h)=(%3d,%3d)  B(l,h)=(%3d,%3d)\n",
                (int)prev_r_low, (int)prev_r_high,
                (int)prev_g_low, (int)prev_g_high,
                (int)prev_b_low, (int)prev_b_high);

            // Print Red results and diffs
            printf("  R     r_low=%3d r_high=%3d  diff_r(l,h)=(%4d,%4d)\n",
                (int)r_low, (int)r_high, (int)diff_r_low, (int)diff_r_high);

            if (rgb_changed & (1u << 6)) {
                printf("  G/B   bit6 set -> G=B=R  g(l,h)=(%3d,%3d) b(l,h)=(%3d,%3d)\n",
                    (int)g_low, (int)g_high, (int)b_low, (int)b_high);
            }
            else {
                // Compute diffs actually used (already computed above)
                int diff_g_low = (int)g_low - (int)prev_g_low;
                int diff_g_high = (int)g_high - (int)prev_g_high;

                // Recompute predictor bases to print them (match your code)
                int base_g_l = clamp255((int)prev_g_low + diff_r_low);
                int base_g_h = clamp255((int)prev_g_high + diff_r_high);
                int pred_b_l = (diff_r_low + diff_g_low) / 2; // round toward 0
                int pred_b_h = (diff_r_high + diff_g_high) / 2;
                int base_b_l = clamp255((int)prev_b_low + pred_b_l);
                int base_b_h = clamp255((int)prev_b_high + pred_b_h);

                printf("  G     g_low=%3d g_high=%3d  base(l,h)=(%3d,%3d)  diff_g(l,h)=(%4d,%4d)  bits(L,H)=(%d,%d)\n",
                    (int)g_low, (int)g_high, base_g_l, base_g_h, diff_g_low, diff_g_high,
                    (int)((c >> 2) & 1), (int)((c >> 3) & 1));

                printf("  B     b_low=%3d b_high=%3d  base(l,h)=(%3d,%3d)  pred_avg(l,h)=(%4d,%4d)  bits(L,H)=(%d,%d)\n",
                    (int)b_low, (int)b_high, base_b_l, base_b_h, pred_b_l, pred_b_h,
                    (int)((c >> 4) & 1), (int)((c >> 5) & 1));
            }

            printf("  RGB16 final = (%5u,%5u,%5u)\n",
                (unsigned)(((uint16_t)r_high << 8) | r_low),
                (unsigned)(((uint16_t)g_high << 8) | g_low),
                (unsigned)(((uint16_t)b_high << 8) | b_low));
        }
        out_points[base_idx + i] = p;
        prev = p;
    }
}



int decompress(const std::vector<uint8_t>& raw_file_data,
    uint32_t num_chunks,
    const std::vector<uint64_t>& chunk_offsets,
    uint64_t actual_total_points)
{
    int points_per_chunk = 5000;
    uint64_t total_points = actual_total_points;
    size_t raw_byte_size = raw_file_data.size();

    // Status variable to track success/failure
    int status = 0;

    // Device Pointers initialized to nullptr so cudaFree is safe to call on them
    uint8_t* d_compressed = nullptr;
    uint64_t* d_chunk_offsets = nullptr;
    PointFormat2* d_out_points = nullptr;
    ChunkState* d_states = nullptr;
    // Allocate Device Memory
    gpuErrchk(cudaMalloc(&d_compressed, raw_byte_size));
    gpuErrchk(cudaMalloc(&d_chunk_offsets, num_chunks * sizeof(uint64_t)));
    gpuErrchk(cudaMalloc(&d_out_points, total_points * sizeof(PointFormat2)));
    gpuErrchk(cudaMalloc(&d_states, num_chunks * sizeof(ChunkState)));
    // Copy to Device
    gpuErrchk(cudaMemcpy(d_compressed, raw_file_data.data(), raw_byte_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_chunk_offsets, chunk_offsets.data(), num_chunks * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Kernel Configuration
    int threads_per_block = 32;
    int blocks = (num_chunks + threads_per_block - 1) / threads_per_block;

    // Launch Kernel
    std::cout << "Launching Kernel: " << blocks << " blocks, "
        << threads_per_block << " threads/block..." << std::endl;
    laszip_format2_kernel << <blocks, threads_per_block >> > (
        d_compressed, d_chunk_offsets, d_out_points,
        points_per_chunk, num_chunks, d_states
        );
    std::cout << "Kernel execution complete.\n";
    // 1st Check: Catch synchronous errors (e.g., invalid grid/block dimensions)
    gpuErrchk(cudaGetLastError());

    // 2nd Check: Catch asynchronous errors (e.g., memory out-of-bounds inside the kernel)
    gpuErrchk(cudaDeviceSynchronize());

    // Copy Results Back
    {
        std::vector<PointFormat2> h_points(total_points);
        gpuErrchk(cudaMemcpy(h_points.data(), d_out_points, total_points * sizeof(PointFormat2), cudaMemcpyDeviceToHost));

        int num_to_print = std::min(10, (int)h_points.size());
        std::cout << "\n--- First " << num_to_print << " Points ---\n";
        for (int i = 0; i < num_to_print; i++) {
            const auto& p = h_points[i];

            // Note: We cast uint8_t and int8_t to (int) so std::cout prints 
            // numbers instead of ASCII characters.
            std::cout << "Point " << i << ": "
                << "X=" << p.X << ", "
                << "Y=" << p.Y << ", "
                << "Z=" << p.Z << " | "
                << "Int=" << p.Intensity << " | "
                //<< "Return=" << (int)p.ReturnMask << " | "
                << "Class=" << (int)p.Classification << " | "
                //<< "Scan=" << (int)p.ScanAngleRank << " | "
               // << "User=" << (int)p.UserData << " | "
                << "RGB=(" << p.Red << "," << p.Green << "," << p.Blue << ")\n";
        }
        std::cout << "-----------------------\n\n";
    }


    if (d_compressed) cudaFree(d_compressed);
    if (d_chunk_offsets) cudaFree(d_chunk_offsets);
    if (d_out_points) cudaFree(d_out_points);

    return status;
}