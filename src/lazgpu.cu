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

    __device__ __forceinline__ void init(const uint8_t* data, uint32_t offset) {
        buffer = data;
        pos = offset;
        length = 0xFFFFFFFFU;
        value = (uint32_t)buffer[pos] << 24 | (uint32_t)buffer[pos + 1] << 16 |
            (uint32_t)buffer[pos + 2] << 8 | (uint32_t)buffer[pos + 3];
        pos += 4;
    }

    __device__ __forceinline__ void renorm() {
        while (length < AC__MinLength) {
            value = (value << 8) | buffer[pos++];
            length <<= 8;
        }
    }

    __device__ __forceinline__ uint32_t readBits(int bit_count) {
        if (bit_count > 19) {
            uint32_t lower = readBits(16);
            return lower | (readBits(bit_count - 16) << 16);
        }
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
    uint16_t symbol_count[256];
    uint16_t distribution[256];
    uint32_t update_cycle;
    uint32_t symbols_until_update;
    uint32_t num_symbols;

    __device__ void init(uint32_t symbols) {
        num_symbols = symbols;
        for (uint32_t i = 0; i < num_symbols; i++) symbol_count[i] = 1;

        update_cycle = num_symbols; // Add this line
        update_distribution();

        update_cycle = (num_symbols + 6) / 2;
        symbols_until_update = update_cycle;
    }

    __device__ void update_distribution() {
        uint32_t total_count = 0;
        for (uint32_t i = 0; i < num_symbols; i++) total_count += symbol_count[i];

        if (total_count > 32768) {
            total_count = 0;
            for (uint32_t i = 0; i < num_symbols; i++) {
                symbol_count[i] = (symbol_count[i] + 1) >> 1;
                total_count += symbol_count[i];
            }
        }

        uint32_t sum = 0;
        uint32_t scale = 0x80000000U / total_count;
        for (uint32_t i = 0; i < num_symbols; i++) {
            distribution[i] = (uint16_t)((scale * sum) >> 16);
            sum += symbol_count[i];
        }
        update_cycle += (update_cycle >> 2);
        if (update_cycle > 8 * (num_symbols + 6)) update_cycle = 8 * (num_symbols + 6);
        symbols_until_update = update_cycle;
    }

    __device__ __forceinline__ uint32_t decode(ArithmeticDecoder& dec) {
        uint32_t ltmp = dec.length >> 15;

        // Binary search on distribution[0..num_symbols-1]
        int lo = 0, hi = (int)num_symbols - 1, sym = 0;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            uint32_t lower_mid = (uint32_t)distribution[mid] * ltmp;
            if (lower_mid <= dec.value) {
                sym = mid;         // mid is a feasible lower bound
                lo = mid + 1;
            }
            else {
                hi = mid - 1;
            }
        }

        uint32_t lower = (uint32_t)distribution[sym] * ltmp;
        dec.value -= lower;
        if ((uint32_t)sym < num_symbols - 1) {
            dec.length = ((uint32_t)distribution[sym + 1] * ltmp) - lower;
        }
        else {
            dec.length -= lower;
        }
        dec.renorm();

        symbol_count[sym]++;                      // adapt as before
        if (--symbols_until_update == 0) update_distribution();
        return sym;
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

        // Exactly mirror the C++ precision loss
        uint32_t scale = 0x80000000U / bit_count;
        bit_0_prob = (bit_0_count * scale) >> 18;

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
// Add CONTEXTS to the template arguments
template<int BITS, int CONTEXTS, int BITS_HIGH>
struct IntegerCompressor {
    // Array of contexts for 'k'
    SymbolModel model_k[CONTEXTS];

    // Shared correctors
    BitModel model_bit_0;
    SymbolModel model_corrector[BITS + 1];

    __device__ void init() {
        for (int i = 0; i < CONTEXTS; i++) {
            model_k[i].init(BITS + 1);
        }
        model_bit_0.init();

        for (int i = 1; i <= BITS_HIGH; i++) {
            model_corrector[i].init(1 << i);
        }
        for (int i = BITS_HIGH + 1; i <= BITS; i++) {
            model_corrector[i].init(1 << BITS_HIGH);
        }
    }

    // Pass the context into the decode function
    __device__ int32_t decode(ArithmeticDecoder& dec, int context, int& out_k) {
        int k = model_k[context].decode(dec);
        out_k = k;

        if (k == 0) return model_bit_0.decode(dec);
        if (k == 32) return INT32_MIN;

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

    __device__ inline int32_t decode(ArithmeticDecoder& dec, int context) {
        int dummy_k;
        return decode(dec, context, dummy_k);
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
 

    // Classification (Context: Previous Classification)
    SymbolModel model_classification[256];

    // Scan Angle Rank (Context: Flight line direction / edge of flight line)
    SymbolModel model_scan_angle[2];

    // User Data (Context: Previous User Data)
    SymbolModel model_user_data[256];


    // -----------------------------------------------------------
    // Spatial Coordinate Models (X, Y, Z)
    // -----------------------------------------------------------
    IntegerCompressor<16, 4, 8> model_intensity;
    IntegerCompressor<16, 1, 8> model_point_source;

    IntegerCompressor<32, 2, 8> model_X;
    IntegerCompressor<32, 22, 8> model_Y;
    IntegerCompressor<32, 20, 8> model_Z;

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
        model_rgb_changed.init(128);

        // Initialize Context Arrays of size 256
        for (int i = 0; i < 256; i++) {
            model_bit_byte[i].init(256);
            model_classification[i].init(256);
            model_user_data[i].init(256);
        }

        // Initialize Scan Angle (2 contexts based on scan direction bit)
        for (int i = 0; i < 2; i++) {
            model_scan_angle[i].init(256);
        }


        // Initialize the 6 byte-based models for RGB channels
        for (int i = 0; i < 6; i++) {
            rgb_models[i].init(256);
        }

        model_point_source.init();
        model_intensity.init();
        model_X.init();
        model_Y.init();
        model_Z.init();
    }
};


struct StreamingMedian5 {
    int32_t values[5];
    bool remove_largest; // State tracking for the removal rule

    __device__ void init() {
        for (int i = 0; i < 5; i++) values[i] = 0;
        remove_largest = true; // First insertion always removes largest
    }

    __device__ void add(int32_t v) {
        // 1. Sort current values to find current median and min/max
        // Using a simple insertion sort logic for in-place maintenance
        sort_internal();

        int32_t current_median = values[2];

        // 2. Determine index to replace
        int replace_idx = remove_largest ? 4 : 0;

        // 3. Update the value
        values[replace_idx] = v;

        // 4. Update the state for the NEXT insertion
        bool current_was_largest = remove_largest;

        if (v < current_median) {
            remove_largest = true;
        }
        else if (v > current_median) {
            remove_largest = false;
        }
        else {
            // Rule: "removes the opposite of the current insertion"
            remove_largest = !current_was_largest;
        }

        // 5. Re-sort so get() is always O(1) and indices are predictable
        sort_internal();
    }

    __device__ int32_t get() const {
        return values[2]; // Return the 3rd ordered value
    }

private:
    __device__ void sort_internal() {
        // Optimized Bubble Sort for 5 elements
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4 - i; j++) {
                if (values[j] > values[j + 1]) {
                    int32_t tmp = values[j];
                    values[j] = values[j + 1];
                    values[j + 1] = tmp;
                }
            }
        }
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
#define U8_FOLD(v) ((uint8_t)((v) & 0xFF))
__global__ void laszip_format2_kernel(
    const uint8_t* __restrict__ compressed,
    const uint64_t* __restrict__ chunk_offsets,
    PointFormat2* __restrict__ out_points,
    int points_per_chunk,
    int total_chunks,
	uint64_t total_points,
    ChunkState* states)
{
    int chunk_id = blockIdx.x;
	if (threadIdx.x > 0) return;
    if (chunk_id >= total_chunks-1) return;
    int tid = threadIdx.x;

    // Shared flag to signal workers
    __shared__ int signal_update;
    if (tid == 0) signal_update = -1;
    __syncthreads();
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
    
    for (int i = 1; i < points_per_chunk && i < total_points; i++) {
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
            int32_t dInt = state.model_intensity.decode(dec, ctx_int);
            p.Intensity = (uint16_t)(last_Intensity[m] + dInt);
        }
        else {
            p.Intensity = last_Intensity[m];
        }
        last_Intensity[m] = p.Intensity;

        if (changed & (1 << 3)) p.Classification = state.model_classification[prev.Classification].decode(dec);
        if (changed & (1 << 2)) p.ScanAngleRank = U8_FOLD(p.ScanAngleRank + state.model_scan_angle[(p.ReturnMask >> 6) & 1].decode(dec));
        if (changed & (1 << 1)) p.UserData = state.model_user_data[prev.UserData].decode(dec);
        if (changed & (1 << 0)) p.PointSourceID = prev.PointSourceID + state.model_point_source.decode(dec, 0);

        // 3. Decode Coordinates
        int k_x, k_y, k_z;

        // X Coordinate
        int ctx_x = (n == 1) ? 1 : 0;
        int32_t dx = state.model_X.decode(dec, ctx_x, k_x);
        int32_t dxMedian = median_X[m].get() + dx;
        p.X = ISum32(prev.X, dxMedian);
        median_X[m].add(dxMedian);
        // Y Coordinate
        int ctx_y = (k_x < 20) ? (k_x & ~1) : 20;
        if (n == 1) ctx_y += 1;
        int32_t dy = state.model_Y.decode(dec, ctx_y, k_y);
        int32_t dyMedian = median_Y[m].get() + dy;
        p.Y = ISum32(prev.Y, dyMedian);
        median_Y[m].add(dyMedian);
        // Z Coordinate 
        int kXY = (k_x + k_y) / 2;
        int ctx_z = (kXY < 18) ? (kXY & ~1) : 18;
        if (n == 1) ctx_z += 1;
        int32_t dz = state.model_Z.decode(dec, ctx_z, k_z);

    
        p.Z = last_Z[I] + dz;
        last_Z[I] = p.Z;

        // 4. Decode RGB12 (Table 42)
        uint32_t sym = state.model_rgb_changed.decode(dec);

        uint16_t last_r = prev.Red;
        uint16_t last_g = prev.Green;
        uint16_t last_b = prev.Blue;

        uint16_t current_r, current_g, current_h;
        uint8_t r_low, r_high;

        // 1. Decode Red
        if (sym & (1 << 0)) {
            uint8_t corr = (uint8_t)state.rgb_models[0].decode(dec);
            r_low = U8_FOLD(corr + (last_r & 0xFF));
        }
        else {
            r_low = last_r & 0xFF;
        }

        if (sym & (1 << 1)) {
            uint8_t corr = (uint8_t)state.rgb_models[1].decode(dec);
            r_high = U8_FOLD(corr + (last_r >> 8));
        }
        else {
            r_high = (last_r >> 8) & 0xFF;
        }

        uint16_t final_red = (uint16_t)r_low | ((uint16_t)r_high << 8);

        if (sym & (1 << 6)) {
            // --- CORRELATED MODE (Bit 6 is SET in code) ---
            int32_t diff;
            uint8_t g_low, g_high, b_low, b_high;

            // Process LOW bytes
            diff = (int32_t)r_low - (int32_t)(last_r & 0xFF);

            // Green Low
            if (sym & (1 << 2)) {
                uint8_t corr = (uint8_t)state.rgb_models[2].decode(dec);
                g_low = U8_FOLD(corr + clamp255(diff + (last_g & 0xFF)));
            }
            else {
                g_low = last_g & 0xFF;
            }

            // Blue Low
            if (sym & (1 << 4)) {
                uint8_t corr = (uint8_t)state.rgb_models[4].decode(dec);
                // Re-calculate diff as average of Red and Green deltas
                int32_t blue_diff = (diff + ((int32_t)g_low - (int32_t)(last_g & 0xFF))) / 2;
                b_low = U8_FOLD(corr + clamp255(blue_diff + (last_b & 0xFF)));
            }
            else {
                b_low = last_b & 0xFF;
            }

            // Process HIGH bytes
            diff = (int32_t)r_high - (int32_t)(last_r >> 8);

            // Green High
            if (sym & (1 << 3)) {
                uint8_t corr = (uint8_t)state.rgb_models[3].decode(dec);
                g_high = U8_FOLD(corr + clamp255(diff + (last_g >> 8)));
            }
            else {
                g_high = (last_g >> 8) & 0xFF;
            }

            // Blue High
            if (sym & (1 << 5)) {
                uint8_t corr = (uint8_t)state.rgb_models[5].decode(dec);
                int32_t blue_diff = (diff + ((int32_t)g_high - (int32_t)(last_g >> 8))) / 2;
                b_high = U8_FOLD(corr + clamp255(blue_diff + (last_b >> 8)));
            }
            else {
                b_high = (last_b >> 8) & 0xFF;
            }

            p.Red = final_red;
            p.Green = (uint16_t)g_low | ((uint16_t)g_high << 8);
            p.Blue = (uint16_t)b_low | ((uint16_t)b_high << 8);
        }
        else {
            // --- DUPLICATION MODE (Bit 6 is NOT SET) ---
            p.Red = final_red;
            p.Green = final_red;
            p.Blue = final_red;
        }
        out_points[base_idx + i] = p;
        prev = p;

    }
}



int decompress(const std::vector<uint8_t>& raw_file_data,
    uint32_t num_chunks,
    const std::vector<uint64_t>& chunk_offsets,
    uint64_t actual_total_points,
    std::vector<PointFormat2>& points)
{
    int points_per_chunk = 50000;
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
    int threads_per_block = 1;
    int blocks = (num_chunks + threads_per_block - 1) / threads_per_block;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    // Launch Kernel
    std::cout << "Launching Kernel: " << blocks << " blocks, "
        << threads_per_block << " threads/block..." << std::endl;
    laszip_format2_kernel << <num_chunks, 32 >> > (
        d_compressed, d_chunk_offsets, d_out_points,
        points_per_chunk, num_chunks, total_points, d_states
        );
    std::cout << "Kernel execution complete.\n";
    // 1st Check: Catch synchronous errors (e.g., invalid grid/block dimensions)
    gpuErrchk(cudaGetLastError());

    // 2nd Check: Catch asynchronous errors (e.g., memory out-of-bounds inside the kernel)
    gpuErrchk(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    // Copy Results Back
    {
        points = std::vector<PointFormat2>(total_points);
        gpuErrchk(cudaMemcpy(points.data(), d_out_points, total_points * sizeof(PointFormat2), cudaMemcpyDeviceToHost));

        int num_to_print = std::min(0, (int)points.size());
        std::cout << "\n--- First " << num_to_print << " Points ---\n";
        for (int i = 0; i < num_to_print; i++) {
            const auto& p = points[i];

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

    double pts_per_ms_kernel = (ms > 0.0f) ? (double)total_points / (double)ms : 0.0;
    //double pts_per_ms_e2e = (ms > 0.0) ? (double)total_points / ms : 0.0;

    std::cout << "Decoded " << total_points << " points\n"
        << "Kernel time: " << ms << " ms, throughput: "
        << pts_per_ms_kernel << " points/ms\n";
        //<< "End-to-end time: " << ms << " ms, throughput: "
        //<< pts_per_ms_e2e << " points/ms\n"
        

    if (d_compressed) cudaFree(d_compressed);
    if (d_chunk_offsets) cudaFree(d_chunk_offsets);
    if (d_out_points) cudaFree(d_out_points);

    return status;
}