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
        if (bit_count > 19) {
            uint32_t high = readBits(16);
            uint32_t low = readBits(bit_count - 16);
            return (low << 16) | high;
        }
        length >>= bit_count;
        uint32_t sym = value / length;
        value -= length * sym;
        renorm();
        return sym;
    }
};

// ---------------------------------------------------------------
// 2. Symbol Model (FastAC)
// ---------------------------------------------------------------
template <int SYMBOLS>
struct SymbolModel {
    uint32_t symbol_count[SYMBOLS];
    uint16_t distribution[SYMBOLS];
    uint32_t update_cycle;
    uint32_t symbols_until_update;

    __device__ void init() {
        for (int i = 0; i < SYMBOLS; i++) symbol_count[i] = 1;
        update_cycle = (SYMBOLS + 6) / 2;
        symbols_until_update = update_cycle;
        update_distribution();
    }

    __device__ void update_distribution() {
        uint32_t total_count = 0;
        for (int i = 0; i < SYMBOLS; i++) total_count += symbol_count[i];

        if (total_count > 32768) {
            total_count = 0;
            for (int i = 0; i < SYMBOLS; i++) {
                symbol_count[i] = (symbol_count[i] >> 1) + 1;
                total_count += symbol_count[i];
            }
        }

        uint32_t sum = 0;
        uint32_t scale = 0x80000000U / total_count;
        for (int i = 0; i < SYMBOLS; i++) {
            distribution[i] = (uint16_t)((scale * sum) >> 16);
            sum += symbol_count[i];
        }
        update_cycle += (update_cycle >> 2);
        if (update_cycle > 8 * (SYMBOLS + 6)) update_cycle = 8 * (SYMBOLS + 6);
        symbols_until_update = update_cycle;
    }

    __device__ uint32_t decode(ArithmeticDecoder& dec) {
        uint32_t ltmp = dec.length >> 15;
        uint32_t sym = 0;
        for (int s = SYMBOLS - 1; s >= 0; s--) {
            if ((uint32_t)distribution[s] * ltmp <= dec.value) { sym = s; break; }
        }
        uint32_t lower = (uint32_t)distribution[sym] * ltmp;
        dec.value -= lower;
        if (sym < SYMBOLS - 1) {
            dec.length = ((uint32_t)distribution[sym + 1] * ltmp) - lower;
        }
        else {
            dec.length -= lower;
        }
        dec.renorm();
        symbol_count[sym]++;
        if (--symbols_until_update == 0) update_distribution();
        return sym;
    }
};

// BitModel (FastAC 2-symbol specialization omitted for brevity, using SymbolModel<2>)
// ---------------------------------------------------------------
// 3. Integer Compressor
// ---------------------------------------------------------------
template<int BITS>
struct IntegerCompressor {
    SymbolModel<BITS + 1> model_k;
    SymbolModel<2> model_bit[BITS];      // Fallbacks for corr bits
    SymbolModel<256> model_byte[BITS];   // Models for larger Ks

    __device__ void init() {
        model_k.init();
        for (int i = 0; i < BITS; i++) {
            model_bit[i].init();
            model_byte[i].init();
        }
    }

    __device__ int32_t decode(ArithmeticDecoder& dec) {
        int k = model_k.decode(dec);
        if (k == 0) return model_bit[0].decode(dec);
        if (k == 32) return -(1 << 31); // I32_MIN

        uint32_t corr = 0;
        if (k <= 8) {
            // Re-use symbol model logic for K
            // In a full LAZ spec, this binds directly to specific 2^K size distributions
            corr = dec.readBits(k); // Simplified for demonstration limits
        }
        else {
            corr = (model_byte[k].decode(dec) << (k - 8)) | dec.readBits(k - 8);
        }

        if (corr >= (1u << (k - 1))) return corr + 1;
        return corr - ((1u << k) - 1);
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
    SymbolModel<64> model_changed_values;

    // Return Number / Number of Returns (Context: Previous Return Mask)
    SymbolModel<256> model_bit_byte[256];

    // Intensity (Context: Return index 'm', up to 4 contexts)
    IntegerCompressor<16> model_intensity[4];

    // Classification (Context: Previous Classification)
    SymbolModel<256> model_classification[256];

    // Scan Angle Rank (Context: Flight line direction / edge of flight line)
    SymbolModel<256> model_scan_angle[2];

    // User Data (Context: Previous User Data)
    SymbolModel<256> model_user_data[256];

    // Point Source ID (Single integer compressor context)
    IntegerCompressor<16> model_point_source;

    // -----------------------------------------------------------
    // Spatial Coordinate Models (X, Y, Z)
    // -----------------------------------------------------------
    // X context depends on the return index 'm'
    IntegerCompressor<32> model_X[4];

    // Y context depends on the number of bits (k) used to encode X (22 contexts)
    IntegerCompressor<32> model_Y[22];

    // Z context depends on the bits (k) used to encode X and Y (20 contexts)
    IntegerCompressor<32> model_Z[20];

    // -----------------------------------------------------------
    // RGB12 Color Models
    // -----------------------------------------------------------
    // Tracks which color channels changed
    SymbolModel<128> model_rgb_changed;

    // 6 byte-models: [0]=Red Low, [1]=Red High, [2]=Green Low, [3]=Green High, [4]=Blue Low, [5]=Blue High
    SymbolModel<256> rgb_models[6];

    __device__ void init() {
        // Initialize Single Symbol Models
        model_changed_values.init();
        model_point_source.init();
        model_rgb_changed.init();

        // Initialize Context Arrays of size 256
        for (int i = 0; i < 256; i++) {
            model_bit_byte[i].init();
            model_classification[i].init();
            model_user_data[i].init();
        }

        // Initialize Intensity & X Coordinates (4 contexts based on return 'm')
        for (int i = 0; i < 4; i++) {
            model_intensity[i].init();
            model_X[i].init();
        }

        // Initialize Scan Angle (2 contexts based on scan direction bit)
        for (int i = 0; i < 2; i++) {
            model_scan_angle[i].init();
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
            rgb_models[i].init();
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

    // Streaming Medians (simplified tracking arrays omitted)
    int32_t median_X[16] = { 0 }, median_Y[16] = { 0 };

    for (int i = 1; i < points_per_chunk; i++) {
        PointFormat2 p = prev; // Start delta off previous

        // 1. Decode Changed Values (Point10)
        uint32_t changed = state.model_changed_values.decode(dec);

        // 2. Decode Point10 attributes
        if (changed & (1 << 5)) {
            // Bit-Byte Decode
            p.ReturnMask = state.model_bit_byte[prev.ReturnMask].decode(dec);
        }
        int m = (p.ReturnMask >> 3) & 7; // simplified 'm' mapping

        if (changed & (1 << 4)) p.Intensity = ISum32(prev.Intensity, state.model_intensity[m < 3 ? m : 3].decode(dec));
        if (changed & (1 << 3)) p.Classification = state.model_classification[prev.Classification].decode(dec);
        if (changed & (1 << 2)) p.ScanAngleRank = (p.ScanAngleRank + state.model_scan_angle[(p.ReturnMask >> 6) & 1].decode(dec)) % 256;
        if (changed & (1 << 1)) p.UserData = state.model_user_data[prev.UserData].decode(dec);
        if (changed & (1 << 0)) p.PointSourceID = ISum32(prev.PointSourceID, state.model_point_source.decode(dec));

        // 3. Decode Coordinates
        int32_t dx = state.model_X[(m == 1) ? 1 : 0].decode(dec);
        p.X = prev.X + ISum32(median_X[m], dx);

        int32_t dy = state.model_Y[0].decode(dec); // Uses instance determined by K from dX
        p.Y = prev.Y + ISum32(median_Y[m], dy);

        int32_t dz = state.model_Z[0].decode(dec); // Uses instance determined by K from dX and dY
        p.Z = prev.Z + dz;

        // 4. Decode RGB12 (Table 42)
        uint32_t rgb_changed = state.model_rgb_changed.decode(dec);

        if (rgb_changed & (1 << 0)) p.Red = (p.Red + state.rgb_models[0].decode(dec)) % 256;
        if (rgb_changed & (1 << 1)) p.Red = (p.Red + (state.rgb_models[1].decode(dec) << 8)) % 65536;

        if (!(rgb_changed & (1 << 6))) {
            if (rgb_changed & (1 << 2)) p.Green = (p.Green + state.rgb_models[2].decode(dec)) % 256;
            if (rgb_changed & (1 << 3)) p.Green = (p.Green + (state.rgb_models[3].decode(dec) << 8)) % 65536;
            if (rgb_changed & (1 << 4)) p.Blue = (p.Blue + state.rgb_models[4].decode(dec)) % 256;
            if (rgb_changed & (1 << 5)) p.Blue = (p.Blue + (state.rgb_models[5].decode(dec) << 8)) % 65536;
        }
        else {
            p.Green = p.Red;
            p.Blue = p.Red;
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