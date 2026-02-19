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

struct RangeDecoder {
    uint32_t range;
    uint32_t code;
    const uint8_t* buffer;
    uint32_t pos;

    __device__ void init(const uint8_t* data, uint32_t offset) {
        buffer = data;
        pos = offset;
        range = 0xFFFFFFFF;
        code = 0;
        for (int i = 0; i < 4; i++)
            code = (code << 8) | buffer[pos++];
    }

    __device__ inline uint8_t read_byte() {
        return buffer[pos++];
    }

    __device__ inline void renorm() {
        while (range < RENORM_THRESHOLD) {
            range <<= 8;
            code = (code << 8) | read_byte();
        }
    }
};

// ---------------------------------------------------------------
// Binary Probability Model
// ---------------------------------------------------------------

struct BitModel {
    uint16_t prob;
    __device__ void init() { prob = 1 << 15; }
};

__device__ inline uint32_t decode_bit(RangeDecoder& dec, BitModel& m) {
    uint32_t bound = (dec.range >> 16) * m.prob;
    uint32_t bit;
    if (dec.code < bound) {
        dec.range = bound;
        m.prob += (0xFFFF - m.prob) >> 5;
        bit = 0;
    }
    else {
        dec.code -= bound;
        dec.range -= bound;
        m.prob -= (m.prob) >> 5;
        bit = 1;
    }
    dec.renorm();
    return bit;
}

// ---------------------------------------------------------------
// Integer Decoder (Unary k + Magnitude + Sign)
// ---------------------------------------------------------------

struct IntegerModel {
    BitModel k_model[32];
    BitModel sign_model;
    BitModel mag_model[32];

    __device__ void init() {
        for (int i = 0; i < 32; i++) {
            k_model[i].init();
            mag_model[i].init();
        }
        sign_model.init();
    }

    __device__ int32_t decode(RangeDecoder& dec) {
        int k = 0;
        while (k < 31 && decode_bit(dec, k_model[k])) k++;

        uint32_t val = 0;
        if (k > 0) {
            val = (1u << k);
            for (int i = 0; i < k; i++) {
                val |= (decode_bit(dec, mag_model[i]) << i);
            }
            val -= 1;
        }
        if (val != 0 && decode_bit(dec, sign_model))
            return -(int32_t)val;
        return (int32_t)val;
    }
};

// ---------------------------------------------------------------
// Chunk State: Contains All Models
// ---------------------------------------------------------------

struct ChunkState {
    IntegerModel model_X[4];    // context by return class
    IntegerModel model_Y[4];
    IntegerModel model_Z[4];

    IntegerModel model_Intensity;
    IntegerModel model_Returns;
    IntegerModel model_Class;
    IntegerModel model_Angle;
    IntegerModel model_User;
    IntegerModel model_Source;

    IntegerModel model_G;
    IntegerModel model_R;
    IntegerModel model_B;

    __device__ void init() {
        for (int i = 0; i < 4; i++) {
            model_X[i].init();
            model_Y[i].init();
            model_Z[i].init();
        }
        model_Intensity.init();
        model_Returns.init();
        model_Class.init();
        model_Angle.init();
        model_User.init();
        model_Source.init();
        model_G.init();
        model_R.init();
        model_B.init();
    }
};

// ---------------------------------------------------------------
// Kernel: Decode Points Per Chunk
// ---------------------------------------------------------------

__device__ inline int32_t read_i32(const uint8_t* ptr)
{
    return (int32_t)(
        (uint32_t)ptr[0] |
        ((uint32_t)ptr[1] << 8) |
        ((uint32_t)ptr[2] << 16) |
        ((uint32_t)ptr[3] << 24));
}

__device__ inline uint16_t read_u16(const uint8_t* ptr)
{
    return (uint16_t)(ptr[0] | (ptr[1] << 8));
}

__global__
void laszip_format2_kernel(
    const uint8_t* __restrict__ compressed,
    const uint64_t* __restrict__ chunk_offsets,
    PointFormat2* __restrict__ out_points,
    int points_per_chunk,
    int total_chunks)
{
    int chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk_id >= total_chunks) return;

    ChunkState state;
    state.init();

    int base_idx = chunk_id * points_per_chunk;
    uint64_t offset = chunk_offsets[chunk_id];

    PointFormat2& first = out_points[base_idx];

    // ----------------------------------------------------
    // 1️⃣ Read FIRST POINT RAW (no range coder yet)
    // ----------------------------------------------------

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

    // ----------------------------------------------------
    // 2️⃣ NOW initialize range decoder
    // ----------------------------------------------------

    RangeDecoder dec;
    dec.init(compressed, offset);

    // ----------------------------------------------------
    // 3️⃣ Initialize predictors
    // ----------------------------------------------------

    int32_t pred_X = first.X;
    int32_t pred_Y = first.Y;
    int32_t pred_Z = first.Z;

    uint16_t pred_G = first.Green;
    uint16_t pred_R = first.Red;
    uint16_t pred_B = first.Blue;

    // ----------------------------------------------------
    // 4️⃣ Decode remaining points
    // ----------------------------------------------------

    for (int i = 1; i < points_per_chunk; i++)
    {
        int ctx = first.ReturnMask & 3; // placeholder context logic

        pred_X += state.model_X[ctx].decode(dec);
        pred_Y += state.model_Y[ctx].decode(dec);
        pred_Z += state.model_Z[ctx].decode(dec);

        pred_G += state.model_G.decode(dec);

        int32_t dR = state.model_R.decode(dec);
        int32_t dB = state.model_B.decode(dec);

        pred_R = pred_G + dR;
        pred_B = pred_G + dB;

        PointFormat2& p = out_points[base_idx + i];

        p.X = pred_X;
        p.Y = pred_Y;
        p.Z = pred_Z;
        p.Green = pred_G;
        p.Red = pred_R;
        p.Blue = pred_B;
    }
}



int decompress(const std::vector<uint8_t>& raw_file_data,
    uint32_t num_chunks,
    const std::vector<uint64_t>& chunk_offsets,
    uint64_t actual_total_points)
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
    int threads_per_block = 32;
    int blocks = (num_chunks + threads_per_block - 1) / threads_per_block;

    // Launch Kernel
    std::cout << "Launching Kernel: " << blocks << " blocks, "
        << threads_per_block << " threads/block..." << std::endl;
    laszip_format2_kernel << <blocks, threads_per_block >> > (
        d_compressed, d_chunk_offsets, d_out_points,
        points_per_chunk, num_chunks
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