
#pragma once
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

int decompress(const std::vector<uint8_t>& raw_file_data,
    uint32_t num_chunks,
    const std::vector<uint64_t>& chunk_offsets,
    uint64_t actual_total_points,
    std::vector<PointFormat2>& points);

