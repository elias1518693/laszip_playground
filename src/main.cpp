
#include <vector>
#include <unordered_map>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <print>
#include <format>
#include <algorithm>
#include "laszip/laszip_api.h"
#include <ans.cuh>
using namespace std;


vector<uint8_t> int32_to_bytes(const vector<int32_t>& input) {
    vector<uint8_t> output;
    output.reserve(input.size() * sizeof(int32_t));
    for (int32_t val : input) {
        // Little-endian byte extraction (low byte first)
        output.push_back(val & 0xFF);
        output.push_back((val >> 8) & 0xFF);
        output.push_back((val >> 16) & 0xFF);
        output.push_back((val >> 24) & 0xFF);
    }
    return output;
}

// Converts a vector of bytes back to a vector of int32_t (for decoding)
vector<int32_t> bytes_to_int32(const vector<uint8_t>& input) {
    size_t num_ints = input.size() / sizeof(int32_t);
    vector<int32_t> output;
    output.reserve(num_ints);
    for (size_t i = 0; i < num_ints; i++) {
        int32_t val =
            (static_cast<int32_t>(input[i * 4 + 0])) |
            (static_cast<int32_t>(input[i * 4 + 1]) << 8) |
            (static_cast<int32_t>(input[i * 4 + 2]) << 16) |
            (static_cast<int32_t>(input[i * 4 + 3]) << 24);
        output.push_back(val);
    }
    return output;
}

int main()
{
    string file = "./resources/pointclouds/ot_35120A4201B_1.laz";

    laszip_POINTER laszip_reader = nullptr;
    laszip_header* lazHeader = nullptr;
    laszip_point* laz_point = nullptr;

    laszip_create(&laszip_reader);

    laszip_BOOL is_compressed;
    laszip_BOOL request_reader = true;

    laszip_request_compatibility_mode(laszip_reader, request_reader);

    laszip_open_reader(laszip_reader, file.c_str(), &is_compressed);

    laszip_get_header_pointer(laszip_reader, &lazHeader);
    laszip_get_point_pointer(laszip_reader, &laz_point);

    vector<int32_t> deltaX_orig;
    vector<int32_t> deltaY_orig;
    vector<int32_t> deltaZ_orig;

    int32_t prevX = 0, prevY = 0, prevZ = 0;
    bool first = true;

    uint64_t total_points = lazHeader->number_of_point_records;
    const int pointLimit = min((int)total_points, 10000000);

    println("Reading {} points from '{}'...", pointLimit, file);

    for (int i = 0; i < pointLimit; i++) {
        laszip_read_point(laszip_reader);

        int32_t X = laz_point->X;
        int32_t Y = laz_point->Y;
        int32_t Z = laz_point->Z;

        if (!first) {
            deltaX_orig.push_back(X - prevX);
            deltaY_orig.push_back(Y - prevY);
            deltaZ_orig.push_back(Z - prevZ);
        }
        else {
            first = false;
        }

        prevX = X;
        prevY = Y;
        prevZ = Z;
    }

    laszip_close_reader(laszip_reader);
    laszip_destroy(laszip_reader);
	std::vector<uint8_t> dataX = int32_to_bytes(deltaX_orig);
    test(dataX.data(), dataX.size());
    return 0;
}
