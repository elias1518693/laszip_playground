#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <print>
#include <format>
#include <algorithm>
#include "rans_byte.h"
#include "laszip/laszip_api.h"
// -----------

#include <cassert>
using namespace std;
// --- SymbolStats Structure ---
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

            // find best symbol to steal frequency from (try to steal from low-freq ones)
            uint32_t best_freq = ~0u;
            int best_steal = -1;
            for (int j = 0; j < 256; j++) {
                uint32_t freq = cum_freqs[j + 1] - cum_freqs[j];
                if (freq > 1 && freq < best_freq) {
                    best_freq = freq;
                    best_steal = j;
                    if (best_freq == 2) break;
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

// Global constant for rANS scale
constexpr uint32_t RANS_SCALE_BITS = 12; // A common, safe value
constexpr uint32_t RANS_TOTAL_FREQ = (1u << RANS_SCALE_BITS);

// --- Conversion Helpers ---

// Converts the vector of int32_t deltas into a vector of bytes (for encoding)
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

// 1. Calculates symbol statistics (freqs)
// 2. Normalizes freqs to the RANS_TOTAL_FREQ
// 3. Initializes the RansEncSymbol array
SymbolStats buildEncStats(const vector<int32_t>& deltas, RansEncSymbol esyms[256]) {
    vector<uint8_t> byte_stream = int32_to_bytes(deltas);
    SymbolStats stats;
    stats.count_freqs(byte_stream.data(), byte_stream.size());
    stats.normalize_freqs(RANS_TOTAL_FREQ);

    for (int i = 0; i < 256; i++) {
        RansEncSymbolInit(&esyms[i], stats.cum_freqs[i], stats.freqs[i], RANS_SCALE_BITS);
    }
    return stats;
}

// Initializes the RansDecSymbol array from the SymbolStats
void buildDecStats(const SymbolStats& stats, RansDecSymbol dsyms[256]) {
    for (int i = 0; i < 256; i++) {
        RansDecSymbolInit(&dsyms[i], stats.cum_freqs[i], stats.freqs[i]);
    }
}

// Encodes the byte stream using the prepared RansEncSymbol array
vector<uint8_t> ransEncode(const vector<int32_t>& deltas, const RansEncSymbol esyms[256]) {
    vector<uint8_t> byte_stream = int32_to_bytes(deltas);

    size_t max_out_size = byte_stream.size() + 16;
    vector<uint8_t> output(max_out_size);
    uint8_t* ptr = output.data() + max_out_size; // Start at the end

    RansState r;
    RansEncInit(&r);

    // rANS MUST encode in reverse: last symbol first!
    for (size_t i = byte_stream.size(); i-- > 0; ) {
        uint8_t sym = byte_stream[i];
        RansEncPutSymbol(&r, &ptr, &esyms[sym]);
    }

    RansEncFlush(&r, &ptr);

    size_t compressed_size = output.data() + max_out_size - ptr;

    std::vector<uint8_t> final_output;
    final_output.reserve(compressed_size);
    final_output.insert(final_output.begin(), ptr, ptr + compressed_size);

    return final_output;
}

// NEW: Decodes the compressed data back into a byte stream
vector<uint8_t> ransDecode(const vector<uint8_t>& encoded, size_t uncompressed_byte_size, const RansDecSymbol dsyms[256]) {
    vector<uint8_t> output(uncompressed_byte_size);

    uint8_t* ptr = (uint8_t*)encoded.data(); // Start at the beginning
    RansState r;
    RansDecInit(&r, &ptr);

    // Decoding happens forwards
    for (size_t i = 0; i < uncompressed_byte_size; i++) {
        // 1. Map the current rANS state value to a cumulative frequency range
        uint32_t val = RansDecGet(&r, RANS_SCALE_BITS);

        // 2. Find the symbol 'sym' that corresponds to this value
        // We use a linear search here, but a binary search on cum_freqs is faster
        int sym = 0;
        for (int j = 0; j < 256; j++) {
            if (val < dsyms[j].start + dsyms[j].freq) {
                sym = j;
                break;
            }
        }

        // 3. Output the symbol
        output[i] = (uint8_t)sym;

        // 4. Advance the rANS state and renormalize
        RansDecAdvanceSymbol(&r, &ptr, &dsyms[sym], RANS_SCALE_BITS);
    }

    return output;
}


// --- Main Function (Modified for Decoding and Verification) ---
int main() {
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

    // --- ENCODING ---
    RansEncSymbol esymsX[1024];
    RansEncSymbol esymsY[1024];
    RansEncSymbol esymsZ[1024];

    println("\nBuilding rANS models (scale_bits={})...", RANS_SCALE_BITS);
    auto statsX = buildEncStats(deltaX_orig, esymsX);
    auto statsY = buildEncStats(deltaY_orig, esymsY);
    auto statsZ = buildEncStats(deltaZ_orig, esymsZ);

    println("Encoding delta streams...");
    auto encX = ransEncode(deltaX_orig, esymsX);
    auto encY = ransEncode(deltaY_orig, esymsY);
    auto encZ = ransEncode(deltaZ_orig, esymsZ);

    size_t rawX = deltaX_orig.size() * sizeof(int32_t);

    // --- DECODING & VERIFICATION ---
    RansDecSymbol dsymsX[256];
    RansDecSymbol dsymsY[256];
    RansDecSymbol dsymsZ[256];

    buildDecStats(statsX, dsymsX);
    buildDecStats(statsY, dsymsY);
    buildDecStats(statsZ, dsymsZ);

    println("\nDecoding and Verifying...");

    // Decode from compressed bytes back to original byte stream size
    auto decX_bytes = ransDecode(encX, rawX, dsymsX);
    auto decY_bytes = ransDecode(encY, rawX, dsymsY); // rawX is the size of all byte streams
    auto decZ_bytes = ransDecode(encZ, rawX, dsymsZ);

    // Convert byte stream back to int32_t deltas
    auto deltaX_dec = bytes_to_int32(decX_bytes);
    auto deltaY_dec = bytes_to_int32(decY_bytes);
    auto deltaZ_dec = bytes_to_int32(decZ_bytes);

    // 1. Check if the decoded size is correct
    assert(deltaX_orig.size() == deltaX_dec.size());

    // 2. Check every single delta value
    bool x_ok = (deltaX_orig == deltaX_dec);
    bool y_ok = (deltaY_orig == deltaY_dec);
    bool z_ok = (deltaZ_orig == deltaZ_dec);
    bool all_ok = x_ok && y_ok && z_ok;

    println("Verification:");
    println("X-Axis Deltas: {}", x_ok ? "**MATCH**" : "**MISMATCH**");
    println("Y-Axis Deltas: {}", y_ok ? "**MATCH**" : "**MISMATCH**");
    println("Z-Axis Deltas: {}", z_ok ? "**MATCH**" : "**MISMATCH**");

    if (all_ok) {
        println("\n **SUCCESS!** All points were compressed and decompressed perfectly.");
    }
    else {
        println("\n **FAILURE!** There was a mismatch in the decoded data.");
    }

    // Re-print compression stats for context
    size_t rawY = deltaY_orig.size() * sizeof(int32_t);
    size_t rawZ = deltaZ_orig.size() * sizeof(int32_t);
    size_t rawTotal = rawX + rawY + rawZ;
    size_t encTotal = encX.size() + encY.size() + encZ.size();

    double ratioX = static_cast<double>(encX.size()) / rawX * 100.0;
    double ratioY = static_cast<double>(encY.size()) / rawY * 100.0;
    double ratioZ = static_cast<double>(encZ.size()) / rawZ * 100.0;
    double ratioTotal = static_cast<double>(encTotal) / rawTotal * 100.0;

    println("\n Final Compression Results ({} deltas per axis) ", deltaX_orig.size());
    println("Axis | Uncompressed | Compressed | Ratio");
    println("-----+--------------+------------+--------");
    println(" X   | {:10} B | {:10} B | {:6.2f}%", rawX, encX.size(), ratioX);
    println(" Y   | {:10} B | {:10} B | {:6.2f}%", rawY, encY.size(), ratioY);
    println(" Z   | {:10} B | {:10} B | {:6.2f}%", rawZ, encZ.size(), ratioZ);
    println("-----+--------------+------------+--------");
    println("Total| {:10} B | {:10} B | {:6.2f}%", rawTotal, encTotal, ratioTotal);

    return all_ok ? 0 : 1;
}