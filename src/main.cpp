/**
 * @file laszip_predictor.cpp
 * @brief Demonstrates LASzip-style coordinate prediction for compression.
 *
 * This program reads a .laz file, applies the prediction logic described
 * in LASzip (2nd order for X/Y, 1st order for Z), and calculates
 * the corrected deltas (actual - predicted). These deltas are what
 * would be passed to an entropy coder.
 */

#include <vector>
#include <unordered_map>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <algorithm>
#include <deque>      // Ideal for fixed-size "last N" history
#include <numeric>    // For std::nth_element
#include <stdexcept>  // For error handling

 // C++20/23 features for printing.
#include <print>

// Include LASzip API header
#include "laszip/laszip_api.h"
// Include the header for the compression test function
#include "ans.cuh"

using namespace std;

/**
 * @brief Converts a vector of int32_t to a little-endian byte vector.
 * @param input Vector of 32-bit integers.
 * @return Vector of 8-bit unsigned integers.
 */
vector<uint8_t> int32_to_bytes(const vector<int32_t>& input) {
    vector<uint8_t> output;
    output.reserve(input.size() * sizeof(int32_t));
    for (int32_t val : input) {
        // Little-endian byte extraction (low byte first)
        //output.push_back(val & 0xFF);
        //output.push_back((val >> 8) & 0xFF);
        //output.push_back((val >> 16) & 0xFF);
        output.push_back((val >> 24) & 0xFF);
    }
    return output;
}

vector<vector<uint8_t>> int32_to_bytes_split(const vector<int32_t>& input) {
    vector<vector<uint8_t>> output;
    output.reserve(input.size() * sizeof(int32_t));
    output.push_back(vector<uint8_t>());
    output.push_back(vector<uint8_t>());
    output.push_back(vector<uint8_t>());
    output.push_back(vector<uint8_t>());
    for (int32_t val : input) {
        // Little-endian byte extraction (low byte first)
		
        output[0].push_back(val & 0xFF);
        output[1].push_back((val >> 8) & 0xFF);
        output[2].push_back((val >> 16) & 0xFF);
        output[3].push_back((val >> 24) & 0xFF);
    }
    return output;
}



/**
 * @brief Converts a little-endian byte vector back to a vector of int32_t.
 * @param input Vector of 8-bit unsigned integers.
 * @return Vector of 32-bit integers.
 */
vector<int32_t> bytes_to_int32(const vector<uint8_t>& input) {
    if (input.size() % sizeof(int32_t) != 0) {
        std::println(stderr, "Error: Input byte vector size is not a multiple of 4.");
        return {};
    }
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

/**
 * @brief Finds the median of a deque of int32_t.
 * Uses std::nth_element for O(N) average complexity.
 * @param history A deque (non-const, as nth_element is in-place on a copy).
 * @return The median value, or 0 if the deque is empty.
 */
int32_t get_median(const std::deque<int32_t>& history) {
    if (history.empty()) {
        return 0; // Default prediction if no history
    }

    // Create a temporary copy to find the median without modifying the deque
    std::vector<int32_t> temp(history.begin(), history.end());

    // Find the (lower) middle element index
    size_t n = (temp.size() - 1) / 2;

    // Partially sort the vector so the element at index 'n' is
    // the same as it would be if the whole vector were sorted.
    std::nth_element(temp.begin(), temp.begin() + n, temp.end());

    return temp[n];
}

/**
 * @brief Updates a fixed-size history deque.
 * Removes the oldest item if the size limit (5) is reached,
 * then adds the new value.
 * @param history The deque to update (by reference).
 * @param new_val The new value to add.
 */
static void update_history(std::deque<int32_t>& history, int32_t new_val) {
    if (history.size() >= 5) {
        history.pop_front(); // Remove the oldest
    }
    history.push_back(new_val); // Add the newest
}

template <typename IntType>
void print_histogram(const std::vector<IntType>& data, const std::string& title, int num_bins = 20)
{
    static_assert(std::is_integral_v<IntType>, "print_histogram requires an integer type");

    if (data.empty()) {
        std::println("Histogram: '{}' (No data)", title);
        return;
    }

    // 1. Find min and max
    IntType min_val = data[0];
    IntType max_val = data[0];
    for (auto val : data) {
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    std::println("\n--- Histogram: {} ---", title);
    std::println("Total points: {}", data.size());
    std::println("Min: {}, Max: {}", static_cast<long long>(min_val), static_cast<long long>(max_val));

    if (min_val == max_val) {
        std::println("[{}] count: {}", static_cast<long long>(min_val), data.size());
        return;
    }

    // 2. Determine bin size
    double range = static_cast<double>(max_val) - static_cast<double>(min_val);
    double bin_size = range / num_bins;
    if (bin_size < 1.0) {
        bin_size = 1.0;
        num_bins = static_cast<int>(range) + 1;
        if (num_bins > 100) {
            num_bins = 100;
            bin_size = range / num_bins;
        }
    }

    // 3. Populate bins
    std::vector<long long> bin_counts(num_bins, 0);
    long long max_count = 0;

    for (auto val : data) {
        int bin_index = static_cast<int>((static_cast<double>(val) - static_cast<double>(min_val)) / bin_size);
        if (bin_index >= num_bins) bin_index = num_bins - 1;
        if (bin_index < 0) bin_index = 0;
        bin_counts[bin_index]++;
        if (bin_counts[bin_index] > max_count)
            max_count = bin_counts[bin_index];
    }

    // 4. Print histogram
    const int max_bar_width = 50;
    std::println("{:<25} | {:<10} | {}", "Bin Range", "Count", "Bar");
    std::println("{}", std::string(25 + 13 + max_bar_width, '-'));

    for (int i = 0; i < num_bins; ++i) {
        long long bin_start = static_cast<long long>(min_val) + static_cast<long long>(i * bin_size);
        long long bin_end = static_cast<long long>(min_val) + static_cast<long long>((i + 1) * bin_size);

        std::string range_str;
        if (i == num_bins - 1)
            range_str = std::format("[{}, {}]", bin_start, static_cast<long long>(max_val));
        else
            range_str = std::format("[{}, {})", bin_start, bin_end);

        long long count = bin_counts[i];
        int bar_width = (max_count > 0)
            ? static_cast<int>((static_cast<double>(count) / max_count) * max_bar_width)
            : 0;

        std::string bar(bar_width, '#');
        std::println("{:<25} | {:<10} | {}", range_str, count, bar);
    }

    std::println("---------------------------------");
}

struct k_code {
    // Stores the "Corrector" (upper bits) to be arithmetically coded.
    // We have 33 vectors, one for each k (0-32).
    std::vector<std::vector<uint8_t>> upper_byte;

    // Stores the k-value for each delta
    std::vector<uint8_t> k;

    // Stores the "lower bits" (k-8 bits) to be stored raw.
    std::vector<uint32_t> raw_bits;
};

/**
 * Maps the signed delta 'c' to a positive value for the encoder,
 * based on the 'k' found, as per LAZ spec 10.5.5 .
 */

int main()
{
    string file = "./resources/pointclouds/ot_35120A4201B_1.laz";

    laszip_POINTER laszip_reader = nullptr;
    laszip_header* lazHeader = nullptr;
    laszip_point* laz_point = nullptr;

    laszip_create(&laszip_reader);
    if (!laszip_reader) {
        std::println(stderr, "Failed to create laszip reader.");
        return 1;
    }

    laszip_BOOL is_compressed;
    laszip_BOOL request_reader = true;

    laszip_request_compatibility_mode(laszip_reader, request_reader);

    if (laszip_open_reader(laszip_reader, file.c_str(), &is_compressed) != 0) {
        std::println(stderr, "Failed to open {}", file);
        laszip_destroy(laszip_reader);
        return 1;
    }

    laszip_get_header_pointer(laszip_reader, &lazHeader);
    if (!lazHeader) {
        std::println(stderr, "Failed to get LAS header.");
        laszip_close_reader(laszip_reader);
        laszip_destroy(laszip_reader);
        return 1;
    }
    laszip_get_point_pointer(laszip_reader, &laz_point);
    if (!laz_point) {
        std::println(stderr, "Failed to get LAS point pointer.");
        laszip_close_reader(laszip_reader);
        laszip_destroy(laszip_reader);
        return 1;
    }




    // These will store the final *corrected* deltas to be compressed
    vector<int32_t> corrected_deltaX;
    vector<int32_t> corrected_deltaY;
    vector<int32_t> corrected_deltaZ;
    vector<int32_t> simple_deltaX;
    vector<int32_t> simple_deltaY;
    vector<int32_t> simple_deltaZ;
    // State for prediction
    int32_t prevX = 0, prevY = 0, prevZ = 0;
    bool first = true;

    // --- State for Predictors ---

    // LAS spec 1.4: return_number and number_of_returns are 4 bits (0-15)
    // We will use 5 bits (0-31) just to be safe.
    const uint8_t MAX_RETURNS = 32;

    // State for X/Y prediction (2nd order)
    // Key: return map 'm' = (number_of_returns << 5) | return_number
    // We use a vector for fast O(1) lookup instead of a map.
    const size_t MAX_RETURN_MAPS = (MAX_RETURNS << 5) | MAX_RETURNS;
    std::vector<std::deque<int32_t>> last_5_dx_by_m(MAX_RETURN_MAPS);
    std::vector<std::deque<int32_t>> last_5_dy_by_m(MAX_RETURN_MAPS);

    // State for Z prediction (1st order)
    // Key: return level 'l' = return_number
    // Value: last Z value for that 'l'
    std::vector<int32_t> last_z_by_l(MAX_RETURNS, 0);
    std::vector<bool> has_z_by_l(MAX_RETURNS, false); // Track if we have a value


    uint64_t total_points = lazHeader->number_of_point_records;
    const int pointLimit = min((int)total_points, 500000000);

    std::println("Reading {} points from '{}'...", pointLimit, file);

    for (int i = 0; i < pointLimit; i++) {
        if (laszip_read_point(laszip_reader) != 0) {
            std::println(stderr, "Warning: Error reading point {}. Stopping.", i);
            break;
        }

        int32_t X = laz_point->X;
        int32_t Y = laz_point->Y;
        int32_t Z = laz_point->Z;

        // Get return info
        // 'l' (return level) is the return_number
        uint8_t l = laz_point->return_number;
        // 'm' (return map) is the combination of number_of_returns and return_number
        // We shift by 5 bits to accommodate up to 31 returns.
        uint16_t m = (laz_point->number_of_returns << 5) | laz_point->return_number;

        // Bounds check
        if (l >= MAX_RETURNS || m >= MAX_RETURN_MAPS) {
            std::println(stderr, "Warning: Invalid return number ({}) or map ({}) at point {}. Using simple delta.", l, m, i);

            if (!first) {
                corrected_deltaX.push_back(X - prevX);
                corrected_deltaY.push_back(Y - prevY);
                corrected_deltaZ.push_back(Z - prevZ);
            }
        }
        else if (!first) {

            // --- X and Y Prediction (2nd Order) ---
            int32_t simple_dx = X - prevX;
            int32_t simple_dy = Y - prevY;
            int32_t simple_dz = Z - prevZ;

            // Add to both simple and corrected (as fallback)
            simple_deltaX.push_back(simple_dx);
            simple_deltaY.push_back(simple_dy);
            simple_deltaZ.push_back(simple_dz);

            // 2. Get history for this specific return map 'm'
            std::deque<int32_t>& dx_history = last_5_dx_by_m[m];
            std::deque<int32_t>& dy_history = last_5_dy_by_m[m];


            if (dx_history.empty()) {
                update_history(dx_history, X);
                update_history(dy_history, Y);
                int32_t predicted_Z = has_z_by_l[l] ? last_z_by_l[l] : prevZ;
                corrected_deltaZ.push_back(Z - predicted_Z);
                continue;
            }
            else if (dx_history.size() < 5) {
                update_history(dx_history, X);
                update_history(dy_history, Y);
                continue;
                corrected_deltaX.push_back(X - simple_dx);
                corrected_deltaY.push_back(Y - simple_dy);
                int32_t predicted_Z = has_z_by_l[l] ? last_z_by_l[l] : prevZ;
                corrected_deltaZ.push_back(Z - predicted_Z);
                continue;
            }

            // 3. Predict deltas based on median of history
            int32_t predicted_dx = get_median(dx_history);
            int32_t predicted_dy = get_median(dy_history);

            // 4. Calculate the corrected delta (the value to be compressed)

            corrected_deltaX.push_back(X - predicted_dx);
            corrected_deltaY.push_back(Y - predicted_dy);

            // 5. Update the history for 'm' with the *actual* deltas
            update_history(dx_history, X);
            update_history(dy_history, Y);

            // --- Z Prediction (1st Order) ---
            // 1. Predict Z based on last Z for this return level 'l'
            //    Fallback to previous point's Z if no history for 'l'
            int32_t predicted_Z = has_z_by_l[l] ? last_z_by_l[l] : prevZ;

            // 2. Calculate the corrected delta (the value to be compressed)
            corrected_deltaZ.push_back(Z - predicted_Z);

        }
        // else: This is the 'first' point. We don't store a delta,
        // but we *do* update the state below.

        // --- Update State for Next Iteration ---
        first = false;
        prevX = X;
        prevY = Y;
        prevZ = Z;

        // Update the last Z for this return level 'l'
        if (l < MAX_RETURNS) {
            last_z_by_l[l] = Z;
            has_z_by_l[l] = true;
        }
    }

    laszip_close_reader(laszip_reader);

    // Always destroy the reader
    if (laszip_reader) {
        laszip_destroy(laszip_reader);
    }

    // --- Print Histograms ---
    std::vector<vector<uint8_t>> dataX = int32_to_bytes_split(corrected_deltaX);
    // --- Print Histograms ---
    //print_histogram(dataX[0], "Simple Delta X (0-7)");
    //print_histogram(dataX[1], "Simple Delta X (8-15)");
    //print_histogram(dataX[2], "Simple Delta X (16-23)");
    //print_histogram(dataX[3], "Simple Delta X (24-31)");
    // Now, compress the *corrected* deltas

    if (!corrected_deltaX.empty()) {
        std::println("Compressing X deltas ({} bytes)...", dataX.size());
        test(corrected_deltaX, corrected_deltaX.size()); // Assuming test() is your compression function
    }
}