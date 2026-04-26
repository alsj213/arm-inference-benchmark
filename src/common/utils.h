#ifndef BENCHMARK_COMMON_UTILS_H_
#define BENCHMARK_COMMON_UTILS_H_

#include <string>
#include <vector>
#include <chrono>
#include <cstdint>

namespace utils {

// High-resolution timer
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    uint64_t elapsed_us() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// Get current memory usage (RSS) in KB
size_t get_memory_usage_kb();

// Generate random input data
void fill_random_float(float* data, size_t size, float min = -1.0f, float max = 1.0f);
void fill_random_float(float* data, size_t size, unsigned int seed);

// Read file into buffer
std::vector<uint8_t> read_file(const std::string& path);

// Calculate statistics
struct Stats {
    double min_ms;
    double max_ms;
    double mean_ms;
    double p50_ms;
    double p90_ms;
    double p95_ms;
    double p99_ms;
    double std_dev;
};

Stats calculate_stats(const std::vector<double>& times_ms);

// Print statistics
void print_stats(const Stats& stats);

} // namespace utils

#endif // BENCHMARK_COMMON_UTILS_H_
