#include "utils.h"

#include <algorithm>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>

#ifdef __linux__
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace utils {

size_t get_memory_usage_kb() {
#ifdef __linux__
    FILE* file = fopen("/proc/self/statm", "r");
    if (!file) return 0;
    size_t rss;
    fscanf(file, "%*zu %zu", &rss);
    fclose(file);
    return rss * (size_t)sysconf(_SC_PAGESIZE) / 1024;
#else
    return 0;
#endif
}

void fill_random_float(float* data, size_t size, float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return {};
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return {};
    }
    return buffer;
}

Stats calculate_stats(const std::vector<double>& times_ms) {
    Stats stats;
    if (times_ms.empty()) {
        stats.min_ms = 0;
        stats.max_ms = 0;
        stats.mean_ms = 0;
        stats.p50_ms = 0;
        stats.p90_ms = 0;
        stats.p95_ms = 0;
        stats.p99_ms = 0;
        stats.std_dev = 0;
        return stats;
    }

    std::vector<double> sorted = times_ms;
    std::sort(sorted.begin(), sorted.end());

    stats.min_ms = sorted.front();
    stats.max_ms = sorted.back();

    double sum = 0.0;
    for (double t : sorted) {
        sum += t;
    }
    stats.mean_ms = sum / sorted.size();

    size_t n = sorted.size();
    stats.p50_ms = sorted[(size_t)(n * 0.5)];
    stats.p90_ms = sorted[(size_t)(n * 0.9)];
    stats.p95_ms = sorted[(size_t)(n * 0.95)];
    stats.p99_ms = sorted[(size_t)(n * 0.99)];

    double variance = 0.0;
    for (double t : sorted) {
        variance += (t - stats.mean_ms) * (t - stats.mean_ms);
    }
    variance /= n;
    stats.std_dev = std::sqrt(variance);

    return stats;
}

void print_stats(const Stats& stats) {
    printf("  Min:    %.2f ms\n", stats.min_ms);
    printf("  P50:    %.2f ms\n", stats.p50_ms);
    printf("  P90:    %.2f ms\n", stats.p90_ms);
    printf("  P99:    %.2f ms\n", stats.p99_ms);
    printf("  Max:    %.2f ms\n", stats.max_ms);
    printf("  Mean:   %.2f ms\n", stats.mean_ms);
    printf("  Std:    %.2f ms\n", stats.std_dev);
}

} // namespace utils
