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

void fill_random_float(float* data, size_t size, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
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

// Remove outliers from data (remove first and last N values)
std::vector<double> remove_outliers(const std::vector<double>& data, int remove_count) {
    if (data.size() <= static_cast<size_t>(remove_count * 2)) {
        return data;
    }

    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    // Remove first and last remove_count values
    return std::vector<double>(
        sorted_data.begin() + remove_count,
        sorted_data.end() - remove_count);
}

// Calculate confidence interval using t-distribution
ConfidenceInterval calculate_confidence_interval(
    const std::vector<double>& data,
    double confidence_level) {
    ConfidenceInterval result;

    if (data.empty()) {
        result.lower = 0.0;
        result.upper = 0.0;
        result.mean = 0.0;
        result.margin_of_error = 0.0;
        return result;
    }

    // Calculate mean
    double sum = 0.0;
    for (double val : data) {
        sum += val;
    }
    result.mean = sum / data.size();

    // Calculate standard deviation
    double variance = 0.0;
    for (double val : data) {
        variance += (val - result.mean) * (val - result.mean);
    }
    variance /= (data.size() - 1);
    double std_dev = std::sqrt(variance);

    // Calculate standard error
    double standard_error = std_dev / std::sqrt(data.size());

    // T-distribution critical values (simplified for common confidence levels)
    // For large samples (>30), t-distribution approaches normal distribution
    double t_critical;
    if (confidence_level >= 0.99) {
        t_critical = 2.576;  // For 99% CI (normal approximation)
    } else if (confidence_level >= 0.95) {
        t_critical = 1.96;   // For 95% CI (normal approximation)
    } else if (confidence_level >= 0.90) {
        t_critical = 1.645;  // For 90% CI (normal approximation)
    } else {
        t_critical = 1.0;    // Default fallback
    }

    result.margin_of_error = t_critical * standard_error;
    result.lower = result.mean - result.margin_of_error;
    result.upper = result.mean + result.margin_of_error;

    return result;
}

} // namespace utils
