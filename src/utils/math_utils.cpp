#include "math_utils.h"
#include <algorithm>
#include <numeric>
#include <immintrin.h>

namespace hpie {

float MathUtils::DotProduct(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return 0.0f;
    }
    
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
}

std::vector<float> MathUtils::VectorAdd(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return {};
    }
    
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    
    return result;
}

std::vector<float> MathUtils::VectorSubtract(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return {};
    }
    
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    
    return result;
}

std::vector<float> MathUtils::VectorMultiply(const std::vector<float>& a, float scalar) {
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * scalar;
    }
    
    return result;
}

float MathUtils::VectorNorm(const std::vector<float>& vec) {
    float sum_squares = 0.0f;
    for (float val : vec) {
        sum_squares += val * val;
    }
    
    return std::sqrt(sum_squares);
}

void MathUtils::NormalizeVector(std::vector<float>& vec) {
    float norm = VectorNorm(vec);
    if (norm > 0.0f) {
        for (auto& val : vec) {
            val /= norm;
        }
    }
}

std::vector<float> MathUtils::MatrixVectorMultiply(
    const std::vector<float>& matrix, 
    const std::vector<float>& vector, 
    size_t rows, size_t cols) {
    
    if (matrix.size() != rows * cols || vector.size() != cols) {
        return {};
    }
    
    std::vector<float> result(rows);
    
    for (size_t i = 0; i < rows; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            sum += matrix[i * cols + j] * vector[j];
        }
        result[i] = sum;
    }
    
    return result;
}

std::vector<float> MathUtils::MatrixMultiply(
    const std::vector<float>& a, 
    const std::vector<float>& b,
    size_t m, size_t n, size_t k) {
    
    if (a.size() != m * n || b.size() != n * k) {
        return {};
    }
    
    std::vector<float> result(m * k, 0.0f);
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            float sum = 0.0f;
            for (size_t l = 0; l < n; ++l) {
                sum += a[i * n + l] * b[l * k + j];
            }
            result[i * k + j] = sum;
        }
    }
    
    return result;
}

float MathUtils::Sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float MathUtils::Tanh(float x) {
    return std::tanh(x);
}

float MathUtils::ReLU(float x) {
    return std::max(0.0f, x);
}

float MathUtils::GELU(float x) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    return 0.5f * x * (1.0f + std::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
}

float MathUtils::Swish(float x) {
    return x * Sigmoid(x);
}

float MathUtils::Mean(const std::vector<float>& data) {
    if (data.empty()) {
        return 0.0f;
    }
    
    return std::accumulate(data.begin(), data.end(), 0.0f) / data.size();
}

float MathUtils::Variance(const std::vector<float>& data) {
    if (data.size() <= 1) {
        return 0.0f;
    }
    
    float mean_val = Mean(data);
    float sum_squares = 0.0f;
    
    for (float val : data) {
        float diff = val - mean_val;
        sum_squares += diff * diff;
    }
    
    return sum_squares / (data.size() - 1);
}

float MathUtils::StandardDeviation(const std::vector<float>& data) {
    return std::sqrt(Variance(data));
}

std::vector<float> MathUtils::Softmax(const std::vector<float>& logits) {
    if (logits.empty()) {
        return {};
    }
    
    std::vector<float> probs(logits.size());
    
    // Find maximum for numerical stability
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    // Compute exponentials and sum
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    
    // Normalize
    if (sum > 0.0f) {
        for (auto& prob : probs) {
            prob /= sum;
        }
    }
    
    return probs;
}

bool MathUtils::IsFinite(float value) {
    return std::isfinite(value);
}

bool MathUtils::IsFinite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float val) {
        return std::isfinite(val);
    });
}

void MathUtils::ClampVector(std::vector<float>& vec, float min_val, float max_val) {
    for (auto& val : vec) {
        val = std::max(min_val, std::min(max_val, val));
    }
}

// SIMD-optimized operations
void MathUtils::SIMDDotProduct(const float* a, const float* b, size_t size, float* result) {
    *result = 0.0f;
    
    // Use SIMD for vectorized operations
    size_t simd_size = size & ~7; // Process 8 elements at a time
    
    __m256 sum_vec = _mm256_setzero_ps();
    
    for (size_t i = 0; i < simd_size; i += 8) {
        __m256 vec_a = _mm256_loadu_ps(&a[i]);
        __m256 vec_b = _mm256_loadu_ps(&b[i]);
        sum_vec = _mm256_fmadd_ps(vec_a, vec_b, sum_vec);
    }
    
    // Horizontal sum
    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum_vec, 0), _mm256_extractf128_ps(sum_vec, 1));
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_movehdup_ps(sum128));
    *result = _mm_cvtss_f32(sum128);
    
    // Handle remaining elements
    for (size_t i = simd_size; i < size; ++i) {
        *result += a[i] * b[i];
    }
}

void MathUtils::SIMDVectorAdd(const float* a, const float* b, size_t size, float* result) {
    size_t simd_size = size & ~7; // Process 8 elements at a time
    
    for (size_t i = 0; i < simd_size; i += 8) {
        __m256 vec_a = _mm256_loadu_ps(&a[i]);
        __m256 vec_b = _mm256_loadu_ps(&b[i]);
        __m256 sum = _mm256_add_ps(vec_a, vec_b);
        _mm256_storeu_ps(&result[i], sum);
    }
    
    // Handle remaining elements
    for (size_t i = simd_size; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void MathUtils::SIMDVectorMultiply(const float* vec, float scalar, size_t size, float* result) {
    __m256 scalar_vec = _mm256_set1_ps(scalar);
    size_t simd_size = size & ~7; // Process 8 elements at a time
    
    for (size_t i = 0; i < simd_size; i += 8) {
        __m256 vec_data = _mm256_loadu_ps(&vec[i]);
        __m256 product = _mm256_mul_ps(vec_data, scalar_vec);
        _mm256_storeu_ps(&result[i], product);
    }
    
    // Handle remaining elements
    for (size_t i = simd_size; i < size; ++i) {
        result[i] = vec[i] * scalar;
    }
}

} // namespace hpie
