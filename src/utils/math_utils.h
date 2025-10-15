#pragma once

#include <vector>
#include <cmath>

namespace hpie {

class MathUtils {
public:
    // Vector operations
    static float DotProduct(const std::vector<float>& a, const std::vector<float>& b);
    static std::vector<float> VectorAdd(const std::vector<float>& a, const std::vector<float>& b);
    static std::vector<float> VectorSubtract(const std::vector<float>& a, const std::vector<float>& b);
    static std::vector<float> VectorMultiply(const std::vector<float>& a, float scalar);
    static float VectorNorm(const std::vector<float>& vec);
    static void NormalizeVector(std::vector<float>& vec);
    
    // Matrix operations
    static std::vector<float> MatrixVectorMultiply(
        const std::vector<float>& matrix, 
        const std::vector<float>& vector, 
        size_t rows, size_t cols);
    
    static std::vector<float> MatrixMultiply(
        const std::vector<float>& a, 
        const std::vector<float>& b,
        size_t m, size_t n, size_t k);
    
    // Activation functions
    static float Sigmoid(float x);
    static float Tanh(float x);
    static float ReLU(float x);
    static float GELU(float x);
    static float Swish(float x);
    
    // Statistical functions
    static float Mean(const std::vector<float>& data);
    static float Variance(const std::vector<float>& data);
    static float StandardDeviation(const std::vector<float>& data);
    static std::vector<float> Softmax(const std::vector<float>& logits);
    
    // Numerical stability
    static bool IsFinite(float value);
    static bool IsFinite(const std::vector<float>& values);
    static void ClampVector(std::vector<float>& vec, float min_val, float max_val);
    
    // SIMD-optimized operations (when available)
    static void SIMDDotProduct(const float* a, const float* b, size_t size, float* result);
    static void SIMDVectorAdd(const float* a, const float* b, size_t size, float* result);
    static void SIMDVectorMultiply(const float* vec, float scalar, size_t size, float* result);
};

} // namespace hpie
