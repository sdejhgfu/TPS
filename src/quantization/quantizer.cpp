#include "quantizer.h"
#include "../utils/logger.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <immintrin.h>

namespace hpie {

// Int8Quantizer implementation
QuantizedTensor<int8_t> Int8Quantizer::QuantizeToInt8(const std::vector<float>& data) {
    if (data.empty()) {
        return QuantizedTensor<int8_t>();
    }
    
    QuantizedTensor<int8_t> result({data.size()});
    
    // Compute quantization parameters
    params_ = ComputeQuantizationParams(data, QuantizationType::INT8);
    
    // Quantize data
    if (simd_enabled_) {
        QuantizeSIMD(data.data(), result.GetData(), data.size());
    } else {
        for (size_t i = 0; i < data.size(); ++i) {
            float quantized = (data[i] - params_.zero_point) / params_.scale;
            result.GetData()[i] = static_cast<int8_t>(std::round(std::max(-128.0f, std::min(127.0f, quantized))));
        }
    }
    
    result.params.push_back(params_);
    return result;
}

QuantizedTensor<int4_t> Int8Quantizer::QuantizeToInt4(const std::vector<float>& data) {
    // INT8 quantizer doesn't support INT4 directly
    // Convert to INT8 first, then to INT4
    auto int8_result = QuantizeToInt8(data);
    
    QuantizedTensor<int4_t> result({data.size()});
    result.params = int8_result.params;
    
    // Convert INT8 to INT4 (pack two values per byte)
    for (size_t i = 0; i < data.size(); i += 2) {
        int8_t val1 = int8_result.GetData()[i];
        int8_t val2 = (i + 1 < data.size()) ? int8_result.GetData()[i + 1] : 0;
        
        // Clamp to 4-bit range
        val1 = std::max(-8, std::min(7, static_cast<int>(val1) / 2));
        val2 = std::max(-8, std::min(7, static_cast<int>(val2) / 2));
        
        result.GetData()[i / 2] = (val1 & 0xF) | ((val2 & 0xF) << 4);
    }
    
    return result;
}

std::vector<float> Int8Quantizer::DequantizeFromInt8(const QuantizedTensor<int8_t>& tensor) {
    if (tensor.GetData() == nullptr || tensor.params.empty()) {
        return {};
    }
    
    std::vector<float> result(tensor.GetSize());
    const QuantizationParams& params = tensor.params[0];
    
    if (simd_enabled_) {
        DequantizeSIMD(tensor.GetData(), result.data(), tensor.GetSize());
    } else {
        for (size_t i = 0; i < tensor.GetSize(); ++i) {
            result[i] = tensor.GetData()[i] * params.scale + params.zero_point;
        }
    }
    
    return result;
}

std::vector<float> Int8Quantizer::DequantizeFromInt4(const QuantizedTensor<int4_t>& tensor) {
    if (tensor.GetData() == nullptr || tensor.params.empty()) {
        return {};
    }
    
    std::vector<float> result(tensor.GetSize() * 2); // INT4 is packed
    const QuantizationParams& params = tensor.params[0];
    
    for (size_t i = 0; i < tensor.GetSize(); ++i) {
        uint8_t packed = tensor.GetData()[i];
        int8_t val1 = static_cast<int8_t>(packed & 0xF);
        int8_t val2 = static_cast<int8_t>((packed >> 4) & 0xF);
        
        // Sign extend 4-bit values
        if (val1 & 0x8) val1 |= 0xF0;
        if (val2 & 0x8) val2 |= 0xF0;
        
        result[i * 2] = val1 * params.scale + params.zero_point;
        if (i * 2 + 1 < result.size()) {
            result[i * 2 + 1] = val2 * params.scale + params.zero_point;
        }
    }
    
    return result;
}

QuantizationParams Int8Quantizer::ComputeQuantizationParams(
    const std::vector<float>& data, QuantizationType type) {
    
    QuantizationParams params;
    params.type = type;
    
    // Find min and max values
    auto minmax = std::minmax_element(data.begin(), data.end());
    params.min_val = *minmax.first;
    params.max_val = *minmax.second;
    
    if (type == QuantizationType::INT8) {
        params.symmetric = (std::abs(params.min_val) > std::abs(params.max_val)) ?
                          (params.min_val == -params.max_val) : false;
        
        if (params.symmetric) {
            float abs_max = std::max(std::abs(params.min_val), std::abs(params.max_val));
            params.scale = abs_max / 127.0f;
            params.zero_point = 0;
        } else {
            params.scale = (params.max_val - params.min_val) / 255.0f;
            params.zero_point = static_cast<int32_t>(std::round(-params.min_val / params.scale));
        }
    } else if (type == QuantizationType::INT4) {
        float abs_max = std::max(std::abs(params.min_val), std::abs(params.max_val));
        params.scale = abs_max / 7.0f; // 4-bit signed range is -8 to 7
        params.zero_point = 0;
        params.symmetric = true;
    }
    
    return params;
}

std::vector<float> Int8Quantizer::QuantizedMatMul(
    const std::vector<float>& input,
    const QuantizedTensor<int8_t>& weights,
    size_t m, size_t n, size_t k) {
    
    std::vector<float> result(m * n, 0.0f);
    const QuantizationParams& params = weights.params[0];
    
    // Optimized quantized matrix multiplication
    #pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            
            // SIMD-optimized inner loop
            if (simd_enabled_ && k >= 8) {
                __m256 sum_vec = _mm256_setzero_ps();
                
                for (size_t l = 0; l < k; l += 8) {
                    __m256 input_vec = _mm256_loadu_ps(&input[i * k + l]);
                    
                    // Load and convert INT8 weights to float
                    __m128i weights_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&weights.GetData()[j * k + l]));
                    __m256 weights_float = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(weights_vec));
                    weights_float = _mm256_mul_ps(weights_float, _mm256_set1_ps(params.scale));
                    weights_float = _mm256_add_ps(weights_float, _mm256_set1_ps(params.zero_point));
                    
                    sum_vec = _mm256_fmadd_ps(input_vec, weights_float, sum_vec);
                }
                
                // Horizontal sum
                __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum_vec, 0), _mm256_extractf128_ps(sum_vec, 1));
                sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
                sum128 = _mm_add_ss(sum128, _mm_movehdup_ps(sum128));
                sum = _mm_cvtss_f32(sum128);
                
                // Handle remaining elements
                for (size_t l = (k / 8) * 8; l < k; ++l) {
                    float weight_val = weights.GetData()[j * k + l] * params.scale + params.zero_point;
                    sum += input[i * k + l] * weight_val;
                }
            } else {
                // Standard implementation
                for (size_t l = 0; l < k; ++l) {
                    float weight_val = weights.GetData()[j * k + l] * params.scale + params.zero_point;
                    sum += input[i * k + l] * weight_val;
                }
            }
            
            result[i * n + j] = sum;
        }
    }
    
    return result;
}

void Int8Quantizer::Calibrate(const std::vector<std::vector<float>>& calibration_data) {
    if (calibration_data.empty()) {
        return;
    }
    
    min_values_.clear();
    max_values_.clear();
    scales_.clear();
    
    // Collect statistics from calibration data
    for (const auto& data : calibration_data) {
        auto minmax = std::minmax_element(data.begin(), data.end());
        min_values_.push_back(*minmax.first);
        max_values_.push_back(*minmax.second);
        
        float scale = (*minmax.second - *minmax.first) / 255.0f;
        scales_.push_back(scale);
    }
    
    // Compute optimal parameters
    float avg_min = std::accumulate(min_values_.begin(), min_values_.end(), 0.0f) / min_values_.size();
    float avg_max = std::accumulate(max_values_.begin(), max_values_.end(), 0.0f) / max_values_.size();
    float avg_scale = std::accumulate(scales_.begin(), scales_.end(), 0.0f) / scales_.size();
    
    params_.min_val = avg_min;
    params_.max_val = avg_max;
    params_.scale = avg_scale;
    params_.zero_point = static_cast<int32_t>(std::round(-avg_min / avg_scale));
    
    Logger::Info("INT8 quantization calibrated with %zu samples", calibration_data.size());
}

void Int8Quantizer::OptimizeForCPU() {
    cpu_optimized_ = true;
    // Enable CPU-specific optimizations
    Logger::Info("INT8 quantizer optimized for CPU");
}

void Int8Quantizer::EnableSIMD() {
    simd_enabled_ = true;
    Logger::Info("INT8 quantizer SIMD optimizations enabled");
}

void Int8Quantizer::QuantizeSIMD(const float* input, int8_t* output, size_t count) {
    const __m256 scale_vec = _mm256_set1_ps(params_.scale);
    const __m256 zero_point_vec = _mm256_set1_ps(static_cast<float>(params_.zero_point));
    const __m256 min_vec = _mm256_set1_ps(-128.0f);
    const __m256 max_vec = _mm256_set1_ps(127.0f);
    
    for (size_t i = 0; i < count; i += 8) {
        __m256 input_vec = _mm256_loadu_ps(&input[i]);
        __m256 quantized = _mm256_div_ps(_mm256_sub_ps(input_vec, zero_point_vec), scale_vec);
        quantized = _mm256_max_ps(_mm256_min_ps(quantized, max_vec), min_vec);
        
        // Convert to int8
        __m256i int32_vec = _mm256_cvtps_epi32(quantized);
        __m128i int16_vec = _mm256_cvtepi32_epi16(int32_vec);
        __m128i int8_vec = _mm_packss_epi16(int16_vec, int16_vec);
        
        _mm_storel_epi64(reinterpret_cast<__m128i*>(&output[i]), int8_vec);
    }
    
    // Handle remaining elements
    for (size_t i = (count / 8) * 8; i < count; ++i) {
        float quantized = (input[i] - params_.zero_point) / params_.scale;
        output[i] = static_cast<int8_t>(std::round(std::max(-128.0f, std::min(127.0f, quantized))));
    }
}

void Int8Quantizer::DequantizeSIMD(const int8_t* input, float* output, size_t count) {
    const __m256 scale_vec = _mm256_set1_ps(params_.scale);
    const __m256 zero_point_vec = _mm256_set1_ps(static_cast<float>(params_.zero_point));
    
    for (size_t i = 0; i < count; i += 8) {
        __m128i int8_vec = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&input[i]));
        __m256i int32_vec = _mm256_cvtepi8_epi32(int8_vec);
        __m256 float_vec = _mm256_cvtepi32_ps(int32_vec);
        
        __m256 result = _mm256_fmadd_ps(float_vec, scale_vec, zero_point_vec);
        _mm256_storeu_ps(&output[i], result);
    }
    
    // Handle remaining elements
    for (size_t i = (count / 8) * 8; i < count; ++i) {
        output[i] = input[i] * params_.scale + params_.zero_point;
    }
}

// Int4Quantizer implementation (similar structure but with INT4-specific optimizations)
QuantizedTensor<int8_t> Int4Quantizer::QuantizeToInt8(const std::vector<float>& data) {
    // Convert to INT4 first, then unpack to INT8
    auto int4_result = QuantizeToInt4(data);
    
    QuantizedTensor<int8_t> result({data.size()});
    result.params = int4_result.params;
    
    // Unpack INT4 to INT8
    for (size_t i = 0; i < data.size(); i += 2) {
        uint8_t packed = int4_result.GetData()[i / 2];
        result.GetData()[i] = static_cast<int8_t>(packed & 0xF);
        if (i + 1 < data.size()) {
            result.GetData()[i + 1] = static_cast<int8_t>((packed >> 4) & 0xF);
        }
    }
    
    return result;
}

QuantizedTensor<int4_t> Int4Quantizer::QuantizeToInt4(const std::vector<float>& data) {
    if (data.empty()) {
        return QuantizedTensor<int4_t>();
    }
    
    QuantizedTensor<int4_t> result({(data.size() + 1) / 2}); // Pack two values per byte
    params_ = ComputeQuantizationParams(data, QuantizationType::INT4);
    
    if (simd_enabled_) {
        QuantizeInt4SIMD(data.data(), result.GetData(), data.size());
    } else {
        for (size_t i = 0; i < data.size(); i += 2) {
            float quantized1 = data[i] / params_.scale;
            float quantized2 = (i + 1 < data.size()) ? data[i + 1] / params_.scale : 0.0f;
            
            int8_t val1 = static_cast<int8_t>(std::round(std::max(-8.0f, std::min(7.0f, quantized1))));
            int8_t val2 = static_cast<int8_t>(std::round(std::max(-8.0f, std::min(7.0f, quantized2))));
            
            result.GetData()[i / 2] = (val1 & 0xF) | ((val2 & 0xF) << 4);
        }
    }
    
    result.params.push_back(params_);
    return result;
}

// DynamicQuantizer implementation
QuantizationType DynamicQuantizer::SelectOptimalType(const std::vector<float>& data) {
    // Analyze data distribution to select optimal quantization type
    auto minmax = std::minmax_element(data.begin(), data.end());
    float range = *minmax.second - *minmax.first;
    float std_dev = 0.0f;
    
    float mean = std::accumulate(data.begin(), data.end(), 0.0f) / data.size();
    for (float val : data) {
        std_dev += (val - mean) * (val - mean);
    }
    std_dev = std::sqrt(std_dev / data.size());
    
    // Selection logic based on data characteristics
    if (range < 1.0f && std_dev < 0.1f) {
        return QuantizationType::INT4;
    } else if (range < 10.0f && std_dev < 1.0f) {
        return QuantizationType::INT8;
    } else {
        return QuantizationType::INT8; // Default fallback
    }
}

// QuantizationUtils implementation
float QuantizationUtils::CalculateRMS(const std::vector<float>& data) {
    float sum_squares = 0.0f;
    for (float val : data) {
        sum_squares += val * val;
    }
    return std::sqrt(sum_squares / data.size());
}

QuantizationParams QuantizationUtils::ComputeOptimalParams(
    const std::vector<float>& data,
    QuantizationType type,
    float target_error) {
    
    QuantizationParams params;
    params.type = type;
    
    auto minmax = std::minmax_element(data.begin(), data.end());
    params.min_val = *minmax.first;
    params.max_val = *minmax.second;
    
    // Iterative optimization to minimize quantization error
    float best_error = std::numeric_limits<float>::max();
    float best_scale = 0.0f;
    
    for (int iter = 0; iter < 100; ++iter) {
        float scale = (params.max_val - params.min_val) / (1 << (type == QuantizationType::INT8 ? 8 : 4));
        int32_t zero_point = static_cast<int32_t>(std::round(-params.min_val / scale));
        
        // Calculate quantization error
        float error = 0.0f;
        for (float val : data) {
            float quantized = std::round((val - zero_point) / scale) * scale + zero_point;
            error += (val - quantized) * (val - quantized);
        }
        error = std::sqrt(error / data.size());
        
        if (error < best_error) {
            best_error = error;
            best_scale = scale;
            params.scale = scale;
            params.zero_point = zero_point;
        }
        
        if (error <= target_error) {
            break;
        }
    }
    
    return params;
}

} // namespace hpie

