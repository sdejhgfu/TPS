#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <functional>

namespace hpie {

// Quantization types
enum class QuantizationType {
    INT8,
    INT4,
    FP16,
    BF16,
    DYNAMIC
};

// Quantization parameters
struct QuantizationParams {
    QuantizationType type;
    float scale;
    int32_t zero_point;
    float min_val;
    float max_val;
    bool symmetric;
};

// Quantized tensor structure
template<typename T>
struct QuantizedTensor {
    std::vector<T> data;
    std::vector<QuantizationParams> params;
    std::vector<size_t> shape;
    size_t total_elements;
    
    QuantizedTensor() : total_elements(0) {}
    
    explicit QuantizedTensor(const std::vector<size_t>& shape)
        : shape(shape), total_elements(1) {
        for (size_t dim : shape) {
            total_elements *= dim;
        }
        data.resize(total_elements);
    }
    
    T* GetData() { return data.data(); }
    const T* GetData() const { return data.data(); }
    
    size_t GetSize() const { return total_elements; }
    const std::vector<size_t>& GetShape() const { return shape; }
};

// Base quantizer interface
class Quantizer {
public:
    virtual ~Quantizer() = default;
    
    // Quantization methods
    virtual QuantizedTensor<int8_t> QuantizeToInt8(const std::vector<float>& data) = 0;
    virtual QuantizedTensor<int4_t> QuantizeToInt4(const std::vector<float>& data) = 0;
    
    // Dequantization methods
    virtual std::vector<float> DequantizeFromInt8(const QuantizedTensor<int8_t>& tensor) = 0;
    virtual std::vector<float> DequantizeFromInt4(const QuantizedTensor<int4_t>& tensor) = 0;
    
    // Dynamic quantization
    virtual QuantizationParams ComputeQuantizationParams(
        const std::vector<float>& data, QuantizationType type) = 0;
    
    // Matrix multiplication with quantized weights
    virtual std::vector<float> QuantizedMatMul(
        const std::vector<float>& input,
        const QuantizedTensor<int8_t>& weights,
        size_t m, size_t n, size_t k) = 0;
    
    // Calibration for optimal quantization
    virtual void Calibrate(const std::vector<std::vector<float>>& calibration_data) = 0;
    
    // Performance optimization
    virtual void OptimizeForCPU() = 0;
    virtual void EnableSIMD() = 0;
};

// INT8 quantizer implementation
class Int8Quantizer : public Quantizer {
public:
    Int8Quantizer() = default;
    ~Int8Quantizer() override = default;
    
    QuantizedTensor<int8_t> QuantizeToInt8(const std::vector<float>& data) override;
    QuantizedTensor<int4_t> QuantizeToInt4(const std::vector<float>& data) override;
    
    std::vector<float> DequantizeFromInt8(const QuantizedTensor<int8_t>& tensor) override;
    std::vector<float> DequantizeFromInt4(const QuantizedTensor<int4_t>& tensor) override;
    
    QuantizationParams ComputeQuantizationParams(
        const std::vector<float>& data, QuantizationType type) override;
    
    std::vector<float> QuantizedMatMul(
        const std::vector<float>& input,
        const QuantizedTensor<int8_t>& weights,
        size_t m, size_t n, size_t k) override;
    
    void Calibrate(const std::vector<std::vector<float>>& calibration_data) override;
    void OptimizeForCPU() override;
    void EnableSIMD() override;

private:
    QuantizationParams params_;
    bool simd_enabled_ = false;
    bool cpu_optimized_ = false;
    
    // SIMD-optimized quantization
    void QuantizeSIMD(const float* input, int8_t* output, size_t count);
    void DequantizeSIMD(const int8_t* input, float* output, size_t count);
    
    // Calibration data
    std::vector<float> min_values_;
    std::vector<float> max_values_;
    std::vector<float> scales_;
};

// INT4 quantizer implementation
class Int4Quantizer : public Quantizer {
public:
    Int4Quantizer() = default;
    ~Int4Quantizer() override = default;
    
    QuantizedTensor<int8_t> QuantizeToInt8(const std::vector<float>& data) override;
    QuantizedTensor<int4_t> QuantizeToInt4(const std::vector<float>& data) override;
    
    std::vector<float> DequantizeFromInt8(const QuantizedTensor<int8_t>& tensor) override;
    std::vector<float> DequantizeFromInt4(const QuantizedTensor<int4_t>& tensor) override;
    
    QuantizationParams ComputeQuantizationParams(
        const std::vector<float>& data, QuantizationType type) override;
    
    std::vector<float> QuantizedMatMul(
        const std::vector<float>& input,
        const QuantizedTensor<int8_t>& weights,
        size_t m, size_t n, size_t k) override;
    
    void Calibrate(const std::vector<std::vector<float>>& calibration_data) override;
    void OptimizeForCPU() override;
    void EnableSIMD() override;

private:
    QuantizationParams params_;
    bool simd_enabled_ = false;
    bool cpu_optimized_ = false;
    
    // Pack/unpack INT4 values
    void PackInt4(const std::vector<int8_t>& input, std::vector<uint8_t>& output);
    void UnpackInt4(const std::vector<uint8_t>& input, std::vector<int8_t>& output);
    
    // INT4-specific optimizations
    void QuantizeInt4SIMD(const float* input, uint8_t* output, size_t count);
    void DequantizeInt4SIMD(const uint8_t* input, float* output, size_t count);
};

// Dynamic quantizer that adapts based on data distribution
class DynamicQuantizer : public Quantizer {
public:
    DynamicQuantizer() = default;
    ~DynamicQuantizer() override = default;
    
    QuantizedTensor<int8_t> QuantizeToInt8(const std::vector<float>& data) override;
    QuantizedTensor<int4_t> QuantizeToInt4(const std::vector<float>& data) override;
    
    std::vector<float> DequantizeFromInt8(const QuantizedTensor<int8_t>& tensor) override;
    std::vector<float> DequantizeFromInt4(const QuantizedTensor<int4_t>& tensor) override;
    
    QuantizationParams ComputeQuantizationParams(
        const std::vector<float>& data, QuantizationType type) override;
    
    std::vector<float> QuantizedMatMul(
        const std::vector<float>& input,
        const QuantizedTensor<int8_t>& weights,
        size_t m, size_t n, size_t k) override;
    
    void Calibrate(const std::vector<std::vector<float>>& calibration_data) override;
    void OptimizeForCPU() override;
    void EnableSIMD() override;

private:
    std::unique_ptr<Int8Quantizer> int8_quantizer_;
    std::unique_ptr<Int4Quantizer> int4_quantizer_;
    
    // Dynamic selection logic
    QuantizationType SelectOptimalType(const std::vector<float>& data);
    float CalculateQuantizationError(
        const std::vector<float>& original,
        const std::vector<float>& quantized);
};

// Quantization utilities
class QuantizationUtils {
public:
    // Statistical analysis
    static float CalculateRMS(const std::vector<float>& data);
    static float CalculateKLDivergence(
        const std::vector<float>& original,
        const std::vector<float>& quantized);
    
    // Optimal quantization parameter computation
    static QuantizationParams ComputeOptimalParams(
        const std::vector<float>& data,
        QuantizationType type,
        float target_error = 0.01f);
    
    // Weight-aware quantization
    static std::vector<QuantizationParams> ComputePerChannelParams(
        const std::vector<float>& weights,
        const std::vector<size_t>& shape,
        QuantizationType type);
    
    // Quantization error analysis
    static float CalculateQuantizationError(
        const std::vector<float>& original,
        const std::vector<float>& quantized,
        const std::string& metric = "mse");
    
    // Memory usage optimization
    static size_t CalculateMemoryReduction(
        const std::vector<float>& original,
        QuantizationType target_type);
    
    // Performance profiling
    static void ProfileQuantizationPerformance(
        const std::vector<float>& data,
        QuantizationType type,
        size_t iterations = 1000);
};

} // namespace hpie

