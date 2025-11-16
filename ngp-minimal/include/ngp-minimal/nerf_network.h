/*
 * ngp-minimal: NeRF Network
 * Simplified NeRF network wrapper for tiny-cuda-nn
 */

#pragma once

#include <ngp-minimal/common.h>
#include <ngp-minimal/common_device.cuh>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>

#include <json/json.hpp>
#include <memory>

namespace ngp {

template <typename T>
class NerfNetwork : public tcnn::Network<float, T> {
public:
    using json = nlohmann::json;

    NerfNetwork(
        uint32_t n_pos_dims,
        uint32_t n_dir_dims,
        uint32_t n_extra_dims,
        uint32_t dir_offset,
        const json& pos_encoding,
        const json& dir_encoding,
        const json& density_network,
        const json& rgb_network
    );

    virtual ~NerfNetwork() {}

    void inference_mixed_precision_impl(
        cudaStream_t stream,
        const tcnn::GPUMatrixDynamic<float>& input,
        tcnn::GPUMatrixDynamic<T>& output,
        bool use_inference_params = true
    ) override;

    std::unique_ptr<tcnn::Context> forward_impl(
        cudaStream_t stream,
        const tcnn::GPUMatrixDynamic<float>& input,
        tcnn::GPUMatrixDynamic<T>* output = nullptr,
        bool use_inference_params = false,
        bool prepare_input_gradients = false
    ) override;

    void backward_impl(
        cudaStream_t stream,
        const tcnn::Context& ctx,
        const tcnn::GPUMatrixDynamic<float>& input,
        const tcnn::GPUMatrixDynamic<T>& output,
        const tcnn::GPUMatrixDynamic<T>& dL_doutput,
        tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
        bool use_inference_params = false,
        tcnn::GradientMode param_gradients_mode = tcnn::GradientMode::Overwrite
    ) override;

    uint32_t input_width() const override {
        return m_dir_offset + m_n_dir_dims + m_n_extra_dims;
    }

    uint32_t padded_output_width() const override {
        return 4; // RGB + density
    }

    uint32_t output_width() const override {
        return 4; // RGB + density
    }

    size_t n_params() const override {
        return m_pos_encoding->n_params() +
               m_density_network->n_params() +
               m_dir_encoding->n_params() +
               m_rgb_network->n_params();
    }

    // Required pure virtual function implementations
    uint32_t width(uint32_t layer) const override {
        return 64; // Simplified - return fixed width
    }

    uint32_t num_forward_activations() const override {
        return 1; // Simplified
    }

    std::pair<const T*, tcnn::MatrixLayout> forward_activations(const tcnn::Context& ctx, uint32_t layer) const override {
        return {nullptr, tcnn::MatrixLayout::ColumnMajor}; // Simplified
    }

    uint32_t required_input_alignment() const override {
        return 1; // Minimum alignment
    }

    void set_params_impl(T* params, T* inference_params, T* backward_params) override {
        // Simplified - let trainer handle this
    }

    void initialize_params(tcnn::pcg32& rnd, float* params_full_precision, float scale = 1) override {
        // Simplified - let trainer handle this
    }

    std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
        return {}; // Simplified
    }

    nlohmann::json hyperparams() const override {
        return {{"otype", "NerfNetwork"}};
    }

    uint32_t padded_density_output_width() const {
        return m_density_network->padded_output_width();
    }

    std::shared_ptr<tcnn::NetworkWithInputEncoding<T>> density_network_with_encoding() const {
        return m_density_model;
    }

private:
    struct ForwardContext : public tcnn::Context {
        tcnn::GPUMatrixDynamic<T> density_network_input;
        tcnn::GPUMatrixDynamic<T> density_network_output;
        tcnn::GPUMatrixDynamic<T> rgb_network_input;
        tcnn::GPUMatrixDynamic<T> rgb_network_output;

        std::unique_ptr<tcnn::Context> pos_encoding_ctx;
        std::unique_ptr<tcnn::Context> density_network_ctx;
        std::unique_ptr<tcnn::Context> dir_encoding_ctx;
        std::unique_ptr<tcnn::Context> rgb_network_ctx;
    };

    uint32_t m_n_pos_dims;
    uint32_t m_n_dir_dims;
    uint32_t m_n_extra_dims;
    uint32_t m_dir_offset;
    uint32_t m_rgb_network_input_width;

    // Position encoding (HashGrid) + density network
    std::shared_ptr<tcnn::Encoding<T>> m_pos_encoding;
    std::shared_ptr<tcnn::Network<T>> m_density_network;
    std::shared_ptr<tcnn::NetworkWithInputEncoding<T>> m_density_model;

    // Direction encoding + RGB network
    std::shared_ptr<tcnn::Encoding<T>> m_dir_encoding;
    std::shared_ptr<tcnn::Network<T>> m_rgb_network;
};

} // namespace ngp
