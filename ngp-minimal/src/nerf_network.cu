/*
 * ngp-minimal: NeRF Network Implementation
 */

#include <ngp-minimal/nerf_network.h>
#include <tiny-cuda-nn/encodings/grid.h>

namespace ngp {

// CUDA kernels for extracting density and RGB
template <typename T>
__global__ void extract_density_kernel(
    const uint32_t n_elements,
    const uint32_t density_stride,
    const uint32_t rgbd_stride,
    const T* __restrict__ density,
    T* __restrict__ rgbd
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;

    rgbd[i * rgbd_stride] = density[i * density_stride];
}

template <typename T>
__global__ void extract_rgb_kernel(
    const uint32_t n_elements,
    const uint32_t rgb_stride,
    const uint32_t output_stride,
    const T* __restrict__ rgbd,
    T* __restrict__ rgb
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;

    const uint32_t elem_idx = i / 3;
    const uint32_t dim_idx = i - elem_idx * 3;

    rgb[elem_idx * rgb_stride + dim_idx] = rgbd[elem_idx * output_stride + dim_idx];
}

template <typename T>
__global__ void add_density_gradient_kernel(
    const uint32_t n_elements,
    const uint32_t rgbd_stride,
    const T* __restrict__ rgbd,
    const uint32_t density_stride,
    T* __restrict__ density
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;

    density[i * density_stride] += rgbd[i * rgbd_stride + 3];
}

template <typename T>
NerfNetwork<T>::NerfNetwork(
    uint32_t n_pos_dims,
    uint32_t n_dir_dims,
    uint32_t n_extra_dims,
    uint32_t dir_offset,
    const json& pos_encoding,
    const json& dir_encoding,
    const json& density_network,
    const json& rgb_network
) : m_n_pos_dims{n_pos_dims},
    m_n_dir_dims{n_dir_dims},
    m_dir_offset{dir_offset},
    m_n_extra_dims{n_extra_dims}
{
    using namespace tcnn;

    // Create position encoding (typically HashGrid)
    uint32_t alignment = density_network.contains("otype") &&
        (tcnn::equals_case_insensitive(density_network["otype"], "FullyFusedMLP") ||
         tcnn::equals_case_insensitive(density_network["otype"], "MegakernelMLP")) ? 16u : 8u;

    m_pos_encoding.reset(create_encoding<T>(n_pos_dims, pos_encoding, alignment));

    // Create density network
    json local_density_network_config = density_network;
    local_density_network_config["n_input_dims"] = m_pos_encoding->padded_output_width();
    if (!density_network.contains("n_output_dims")) {
        local_density_network_config["n_output_dims"] = 16;
    }
    m_density_network.reset(create_network<T>(local_density_network_config));

    // Create direction encoding
    uint32_t rgb_alignment = minimum_alignment(rgb_network);
    m_dir_encoding.reset(create_encoding<T>(m_n_dir_dims + m_n_extra_dims, dir_encoding, rgb_alignment));

    // Create RGB network
    m_rgb_network_input_width = next_multiple(
        m_dir_encoding->padded_output_width() + m_density_network->padded_output_width(),
        rgb_alignment
    );

    json local_rgb_network_config = rgb_network;
    local_rgb_network_config["n_input_dims"] = m_rgb_network_input_width;
    local_rgb_network_config["n_output_dims"] = 3;
    m_rgb_network.reset(create_network<T>(local_rgb_network_config));

    // Combined density model (encoding + network)
    m_density_model = std::make_shared<NetworkWithInputEncoding<T>>(
        m_pos_encoding,
        m_density_network
    );
}

template <typename T>
void NerfNetwork<T>::inference_mixed_precision_impl(
    cudaStream_t stream,
    const tcnn::GPUMatrixDynamic<float>& input,
    tcnn::GPUMatrixDynamic<T>& output,
    bool use_inference_params
) {
    using namespace tcnn;

    uint32_t batch_size = input.n();

    // Allocate temporary buffers
    GPUMatrixDynamic<T> density_network_input{
        m_pos_encoding->padded_output_width(),
        batch_size,
        stream,
        m_pos_encoding->preferred_output_layout()
    };

    GPUMatrixDynamic<T> rgb_network_input{
        m_rgb_network_input_width,
        batch_size,
        stream,
        m_dir_encoding->preferred_output_layout()
    };

    // Density network output goes into the first part of RGB network input
    GPUMatrixDynamic<T> density_network_output = rgb_network_input.slice_rows(
        0,
        m_density_network->padded_output_width()
    );

    GPUMatrixDynamic<T> rgb_network_output{
        output.data(),
        m_rgb_network->padded_output_width(),
        batch_size,
        output.layout()
    };

    // 1. Encode position
    m_pos_encoding->inference_mixed_precision(
        stream,
        input.slice_rows(0, m_pos_encoding->input_width()),
        density_network_input,
        use_inference_params
    );

    // 2. Density network
    m_density_network->inference_mixed_precision(
        stream,
        density_network_input,
        density_network_output,
        use_inference_params
    );

    // 3. Encode direction
    auto dir_out = rgb_network_input.slice_rows(
        m_density_network->padded_output_width(),
        m_dir_encoding->padded_output_width()
    );

    m_dir_encoding->inference_mixed_precision(
        stream,
        input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
        dir_out,
        use_inference_params
    );

    // 4. RGB network
    m_rgb_network->inference_mixed_precision(
        stream,
        rgb_network_input,
        rgb_network_output,
        use_inference_params
    );

    // 5. Extract density into output (4th channel)
    linear_kernel(extract_density_kernel<T>, 0, stream,
        batch_size,
        density_network_output.layout() == AoS ? density_network_output.stride() : 1,
        output.layout() == AoS ? padded_output_width() : 1,
        density_network_output.data(),
        output.data() + 3 * (output.layout() == AoS ? 1 : batch_size)
    );
}

template <typename T>
std::unique_ptr<tcnn::Context> NerfNetwork<T>::forward_impl(
    cudaStream_t stream,
    const tcnn::GPUMatrixDynamic<float>& input,
    tcnn::GPUMatrixDynamic<T>* output,
    bool use_inference_params,
    bool prepare_input_gradients
) {
    using namespace tcnn;

    uint32_t batch_size = input.n();
    auto forward = std::make_unique<ForwardContext>();

    // Allocate forward buffers
    forward->density_network_input = GPUMatrixDynamic<T>{
        m_pos_encoding->padded_output_width(),
        batch_size,
        stream,
        m_pos_encoding->preferred_output_layout()
    };

    forward->rgb_network_input = GPUMatrixDynamic<T>{
        m_rgb_network_input_width,
        batch_size,
        stream,
        m_dir_encoding->preferred_output_layout()
    };

    // Position encoding forward
    forward->pos_encoding_ctx = m_pos_encoding->forward(
        stream,
        input.slice_rows(0, m_pos_encoding->input_width()),
        &forward->density_network_input,
        use_inference_params,
        prepare_input_gradients
    );

    // Density network forward
    forward->density_network_output = forward->rgb_network_input.slice_rows(
        0,
        m_density_network->padded_output_width()
    );

    forward->density_network_ctx = m_density_network->forward(
        stream,
        forward->density_network_input,
        &forward->density_network_output,
        use_inference_params,
        prepare_input_gradients
    );

    // Direction encoding forward
    auto dir_out = forward->rgb_network_input.slice_rows(
        m_density_network->padded_output_width(),
        m_dir_encoding->padded_output_width()
    );

    forward->dir_encoding_ctx = m_dir_encoding->forward(
        stream,
        input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
        &dir_out,
        use_inference_params,
        prepare_input_gradients
    );

    // RGB network forward
    if (output) {
        forward->rgb_network_output = GPUMatrixDynamic<T>{
            output->data(),
            m_rgb_network->padded_output_width(),
            batch_size,
            output->layout()
        };
    }

    forward->rgb_network_ctx = m_rgb_network->forward(
        stream,
        forward->rgb_network_input,
        output ? &forward->rgb_network_output : nullptr,
        use_inference_params,
        prepare_input_gradients
    );

    // Extract density if output is provided
    if (output) {
        linear_kernel(extract_density_kernel<T>, 0, stream,
            batch_size,
            m_dir_encoding->preferred_output_layout() == AoS ? forward->density_network_output.stride() : 1,
            padded_output_width(),
            forward->density_network_output.data(),
            output->data() + 3
        );
    }

    return forward;
}

template <typename T>
void NerfNetwork<T>::backward_impl(
    cudaStream_t stream,
    const tcnn::Context& ctx,
    const tcnn::GPUMatrixDynamic<float>& input,
    const tcnn::GPUMatrixDynamic<T>& output,
    const tcnn::GPUMatrixDynamic<T>& dL_doutput,
    tcnn::GPUMatrixDynamic<float>* dL_dinput,
    bool use_inference_params,
    tcnn::GradientMode param_gradients_mode
) {
    using namespace tcnn;

    const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
    uint32_t batch_size = input.n();

    // Extract RGB gradients
    GPUMatrix<T> dL_drgb{m_rgb_network->padded_output_width(), batch_size, stream};
    CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb.data(), 0, dL_drgb.n_bytes(), stream));

    linear_kernel(extract_rgb_kernel<T>, 0, stream,
        batch_size * 3,
        dL_drgb.m(),
        dL_doutput.m(),
        dL_doutput.data(),
        dL_drgb.data()
    );

    // RGB network backward
    const GPUMatrixDynamic<T> rgb_network_output{
        (T*)output.data(),
        m_rgb_network->padded_output_width(),
        batch_size,
        output.layout()
    };

    GPUMatrixDynamic<T> dL_drgb_network_input{
        m_rgb_network_input_width,
        batch_size,
        stream,
        m_dir_encoding->preferred_output_layout()
    };

    m_rgb_network->backward(
        stream,
        *forward.rgb_network_ctx,
        forward.rgb_network_input,
        rgb_network_output,
        dL_drgb,
        &dL_drgb_network_input,
        use_inference_params,
        param_gradients_mode
    );

    // Direction encoding backward
    if (m_dir_encoding->n_params() > 0 || dL_dinput) {
        GPUMatrixDynamic<T> dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(
            m_density_network->padded_output_width(),
            m_dir_encoding->padded_output_width()
        );

        GPUMatrixDynamic<float> dL_ddir_encoding_input;
        if (dL_dinput) {
            dL_ddir_encoding_input = dL_dinput->slice_rows(
                m_dir_offset,
                m_dir_encoding->input_width()
            );
        }

        m_dir_encoding->backward(
            stream,
            *forward.dir_encoding_ctx,
            input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
            forward.rgb_network_input.slice_rows(
                m_density_network->padded_output_width(),
                m_dir_encoding->padded_output_width()
            ),
            dL_ddir_encoding_output,
            dL_dinput ? &dL_ddir_encoding_input : nullptr,
            use_inference_params,
            param_gradients_mode
        );
    }

    // Add density gradient
    GPUMatrixDynamic<T> dL_ddensity_network_output = dL_drgb_network_input.slice_rows(
        0,
        m_density_network->padded_output_width()
    );

    linear_kernel(add_density_gradient_kernel<T>, 0, stream,
        batch_size,
        dL_doutput.m(),
        dL_doutput.data(),
        dL_ddensity_network_output.layout() == RM ? 1 : dL_ddensity_network_output.stride(),
        dL_ddensity_network_output.data()
    );

    // Density network backward
    GPUMatrixDynamic<T> dL_ddensity_network_input;
    if (m_pos_encoding->n_params() > 0 || dL_dinput) {
        dL_ddensity_network_input = GPUMatrixDynamic<T>{
            m_pos_encoding->padded_output_width(),
            batch_size,
            stream,
            m_pos_encoding->preferred_output_layout()
        };
    }

    m_density_network->backward(
        stream,
        *forward.density_network_ctx,
        forward.density_network_input,
        forward.density_network_output,
        dL_ddensity_network_output,
        dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr,
        use_inference_params,
        param_gradients_mode
    );

    // Position encoding backward
    if (dL_ddensity_network_input.data()) {
        GPUMatrixDynamic<float> dL_dpos_encoding_input;
        if (dL_dinput) {
            dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
        }

        m_pos_encoding->backward(
            stream,
            *forward.pos_encoding_ctx,
            input.slice_rows(0, m_pos_encoding->input_width()),
            forward.density_network_input,
            dL_ddensity_network_input,
            dL_dinput ? &dL_dpos_encoding_input : nullptr,
            use_inference_params,
            param_gradients_mode
        );
    }
}

// Explicit template instantiation
template class NerfNetwork<__half>;
template class NerfNetwork<float>;

} // namespace ngp

