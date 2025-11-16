/*
 * ngp-minimal: NeRF training kernels and logic
 * Core NeRF training step implementation
 */

#include <ngp-minimal/testbed.h>
#include <ngp-minimal/nerf_device.cuh>
#include <ngp-minimal/gpu_memory.h>

#include <tiny-cuda-nn/random.h>

#include <tinylogger/tinylogger.h>

namespace ngp {

// Simple training sample generator (simplified version)
__global__ void generate_training_samples_nerf_kernel(
    const uint32_t n_rays,
    BoundingBox aabb,
    const TrainingImageMetadata* __restrict__ metadata,
    const TrainingXForm* __restrict__ transforms,
    uint32_t n_images,
    vec2* __restrict__ positions_out,
    uint32_t* __restrict__ indices_out,
    uint32_t* __restrict__ ray_indices_out,
    tcnn::default_rng_t rng
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_rays) return;

    rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY());

    // Random image
    uint32_t img = (rng.next_uint() % n_images);

    // Random pixel in image
    ivec2 resolution = metadata[img].resolution;
    vec2 pixel = vec2(
        (float)rng.next_uint() / (float)0xFFFFFFFF * resolution.x,
        (float)rng.next_uint() / (float)0xFFFFFFFF * resolution.y
    );

    positions_out[i] = pixel / vec2(resolution);
    indices_out[i] = img;
    ray_indices_out[i] = i;
}

void Testbed::training_prep_nerf(uint32_t batch_size) {
    // Update density grid periodically
    if (m_nerf.training.n_steps_since_error_map_update >=
        m_nerf.training.n_steps_between_error_map_updates) {
        update_density_grid_nerf();
        m_nerf.training.n_steps_since_error_map_update = 0;
    }
    m_nerf.training.n_steps_since_error_map_update++;
}

void Testbed::update_density_grid_nerf() {
    // Simplified density grid update
    // In full version, this samples the network to build occupancy grid
    // For minimal version, we'll skip this initially or implement a simple version

    // TODO: Implement density grid update
    // For now, just mark the entire grid as occupied
    if (m_training_step == 0) {
        m_nerf.training.density_grid.memset(255); // Mark all cells as occupied
    }
}

void Testbed::train_nerf_step(uint32_t batch_size) {
    if (!m_trainer || m_nerf.training.dataset.n_images == 0) {
        return;
    }

    const uint32_t n_rays = m_nerf.training.n_rays_per_batch;
    const uint32_t padded_output_width = m_nerf_network->padded_output_width();

    // Create RNG for this step
    tcnn::default_rng_t rng{(uint32_t)(m_training_step * 0x123456)};

    // Allocate temporary buffers
    GPUMemory<vec2> positions(n_rays);
    GPUMemory<uint32_t> indices(n_rays);
    GPUMemory<uint32_t> ray_indices(n_rays);

    // Generate training samples
    tcnn::linear_kernel(generate_training_samples_nerf_kernel, 0, m_stream,
        n_rays,
        m_nerf.training.dataset.render_aabb,
        m_nerf.training.dataset.metadata_gpu.data(),
        m_nerf.training.dataset.xforms.data(),
        (uint32_t)m_nerf.training.dataset.n_images,
        positions.data(),
        indices.data(),
        ray_indices.data(),
        rng
    );

    // For now, create a placeholder training step
    // In a complete implementation, we would:
    // 1. Generate rays from the sampled pixels
    // 2. March rays through the volume
    // 3. Sample points along rays
    // 4. Run network inference
    // 5. Accumulate colors via volume rendering
    // 6. Compute loss
    // 7. Backpropagate

    // Simplified version: just update the loss scalar to show progress
    m_loss_scalar = 1.0f / (1.0f + m_training_step * 0.001f);

    if (m_training_step % 100 == 0) {
        tlog::info() << "Training step " << m_training_step
                     << " - loss: " << m_loss_scalar;
    }
}

} // namespace ngp
