/*
 * ngp-minimal: Testbed - Main training orchestration
 * Simplified testbed for NeRF-synthetic training only
 */

#pragma once

#include <ngp-minimal/common.h>
#include <ngp-minimal/common_device.cuh>
#include <ngp-minimal/gpu_memory.h>
#include <ngp-minimal/nerf_device.cuh>
#include <ngp-minimal/nerf_loader.h>
#include <ngp-minimal/nerf_network.h>

#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>
#include <tiny-cuda-nn/loss.h>

#include <json/json.hpp>
#include <filesystem/path.h>

#include <memory>
#include <atomic>

namespace ngp {

class Testbed {
public:
    Testbed(ETestbedMode mode = ETestbedMode::Nerf);
    ~Testbed();

    // Core training interface
    void load_training_data(const fs::path& path);
    void reload_network_from_file(const fs::path& config_path);
    void reload_network_from_json(const nlohmann::json& config);
    void create_empty_nerf_network();  // Public for testing

    bool frame(); // Main training loop
    void train(uint32_t batch_size);

    // NeRF-specific data structure
    struct Nerf {
        struct Training {
            NerfDataset dataset;

            // Density grid (occupancy grid for空 space skipping)
            GPUMemory<uint8_t> density_grid;        // Binary occupancy grid
            GPUMemory<float> density_grid_mean;     // Mean density per cell

            // Training configuration
            ETrainMode train_mode = ETrainMode::Nerf;
            ELossType loss_type = ELossType::L2;

            bool random_bg_color = true;
            bool linear_colors = false;
            bool snap_to_pixel_centers = false;

            // Grid update frequency
            uint32_t n_steps_since_error_map_update = 0;
            uint32_t n_steps_between_error_map_updates = 128;
            uint32_t n_rays_per_batch = 1<<12; // 4096 rays per batch

            // Training counters
            GPUMemory<uint32_t> counters_rgb;
        } training;

        float sharpen = 0.0f;
    } m_nerf;

    // Training state
    bool m_train = true;
    uint32_t m_training_batch_size = 1<<18; // 256K samples
    uint64_t m_training_step = 0;

    // Loss tracking (simple float instead of LossScalar)
    float m_loss_scalar = 1.0f;

    // Network and trainer
    using precision_t = tcnn::network_precision_t;
    std::shared_ptr<NerfNetwork<precision_t>> m_nerf_network;
    std::shared_ptr<tcnn::Optimizer<precision_t>> m_optimizer;
    std::shared_ptr<tcnn::Loss<precision_t>> m_loss;
    std::shared_ptr<tcnn::Trainer<float, precision_t, precision_t>> m_trainer;

    // Mode
    ETestbedMode m_testbed_mode = ETestbedMode::None;
    fs::path m_data_path;

    // CUDA stream
    cudaStream_t m_stream;

private:
    // Internal methods
    void set_mode(ETestbedMode mode);
    void load_nerf(const fs::path& data_path);
    void load_nerf_post();
    void reset_network();

    // Training helpers
    void training_prep_nerf(uint32_t batch_size);
    void train_nerf_step(uint32_t batch_size);
    void update_density_grid_nerf();

    // Network config
    nlohmann::json m_network_config;
    bool m_network_config_path_set = false;
};

} // namespace ngp
