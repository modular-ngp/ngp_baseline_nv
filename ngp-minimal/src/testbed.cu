/*
 * ngp-minimal: Testbed implementation
 * Main training orchestration logic
 */

#include <ngp-minimal/testbed.h>
#include <tinylogger/tinylogger.h>

#include <fstream>

namespace ngp {

Testbed::Testbed(ETestbedMode mode) {
    // Create CUDA stream
    cudaStreamCreate(&m_stream);

    // Set initial mode
    if (mode != ETestbedMode::None) {
        set_mode(mode);
    }
}

Testbed::~Testbed() {
    cudaStreamDestroy(m_stream);
}

void Testbed::set_mode(ETestbedMode mode) {
    if (m_testbed_mode == mode) {
        return;
    }

    tlog::info() << "Setting mode to: " << (int)mode;

    // Reset everything
    m_nerf = {};
    m_nerf_network = nullptr;
    m_optimizer = nullptr;
    m_loss = nullptr;
    m_trainer = nullptr;
    m_training_step = 0;

    m_testbed_mode = mode;
}

void Testbed::load_training_data(const fs::path& path) {
    if (!path.exists()) {
        throw std::runtime_error("Training data path does not exist: " + path.str());
    }

    tlog::info() << "Loading training data from: " << path.str();

    // Determine mode from path
    m_testbed_mode = ETestbedMode::Nerf; // Always NeRF for ngp-minimal
    m_data_path = path;

    // Load based on mode
    load_nerf(path);
}

void Testbed::load_nerf(const fs::path& data_path) {
    tlog::info() << "Loading NeRF dataset...";

    // Find JSON files
    std::vector<fs::path> json_paths;

    if (data_path.is_directory()) {
        // Look for transforms_*.json files
        std::vector<std::string> json_names = {
            "transforms_train.json",
            "transforms_val.json",
            "transforms_test.json"
        };

        for (const auto& name : json_names) {
            fs::path json_path = data_path / name;
            if (json_path.exists()) {
                json_paths.push_back(json_path);
            }
        }
    } else if (data_path.is_file()) {
        json_paths.push_back(data_path);
    }

    if (json_paths.empty()) {
        throw std::runtime_error("No transforms JSON files found in: " + data_path.str());
    }

    // Load dataset
    m_nerf.training.dataset = ngp::load_nerf(json_paths, m_nerf.sharpen);

    tlog::success() << "NeRF dataset loaded successfully";

    // Post-load initialization
    load_nerf_post();
}

void Testbed::load_nerf_post() {
    tlog::info() << "Initializing NeRF training structures...";

    // Initialize density grid
    uint32_t n_cells = NERF_GRID_N_CELLS();
    m_nerf.training.density_grid.resize(n_cells);
    m_nerf.training.density_grid.memset(0);

    m_nerf.training.density_grid_mean.resize(n_cells);
    m_nerf.training.density_grid_mean.memset(0);

    // Initialize counters
    m_nerf.training.counters_rgb.resize(4);
    m_nerf.training.counters_rgb.memset(0);

    tlog::success() << "NeRF training structures initialized";
}

void Testbed::reload_network_from_file(const fs::path& config_path) {
    if (!config_path.exists()) {
        throw std::runtime_error("Network config file does not exist: " + config_path.str());
    }

    tlog::info() << "Loading network config from: " << config_path.str();

    std::ifstream f(config_path.str());
    nlohmann::json config = nlohmann::json::parse(f, nullptr, true, true);

    reload_network_from_json(config);
    m_network_config_path_set = true;
}

void Testbed::reload_network_from_json(const nlohmann::json& config) {
    m_network_config = config;
    reset_network();
}

void Testbed::reset_network() {
    tlog::info() << "Creating NeRF network...";

    // If no config and no network exists, create empty one
    if (m_network_config.is_null() || m_network_config.empty()) {
        create_empty_nerf_network();
        return;
    }

    // Parse network config
    nlohmann::json network_config = m_network_config;
    nlohmann::json encoding_config = network_config.value("encoding", nlohmann::json::object());
    nlohmann::json loss_config = network_config.value("loss", nlohmann::json::object());
    nlohmann::json optimizer_config = network_config.value("optimizer", nlohmann::json::object());

    // Direction encoding (simpler than position)
    nlohmann::json dir_encoding_config = network_config.value("dir_encoding", nlohmann::json::object());
    if (dir_encoding_config.empty()) {
        dir_encoding_config = {
            {"otype", "SphericalHarmonics"},
            {"degree", 4}
        };
    }

    // MLP configs
    nlohmann::json density_network_config = network_config.value("network", nlohmann::json::object());
    nlohmann::json rgb_network_config = network_config.value("rgb_network", nlohmann::json::object());

    if (density_network_config.empty()) {
        density_network_config = {
            {"otype", "FullyFusedMLP"},
            {"n_neurons", 64},
            {"n_hidden_layers", 1},
            {"activation", "ReLU"},
            {"output_activation", "None"}
        };
    }

    if (rgb_network_config.empty()) {
        rgb_network_config = {
            {"otype", "FullyFusedMLP"},
            {"n_neurons", 64},
            {"n_hidden_layers", 2},
            {"activation", "ReLU"},
            {"output_activation", "Sigmoid"}
        };
    }

    // Create network
    uint32_t n_pos_dims = 3;
    uint32_t n_dir_dims = 3;
    uint32_t n_extra_dims = 0;
    uint32_t dir_offset = n_pos_dims;

    m_nerf_network = std::make_shared<NerfNetwork<precision_t>>(
        n_pos_dims,
        n_dir_dims,
        n_extra_dims,
        dir_offset,
        encoding_config,
        dir_encoding_config,
        density_network_config,
        rgb_network_config
    );

    // Create loss
    if (loss_config.empty()) {
        loss_config = {{"otype", "L2"}};
    }
    m_loss.reset(tcnn::create_loss<precision_t>(loss_config));

    // Create optimizer
    if (optimizer_config.empty()) {
        optimizer_config = {
            {"otype", "Adam"},
            {"learning_rate", 1e-2},
            {"beta1", 0.9},
            {"beta2", 0.99},
            {"epsilon", 1e-15}
        };
    }
    m_optimizer.reset(tcnn::create_optimizer<precision_t>(optimizer_config));

    // Create trainer
    m_trainer = std::make_shared<tcnn::Trainer<float, precision_t, precision_t>>(
        m_nerf_network,
        m_optimizer,
        m_loss
    );

    tlog::success() << "NeRF network created successfully";
    tlog::info() << "  Network parameters: " << m_nerf_network->n_params();
}

void Testbed::create_empty_nerf_network() {
    tlog::info() << "Creating default NeRF network...";

    // Default HashGrid encoding
    nlohmann::json encoding_config = {
        {"otype", "HashGrid"},
        {"n_levels", 16},
        {"n_features_per_level", 2},
        {"log2_hashmap_size", 19},
        {"base_resolution", 16},
        {"per_level_scale", 1.5}
    };

    nlohmann::json dir_encoding_config = {
        {"otype", "SphericalHarmonics"},
        {"degree", 4}
    };

    nlohmann::json density_network_config = {
        {"otype", "FullyFusedMLP"},
        {"n_neurons", 64},
        {"n_hidden_layers", 1},
        {"activation", "ReLU"},
        {"output_activation", "None"}
    };

    nlohmann::json rgb_network_config = {
        {"otype", "FullyFusedMLP"},
        {"n_neurons", 64},
        {"n_hidden_layers", 2},
        {"activation", "ReLU"},
        {"output_activation", "Sigmoid"}
    };

    nlohmann::json full_config = {
        {"encoding", encoding_config},
        {"dir_encoding", dir_encoding_config},
        {"network", density_network_config},
        {"rgb_network", rgb_network_config}
    };

    reload_network_from_json(full_config);
}

bool Testbed::frame() {
    // Main training loop entry point
    if (m_train && m_trainer) {
        train(m_training_batch_size);
        ++m_training_step;
    }

    return true; // Continue training (minimal version doesn't have exit condition)
}

void Testbed::train(uint32_t batch_size) {
    if (m_testbed_mode != ETestbedMode::Nerf || !m_trainer) {
        return;
    }

    training_prep_nerf(batch_size);
    train_nerf_step(batch_size);
}

} // namespace ngp

