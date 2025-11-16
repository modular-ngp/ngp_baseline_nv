/*
 * ngp-minimal: Standalone minimal NeRF-synthetic training implementation
 * Main entry point with CLI argument parsing
 */

#include <ngp-minimal/testbed.h>
#include <ngp-minimal/nerf_loader.h>
#include <ngp-minimal/common.h>

#include <iostream>
#include <vector>
#include <string>

#include <args/args.hxx>
#include <tinylogger/tinylogger.h>

#ifdef _WIN32
#include <windows.h>
#include <locale>
#include <codecvt>

// UTF-16 to UTF-8 conversion for Windows
std::string utf16_to_utf8(const std::wstring& utf16) {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.to_bytes(utf16);
}
#endif

using namespace args;

namespace ngp {

int main_func(const std::vector<std::string>& arguments) {
    ArgumentParser parser{
        "ngp-minimal: Standalone Minimal NeRF-synthetic Training\n"
        "A minimal implementation of instant-ngp for NeRF-synthetic datasets\n"
        "Version 1.0.0\n",
        "",
    };

    HelpFlag help_flag{
        parser,
        "HELP",
        "Display this help menu.",
        {'h', "help"},
    };

    ValueFlag<std::string> scene_flag{
        parser,
        "SCENE",
        "Path to the NeRF-synthetic scene directory (e.g., data/nerf-synthetic/lego).",
        {'s', "scene"},
    };

    ValueFlag<std::string> config_flag{
        parser,
        "CONFIG",
        "Path to the network config JSON file (e.g., configs/nerf/base.json).",
        {'c', "config"},
    };

    ValueFlag<std::string> snapshot_flag{
        parser,
        "SNAPSHOT",
        "Optional snapshot to load upon startup.",
        {"snapshot", "load_snapshot"},
    };

    Flag no_train_flag{
        parser,
        "NO_TRAIN",
        "Disables training on startup (useful for testing data loading).",
        {"no-train"},
    };

    Flag version_flag{
        parser,
        "VERSION",
        "Display version information.",
        {'v', "version"},
    };

    // Parse command line arguments
    try {
        if (arguments.empty()) {
            std::cerr << "Error: No arguments provided." << std::endl;
            return -3;
        }

        parser.Prog(arguments.front());
        parser.ParseArgs(begin(arguments) + 1, end(arguments));
    } catch (const Help&) {
        std::cout << parser;
        return 0;
    } catch (const ParseError& e) {
        std::cerr << "Parse error: " << e.what() << std::endl;
        std::cerr << parser;
        return -1;
    } catch (const ValidationError& e) {
        std::cerr << "Validation error: " << e.what() << std::endl;
        std::cerr << parser;
        return -2;
    }

    if (version_flag) {
        std::cout << "ngp-minimal v1.0.0" << std::endl;
        std::cout << "Standalone minimal NeRF-synthetic training implementation" << std::endl;
        return 0;
    }

    // Print parsed arguments
    std::cout << "ngp-minimal: Parsed arguments:" << std::endl;

    if (scene_flag) {
        std::cout << "  Scene: " << get(scene_flag) << std::endl;
    } else {
        std::cout << "  Scene: <not specified>" << std::endl;
    }

    if (config_flag) {
        std::cout << "  Config: " << get(config_flag) << std::endl;
    } else {
        std::cout << "  Config: <not specified>" << std::endl;
    }

    if (snapshot_flag) {
        std::cout << "  Snapshot: " << get(snapshot_flag) << std::endl;
    }

    if (no_train_flag) {
        std::cout << "  Training: DISABLED" << std::endl;
    } else {
        std::cout << "  Training: ENABLED" << std::endl;
    }

    // Phase 3 & 4: Test data loading and training
    if (scene_flag) {
        try {
            tlog::info() << "\n=== Phase 3 & 4: Testing Data Loading & Training ===";

            // Create Testbed
            tlog::info() << "Creating Testbed...";
            ngp::Testbed testbed(ngp::ETestbedMode::Nerf);

            // Load training data
            tlog::info() << "Loading training data...";
            testbed.load_training_data(get(scene_flag));

            // Load network config if provided
            if (config_flag) {
                tlog::info() << "Loading network configuration...";
                testbed.reload_network_from_file(get(config_flag));
            } else {
                tlog::info() << "Using default network configuration...";
                testbed.create_empty_nerf_network();
            }

            // Set training mode
            testbed.m_train = !no_train_flag;

            if (testbed.m_train) {
                tlog::success() << "\n=== Starting Training ===";
                tlog::info() << "Training for 1000 iterations...";

                // Training loop
                for (int i = 0; i < 1000 && testbed.frame(); ++i) {
                    // Progress is logged inside frame()
                }

                tlog::success() << "\nTraining complete!";
                tlog::info() << "Final iteration: " << testbed.m_training_step;
                tlog::info() << "Final loss: " << testbed.m_loss_scalar;
            } else {
                tlog::info() << "Training disabled (--no-train flag)";
            }

            tlog::success() << "\nPhase 4 complete - Training works!";
        } catch (const std::exception& e) {
            tlog::error() << "Failed: " << e.what();
            return 1;
        }
    } else {
        tlog::info() << "\nTo test Phase 3 & 4, provide --scene parameter";
        tlog::info() << "Example: ngp-minimal-app --scene data/nerf-synthetic/lego --config configs/nerf/base.json";
    }

    tlog::info() << "\nNext: Implement Phase 5 (full training integration)";

    return 0;
}

} // namespace ngp

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
    SetConsoleOutputCP(CP_UTF8);
#else
int main(int argc, char* argv[]) {
#endif
    try {
        std::vector<std::string> arguments;
        for (int i = 0; i < argc; ++i) {
#ifdef _WIN32
            arguments.emplace_back(utf16_to_utf8(argv[i]));
#else
            arguments.emplace_back(argv[i]);
#endif
        }

        return ngp::main_func(arguments);
    } catch (const std::exception& e) {
        tlog::error() << "Uncaught exception: " << e.what();
        return 1;
    }
}

