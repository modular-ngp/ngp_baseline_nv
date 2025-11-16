/*
 * ngp-minimal: Standalone minimal NeRF-synthetic training implementation
 * Main entry point with CLI argument parsing
 */

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

    // TODO: Phase 3-5 implementation

    tlog::success() << "\nngp-minimal: Phase 2 complete - CLI parsing works!";

    // Phase 3: Test data loading
    if (scene_flag) {
        try {
            tlog::info() << "\n=== Phase 3: Testing Data Loading ===";

            fs::path scene_path = get(scene_flag);
            std::vector<fs::path> json_paths;

            // Look for transform JSON files
            if (scene_path.is_directory()) {
                // Try standard NeRF-synthetic file names
                std::vector<std::string> json_names = {
                    "transforms_train.json",
                    "transforms_val.json",
                    "transforms_test.json"
                };

                for (const auto& name : json_names) {
                    fs::path json_path = scene_path / name;
                    if (json_path.exists()) {
                        json_paths.push_back(json_path);
                        tlog::info() << "Found: " << json_path.str();
                    }
                }
            } else if (scene_path.is_file()) {
                json_paths.push_back(scene_path);
            }

            if (json_paths.empty()) {
                tlog::error() << "No transform JSON files found in: " << scene_path.str();
            } else {
                // Load the dataset
                tlog::info() << "Loading dataset...";
                auto dataset = load_nerf(json_paths);

                tlog::success() << "\n=== Dataset loaded successfully! ===";
                tlog::info() << "Summary:";
                tlog::info() << "  Total images: " << dataset.n_images;
                tlog::info() << "  Image resolution: " << dataset.metadata[0].resolution.x
                            << " x " << dataset.metadata[0].resolution.y;
                tlog::info() << "  Focal length: " << dataset.metadata[0].focal_length.x
                            << " (x), " << dataset.metadata[0].focal_length.y << " (y)";
                tlog::info() << "  AABB scale: " << dataset.aabb_scale;
                tlog::info() << "  Scale: " << dataset.scale;
                tlog::info() << "  Offset: (" << dataset.offset.x << ", "
                            << dataset.offset.y << ", " << dataset.offset.z << ")";

                tlog::success() << "\nPhase 3 complete - Data loading works!";
            }
        } catch (const std::exception& e) {
            tlog::error() << "Failed to load dataset: " << e.what();
            return 1;
        }
    } else {
        tlog::info() << "\nTo test Phase 3 data loading, provide --scene parameter";
        tlog::info() << "Example: ngp-minimal-app --scene data/nerf-synthetic/lego";
    }

    tlog::info() << "\nNext: Implement Phase 4 (network & training)";

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

