#include <neural-graphics-primitives/testbed.h>

#include <tiny-cuda-nn/common.h>

#include <args/args.hxx>

#include <filesystem/path.h>

#include <vector>
#include <string>

using namespace args;
using namespace ngp;
using namespace std;

namespace {

int minimal_main_func(const std::vector<std::string>& arguments) {
	ArgumentParser parser{
		"ngp-minimal NeRF-synthetic trainer\n"
		"(CLI compatible subset of ngp-baseline-nv-app)\n",
		"",
	};

	HelpFlag help_flag{
		parser,
		"HELP",
		"Display this help menu.",
		{'h', "help"},
	};

	// Kept for compatibility, but has no effect.
	ValueFlag<string> mode_flag{
		parser,
		"MODE",
		"Deprecated. Do not use.",
		{'m', "mode"},
	};

	ValueFlag<string> network_config_flag{
		parser,
		"CONFIG",
		"Path to the network config. Uses the scene's default if unspecified.",
		{'n', 'c', "network", "config"},
	};

	Flag no_gui_flag{
		parser,
		"NO_GUI",
		"Disables the GUI and instead reports training progress on the command line.",
		{"no-gui"},
	};

	Flag vr_flag{
		parser,
		"VR",
		"Enables VR (ignored in minimal app).",
		{"vr"}
	};

	Flag no_train_flag{
		parser,
		"NO_TRAIN",
		"Disables training on startup.",
		{"no-train"},
	};

	ValueFlag<string> scene_flag{
		parser,
		"SCENE",
		"The scene to load. For ngp-minimal, this is expected to be a NeRF-synthetic dataset root.",
		{'s', "scene"},
	};

	ValueFlag<string> snapshot_flag{
		parser,
		"SNAPSHOT",
		"Optional snapshot to load upon startup.",
		{"snapshot", "load_snapshot"},
	};

	// GUI-related flags are parsed for compatibility but have no effect.
	ValueFlag<uint32_t> width_flag{
		parser,
		"WIDTH",
		"Resolution width of the GUI (ignored).",
		{"width"},
	};

	ValueFlag<uint32_t> height_flag{
		parser,
		"HEIGHT",
		"Resolution height of the GUI (ignored).",
		{"height"},
	};

	Flag version_flag{
		parser,
		"VERSION",
		"Display the version of ngp-minimal.",
		{'v', "version"},
	};

	PositionalList<string> files{
		parser,
		"files",
		"Files to be loaded. Can be a scene, network config, snapshot, camera path, or a combination of those.",
	};

	try {
		if (arguments.empty()) {
			tlog::error() << "Number of arguments must be bigger than 0.";
			return -3;
		}

		parser.Prog(arguments.front());
		parser.ParseArgs(begin(arguments) + 1, end(arguments));
	} catch (const Help&) {
		cout << parser;
		return 0;
	} catch (const ParseError& e) {
		cerr << e.what() << endl;
		cerr << parser;
		return -1;
	} catch (const ValidationError& e) {
		cerr << e.what() << endl;
		cerr << parser;
		return -2;
	}

	if (version_flag) {
		tlog::none() << "ngp-minimal NeRF-synthetic trainer (v1.0.0-compatible)";
		return 0;
	}

	if (mode_flag) {
		tlog::warning() << "The '--mode' argument is no longer in use. It has no effect. The mode is automatically chosen based on the scene.";
	}

	if (vr_flag) {
		tlog::warning() << "The '--vr' flag is ignored in ngp-minimal.";
	}

	if (width_flag || height_flag || no_gui_flag) {
		tlog::warning() << "GUI-related flags are parsed but ngp-minimal never creates a GUI window.";
	}

	Testbed testbed;

	for (auto file : get(files)) {
		testbed.load_file(file);
	}

	if (scene_flag) {
		testbed.load_training_data(get(scene_flag));
	}

	if (snapshot_flag) {
		testbed.load_snapshot(static_cast<fs::path>(get(snapshot_flag)));
	} else if (network_config_flag) {
		testbed.reload_network_from_file(get(network_config_flag));
	}

	testbed.m_train = !no_train_flag;

	while (testbed.frame()) {
		tlog::info() << "iteration=" << testbed.m_training_step << " loss=" << testbed.m_loss_scalar.val();
	}

	return 0;
}

} // anonymous namespace

int main(int argc, char* argv[]) {
	try {
		std::vector<std::string> arguments;
		arguments.reserve(static_cast<size_t>(argc));
		for (int i = 0; i < argc; ++i) {
			arguments.emplace_back(argv[i]);
		}

		return minimal_main_func(arguments);
	} catch (const std::exception& e) {
		tlog::error() << fmt::format("Uncaught exception in ngp-minimal-app: {}", e.what());
		return 1;
	}
}
