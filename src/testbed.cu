/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   testbed.cu
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/json_binding.h>
#include <neural-graphics-primitives/marching_cubes.h>
#include <neural-graphics-primitives/nerf_loader.h>
#include <neural-graphics-primitives/nerf_network.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/takikawa_encoding.cuh>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/tinyexr_wrapper.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>
#include <neural-graphics-primitives/triangle_bvh.cuh>
#include <neural-graphics-primitives/triangle_octree.cuh>

#include <tiny-cuda-nn/encodings/multi_level_interface.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/rtc_kernel.h>
#include <tiny-cuda-nn/trainer.h>

#include <json/json.hpp>

#include <filesystem/directory.h>
#include <filesystem/path.h>

#include <zstr.hpp>

#include <fstream>
#include <unordered_set>

// Windows.h is evil
#undef min
#undef max
#undef near
#undef far


using namespace std::literals::chrono_literals;

namespace ngp {

int do_system(const std::string& cmd) {
#ifdef _WIN32
	tlog::info() << "> " << cmd;
	return _wsystem(utf8_to_utf16(cmd).c_str());
#else
	tlog::info() << "$ " << cmd;
	return system(cmd.c_str());
#endif
}

std::atomic<size_t> g_total_n_bytes_allocated{0};

json merge_parent_network_config(const json& child, const fs::path& child_path) {
	if (!child.contains("parent")) {
		return child;
	}
	fs::path parent_path = child_path.parent_path() / std::string(child["parent"]);
	tlog::info() << "Loading parent network config from: " << parent_path.str();
	std::ifstream f{native_string(parent_path)};
	json parent = json::parse(f, nullptr, true, true);
	parent = merge_parent_network_config(parent, parent_path);
	parent.merge_patch(child);
	return parent;
}

std::string get_filename_in_data_path_with_suffix(fs::path data_path, fs::path network_config_path, const char* suffix) {
	// use the network config name along with the data path to build a filename with the requested suffix & extension
	std::string default_name = network_config_path.basename();
	if (default_name == "") {
		default_name = "base";
	}

	if (data_path.empty()) {
		return default_name + std::string(suffix);
	}

	if (data_path.is_directory()) {
		return (data_path / (default_name + std::string{suffix})).str();
	}

	return data_path.stem().str() + "_" + default_name + std::string(suffix);
}

void Testbed::load_training_data(const fs::path& path) {
	if (!path.exists()) {
		throw std::runtime_error{fmt::format("Data path '{}' does not exist.", path.str())};
	}

	// Automatically determine the mode from the first scene that's loaded
	ETestbedMode scene_mode = mode_from_scene(path.str());
	if (scene_mode == ETestbedMode::None) {
		throw std::runtime_error{fmt::format("Unknown scene format for path '{}'.", path.str())};
	}

	set_mode(scene_mode);

	m_data_path = path;

	switch (m_testbed_mode) {
		case ETestbedMode::Nerf: load_nerf(path); break;
		// case ETestbedMode::Sdf: load_mesh(path); break;
		// case ETestbedMode::Image: load_image(path); break;
		// case ETestbedMode::Volume: load_volume(path); break;
		default: throw std::runtime_error{"Invalid testbed mode."};
	}

	m_training_data_available = true;
}

void Testbed::reload_training_data() {
	if (m_data_path.exists()) {
		load_training_data(m_data_path.str());
	}
}

void Testbed::clear_training_data() {
	m_training_data_available = false;
	m_nerf.training.dataset.metadata.clear();
}

void Testbed::set_mode(ETestbedMode mode) {
	if (mode == m_testbed_mode) {
		return;
	}

	// Reset mode-specific members
	m_mesh = {};
	m_nerf = {};

	// Kill training-related things
	m_encoding = {};
	m_loss = {};
	m_network = {};
	m_nerf_network = {};
	m_optimizer = {};
	m_trainer = {};
	m_envmap = {};
	m_distortion = {};
	m_training_data_available = false;

	// Clear device-owned data that might be mode-specific
	for (auto&& device : m_devices) {
		device.clear();
	}

	// Reset paths that might be attached to the chosen mode
	m_data_path = fs::path{};

	m_testbed_mode = mode;

	// Set various defaults depending on mode
	if (m_testbed_mode == ETestbedMode::Nerf) {
		if (m_devices.size() > 1) {
			m_use_aux_devices = true;
		}
		if (m_dlss_provider && m_aperture_size == 0.0f) {
			m_dlss = true;
		}
	} else {
		m_use_aux_devices = false;
		m_dlss = false;
	}

	reset_camera();
}

fs::path Testbed::find_network_config(const fs::path& network_config_path) {
	if (network_config_path.exists()) {
		return network_config_path;
	}

	// The following resolution steps do not work if the path is absolute. Treat it as nonexistent.
	if (network_config_path.is_absolute()) {
		return network_config_path;
	}

	fs::path candidate = root_dir() / "configs" / to_string(m_testbed_mode) / network_config_path;
	if (candidate.exists()) {
		return candidate;
	}

	return network_config_path;
}

json Testbed::load_network_config(std::istream& stream, bool is_compressed) {
	if (is_compressed) {
		zstr::istream zstream{stream};
		return json::from_msgpack(zstream);
	}
	return json::from_msgpack(stream);
}

json Testbed::load_network_config(const fs::path& network_config_path) {
	bool is_snapshot = equals_case_insensitive(network_config_path.extension(), "msgpack") ||
		equals_case_insensitive(network_config_path.extension(), "ingp");
	if (network_config_path.empty() || !network_config_path.exists()) {
		throw std::runtime_error{
			fmt::format("Network {} '{}' does not exist.", is_snapshot ? "snapshot" : "config", network_config_path.str())
		};
	}

	tlog::info() << "Loading network " << (is_snapshot ? "snapshot" : "config") << " from: " << network_config_path;

	json result;
	if (is_snapshot) {
		std::ifstream f{native_string(network_config_path), std::ios::in | std::ios::binary};
		if (equals_case_insensitive(network_config_path.extension(), "ingp")) {
			// zstr::ifstream applies zlib compression.
			zstr::istream zf{f};
			result = json::from_msgpack(zf);
		} else {
			result = json::from_msgpack(f);
		}
		// we assume parent pointers are already resolved in snapshots.
	} else if (equals_case_insensitive(network_config_path.extension(), "json")) {
		std::ifstream f{native_string(network_config_path)};
		result = json::parse(f, nullptr, true, true);
		result = merge_parent_network_config(result, network_config_path);
	}

	return result;
}

void Testbed::reload_network_from_file(const fs::path& path) {
	if (!path.empty()) {
		fs::path candidate = find_network_config(path);
		if (candidate.exists() || !m_network_config_path.exists()) {
			// Store the path _argument_ in the member variable. E.g. for the base config,
			// it'll store `base.json`, even though the loaded config will be
			// config/<mode>/base.json. This has the benefit of switching to the
			// appropriate config when switching modes.
			m_network_config_path = path;
		}
	}

	// If the testbed mode hasn't been decided yet, don't load a network yet, but
	// still keep track of the requested config (see above).
	if (m_testbed_mode == ETestbedMode::None) {
		return;
	}

	fs::path full_network_config_path = find_network_config(m_network_config_path);
	bool is_snapshot = equals_case_insensitive(full_network_config_path.extension(), "msgpack");

	if (!full_network_config_path.exists()) {
		tlog::warning() << "Network " << (is_snapshot ? "snapshot" : "config") << " path '" << full_network_config_path
						<< "' does not exist.";
	} else {
		m_network_config = load_network_config(full_network_config_path);
	}

	// Reset training if we haven't loaded a snapshot of an already trained model, in which case, presumably the network
	// configuration changed and the user is interested in seeing how it trains from scratch.
	if (!is_snapshot) {
		reset_network();
	}
}

void Testbed::reload_network_from_json(const json& json, const std::string& config_base_path) {
	// config_base_path is needed so that if the passed in json uses the 'parent' feature, we know where to look...
	// be sure to use a filename, or if a directory, end with a trailing slash
	m_network_config = merge_parent_network_config(json, config_base_path);
	reset_network();
}

void Testbed::load_file(const fs::path& path) {
	if (!path.exists()) {
		// If the path doesn't exist, but a network config can be resolved, load that.
		if (equals_case_insensitive(path.extension(), "json") && find_network_config(path).exists()) {
			reload_network_from_file(path);
			return;
		}

		tlog::error() << "File '" << path.str() << "' does not exist.";
		return;
	}

	if (equals_case_insensitive(path.extension(), "ingp") || equals_case_insensitive(path.extension(), "msgpack")) {
		load_snapshot(path);
		return;
	}

	// If we get a json file, we need to parse it to determine its purpose.
	if (equals_case_insensitive(path.extension(), "json")) {
		json file;
		{
			std::ifstream f{native_string(path)};
			file = json::parse(f, nullptr, true, true);
		}

		// Snapshot in json format... inefficient, but technically supported.
		if (file.contains("snapshot")) {
			load_snapshot(path);
			return;
		}

		// Regular network config
		if (file.contains("parent") || file.contains("network") || file.contains("encoding") || file.contains("loss") ||
			file.contains("optimizer")) {
			reload_network_from_file(path);
			return;
		}

		// Camera path
		if (file.contains("path")) {
			load_camera_path(path);
			return;
		}
	}

	// If the dragged file isn't any of the above, assume that it's training data
	try {
		bool was_training_data_available = m_training_data_available;
		load_training_data(path);

		if (!was_training_data_available) {
			// If we previously didn't have any training data and only now dragged
			// some into the window, it is very unlikely that the user doesn't
			// want to immediately start training on that data. So: go for it.
			m_train = true;
		}
	} catch (const std::runtime_error& e) { tlog::error() << "Failed to load training data: " << e.what(); }
}

void Testbed::reset_accumulation(bool due_to_camera_movement, bool immediate_redraw) {
	if (immediate_redraw) {
		redraw_next_frame();
	}

	if (!due_to_camera_movement || !m_dlss) {
		m_windowless_render_surface.reset_accumulation();
		for (auto& view : m_views) {
			view.render_buffer->reset_accumulation();
		}
	}
}

void Testbed::set_visualized_dim(int dim) {
	m_visualized_dimension = dim;
	reset_accumulation();
}

void Testbed::translate_camera(const vec3& rel, const mat3& rot, bool allow_up_down) {
	vec3 movement = rot * rel;
	if (!allow_up_down) {
		movement -= dot(movement, m_up_dir) * m_up_dir;
	}

	m_camera[3] += movement;
	reset_accumulation(true);
}

void Testbed::set_nerf_camera_matrix(const mat4x3& cam) { m_camera = m_nerf.training.dataset.nerf_matrix_to_ngp(cam); }

vec3 Testbed::look_at() const { return view_pos() + view_dir() * m_scale; }

void Testbed::set_look_at(const vec3& pos) { m_camera[3] += pos - look_at(); }

void Testbed::set_scale(float scale) {
	auto prev_look_at = look_at();
	m_camera[3] = (view_pos() - prev_look_at) * (scale / m_scale) + prev_look_at;
	m_scale = scale;
}

void Testbed::set_view_dir(const vec3& dir) {
	auto old_look_at = look_at();
	m_camera[0] = normalize(cross(dir, m_up_dir));
	m_camera[1] = normalize(cross(dir, m_camera[0]));
	m_camera[2] = normalize(dir);
	set_look_at(old_look_at);
}

void Testbed::first_training_view() {
	m_nerf.training.view = 0;
	set_camera_to_training_view(m_nerf.training.view);
}

void Testbed::last_training_view() {
	m_nerf.training.view = m_nerf.training.dataset.n_images - 1;
	set_camera_to_training_view(m_nerf.training.view);
}

void Testbed::previous_training_view() {
	if (m_nerf.training.view != 0) {
		m_nerf.training.view -= 1;
	}

	set_camera_to_training_view(m_nerf.training.view);
}

void Testbed::next_training_view() {
	if (m_nerf.training.view != m_nerf.training.dataset.n_images - 1) {
		m_nerf.training.view += 1;
	}

	set_camera_to_training_view(m_nerf.training.view);
}

void Testbed::set_camera_to_training_view(int trainview) {
	auto old_look_at = look_at();
	m_camera = m_smoothed_camera = get_xform_given_rolling_shutter(
		m_nerf.training.transforms[trainview], m_nerf.training.dataset.metadata[trainview].rolling_shutter, vec2{0.5f, 0.5f}, 0.0f
	);
	m_relative_focal_length = m_nerf.training.dataset.metadata[trainview].focal_length /
		(float)m_nerf.training.dataset.metadata[trainview].resolution[m_fov_axis];
	m_scale = std::max(dot(old_look_at - view_pos(), view_dir()), 0.1f);

	m_render_with_lens_distortion = true;
	m_render_lens = m_nerf.training.dataset.metadata[trainview].lens;
	if (!m_render_lens.supports_dlss()) {
		m_dlss = false;
	}

	m_screen_center = vec2(1.0f) - m_nerf.training.dataset.metadata[trainview].principal_point;
	m_nerf.training.view = trainview;

	reset_accumulation(true);
}

void Testbed::reset_camera() {
	m_fov_axis = 1;
	m_zoom = 1.0f;
	m_screen_center = vec2(0.5f);

	set_fov(50.625f);
	m_scale = 1.5f;

	m_camera = m_default_camera;
	m_camera[3] -= m_scale * view_dir();

	m_smoothed_camera = m_camera;
	m_sun_dir = normalize(vec3(1.0f));

	reset_accumulation();
}

void Testbed::set_train(bool mtrain) {
	if (m_train && !mtrain && m_max_level_rand_training) {
		set_max_level(1.f);
	}
	m_train = mtrain;
}

void Testbed::compute_and_save_marching_cubes_mesh(const fs::path& filename, ivec3 res3d, BoundingBox aabb, float thresh, bool unwrap_it) {
	mat3 render_aabb_to_local = mat3::identity();
	if (aabb.is_empty()) {
		aabb = m_testbed_mode == ETestbedMode::Nerf ? m_render_aabb : m_aabb;
		render_aabb_to_local = m_render_aabb_to_local;
	}
	marching_cubes(res3d, aabb, render_aabb_to_local, thresh);
	save_mesh(
		m_mesh.verts,
		m_mesh.vert_normals,
		m_mesh.vert_colors,
		m_mesh.indices,
		filename,
		unwrap_it,
		m_nerf.training.dataset.scale,
		m_nerf.training.dataset.offset
	);
}

ivec3 Testbed::compute_and_save_png_slices(
	const fs::path& filename, int res, BoundingBox aabb, float thresh, float density_range, bool flip_y_and_z_axes
) {
	mat3 render_aabb_to_local = mat3::identity();
	if (aabb.is_empty()) {
		aabb = m_testbed_mode == ETestbedMode::Nerf ? m_render_aabb : m_aabb;
		render_aabb_to_local = m_render_aabb_to_local;
	}
	if (thresh == std::numeric_limits<float>::max()) {
		thresh = m_mesh.thresh;
	}
	float range = density_range;
	auto res3d = get_marching_cubes_res(res, aabb);

	std::string fname = fmt::format(".density_slices_{}x{}x{}.png", res3d.x, res3d.y, res3d.z);
	// Density slicing is disabled in the Nerf-only slimmed build.
	return res3d;
}

fs::path Testbed::root_dir() {
	if (m_root_dir.empty()) {
		set_root_dir(discover_root_dir());
	}

	return m_root_dir;
}

void Testbed::set_root_dir(const fs::path& dir) {
	m_root_dir = dir;

	auto rtc_cache_dir = root_dir() / "rtc" / "cache";
	rtc_set_cache_dir(rtc_cache_dir.exists() ? rtc_cache_dir.str() : "");
}

bool Testbed::jit_fusion() {
	bool val = true;
	bool any_on = false;
	bool any_network = false;
	for (const auto& device : m_devices) {
		if (device.network()) {
			bool device_on = device.network()->jit_fusion();
			val &= device_on;
			any_on |= device_on;
			any_network = true;
		}
	}

	if (!any_network) {
		return m_jit_fusion;
	}

	if (val != any_on) {
		set_jit_fusion(false);
	}

	m_jit_fusion = val;
	return val;
}

void Testbed::set_jit_fusion(bool val) {
	m_jit_fusion = val;
	for (auto& device : m_devices) {
		if (device.network()) {
			device.network()->set_jit_fusion(val);
		}
	}
}

inline float linear_to_db(float x) { return -10.f * logf(x) / logf(10.f); }

template <typename T> void Testbed::dump_parameters_as_images(const T* params, const std::string& filename_base) {
	if (!m_network) {
		return;
	}

	size_t non_layer_params_width = 2048;

	size_t layer_params = 0;
	for (auto size : m_network->layer_sizes()) {
		layer_params += size.first * size.second;
	}

	size_t n_params = m_network->n_params();
	size_t n_non_layer_params = n_params - layer_params;

	std::vector<T> params_cpu_network_precision(layer_params + next_multiple(n_non_layer_params, non_layer_params_width));
	std::vector<float> params_cpu(params_cpu_network_precision.size(), 0.0f);
	CUDA_CHECK_THROW(cudaMemcpy(params_cpu_network_precision.data(), params, n_params * sizeof(T), cudaMemcpyDeviceToHost));

	for (size_t i = 0; i < n_params; ++i) {
		params_cpu[i] = (float)params_cpu_network_precision[i];
	}

	size_t offset = 0;
	size_t layer_id = 0;
	for (auto size : m_network->layer_sizes()) {
		save_exr(params_cpu.data() + offset, size.second, size.first, 1, 1, fmt::format("{}-layer-{}.exr", filename_base, layer_id).c_str());
		offset += size.first * size.second;
		++layer_id;
	}

	if (n_non_layer_params > 0) {
		std::string filename = fmt::format("{}-non-layer.exr", filename_base);
		save_exr(params_cpu.data() + offset, non_layer_params_width, n_non_layer_params / non_layer_params_width, 1, 1, filename.c_str());
	}
}

template void Testbed::dump_parameters_as_images<__half>(const __half*, const std::string&);
template void Testbed::dump_parameters_as_images<float>(const float*, const std::string&);

mat4x3 Testbed::crop_box(bool nerf_space) const {
	vec3 cen = transpose(m_render_aabb_to_local) * m_render_aabb.center();
	vec3 radius = m_render_aabb.diag() * 0.5f;
	vec3 x = row(m_render_aabb_to_local, 0) * radius.x;
	vec3 y = row(m_render_aabb_to_local, 1) * radius.y;
	vec3 z = row(m_render_aabb_to_local, 2) * radius.z;
	mat4x3 rv;
	rv[0] = x;
	rv[1] = y;
	rv[2] = z;
	rv[3] = cen;
	if (nerf_space) {
		rv = m_nerf.training.dataset.ngp_matrix_to_nerf(rv, true);
	}
	return rv;
}

void Testbed::set_crop_box(mat4x3 m, bool nerf_space) {
	if (nerf_space) {
		m = m_nerf.training.dataset.nerf_matrix_to_ngp(m, true);
	}

	vec3 radius{length(m[0]), length(m[1]), length(m[2])};
	vec3 cen(m[3]);

	m_render_aabb_to_local = row(m_render_aabb_to_local, 0, m[0] / radius.x);
	m_render_aabb_to_local = row(m_render_aabb_to_local, 1, m[1] / radius.y);
	m_render_aabb_to_local = row(m_render_aabb_to_local, 2, m[2] / radius.z);
	cen = m_render_aabb_to_local * cen;
	m_render_aabb.min = cen - radius;
	m_render_aabb.max = cen + radius;
}

std::vector<vec3> Testbed::crop_box_corners(bool nerf_space) const {
	mat4x3 m = crop_box(nerf_space);
	std::vector<vec3> rv(8);
	for (int i = 0; i < 8; ++i) {
		rv[i] = m * vec4{(i & 1) ? 1.f : -1.f, (i & 2) ? 1.f : -1.f, (i & 4) ? 1.f : -1.f, 1.f};
		/* debug print out corners to check math is all lined up */
		if (0) {
			tlog::info() << rv[i].x << "," << rv[i].y << "," << rv[i].z << " [" << i << "]";
			vec3 mn = m_render_aabb.min;
			vec3 mx = m_render_aabb.max;
			mat3 m = transpose(m_render_aabb_to_local);
			vec3 a;

			a.x = (i & 1) ? mx.x : mn.x;
			a.y = (i & 2) ? mx.y : mn.y;
			a.z = (i & 4) ? mx.z : mn.z;
			a = m * a;
			if (nerf_space) {
				a = m_nerf.training.dataset.ngp_position_to_nerf(a);
			}
			tlog::info() << a.x << "," << a.y << "," << a.z << " [" << i << "]";
		}
	}
	return rv;
}

std::unique_ptr<CameraPredictor> create_camera_predictor(ECameraPredictionMode mode, float smoothing) {
	switch (mode) {
		case ECameraPredictionMode::None: return std::make_unique<StationaryCameraPredictor>();
		case ECameraPredictionMode::MatLogLinear: return std::make_unique<MatLogCameraPredictor>(false, false, smoothing);
		case ECameraPredictionMode::MatLogQuadratic: return std::make_unique<MatLogCameraPredictor>(true, false, smoothing);
		case ECameraPredictionMode::Se3LogLinear: return std::make_unique<MatLogCameraPredictor>(false, true, smoothing);
		case ECameraPredictionMode::Se3LogQuadratic: return std::make_unique<MatLogCameraPredictor>(true, true, smoothing);
	}

	return nullptr;
}




__global__ void to_8bit_color_kernel(ivec2 resolution, EColorSpace output_color_space, cudaSurfaceObject_t surface, uint8_t* result) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	vec4 color;
	surf2Dread((float4*)&color, surface, x * sizeof(float4), y);

	if (output_color_space == EColorSpace::Linear) {
		color.rgb() = linear_to_srgb(color.rgb());
	}

	for (uint32_t i = 0; i < 3; ++i) {
		result[(x + resolution.x * y) * 3 + i] = (uint8_t)(clamp(color[i], 0.0f, 1.0f) * 255.0f + 0.5f);
	}
}

void Testbed::prepare_next_camera_path_frame() {
	if (!m_camera_path.rendering) {
		return;
	}

	// If we're rendering a video, we'd like to accumulate multiple spp
	// for motion blur. Hence dump the frame once the target spp has been reached
	// and only reset _then_.
	if (m_views.front().render_buffer->spp() == m_camera_path.render_settings.spp) {
		auto tmp_dir = fs::path{"tmp"};
		if (!tmp_dir.exists()) {
			if (!fs::create_directory(tmp_dir)) {
				m_camera_path.rendering = false;
				tlog::error() << "Failed to create temporary directory 'tmp' to hold rendered images.";
				return;
			}
		}

		ivec2 res = m_views.front().render_buffer->out_resolution();
		const dim3 threads = {16, 8, 1};
		const dim3 blocks = {div_round_up((uint32_t)res.x, threads.x), div_round_up((uint32_t)res.y, threads.y), 1};

		GPUMemory<uint8_t> image_data(product(res) * 3);
		to_8bit_color_kernel<<<blocks, threads>>>(
			res,
			EColorSpace::SRGB, // the GUI always renders in SRGB
			m_views.front().render_buffer->surface(),
			image_data.data()
		);

		m_render_futures.emplace_back(
			m_thread_pool.enqueue_task([image_data = std::move(image_data), frame_idx = m_camera_path.render_frame_idx++, res, tmp_dir] {
				std::vector<uint8_t> cpu_image_data(image_data.size());
				CUDA_CHECK_THROW(cudaMemcpy(cpu_image_data.data(), image_data.data(), image_data.bytes(), cudaMemcpyDeviceToHost));
				write_stbi(tmp_dir / fmt::format("{:06d}.jpg", frame_idx), res.x, res.y, 3, cpu_image_data.data(), 100);
			})
		);

		reset_accumulation(true);

		if (m_camera_path.render_frame_idx == m_camera_path.render_settings.n_frames()) {
			m_camera_path.rendering = false;

			wait_all(m_render_futures);
			m_render_futures.clear();

			tlog::success() << "Finished rendering '.jpg' video frames to '" << tmp_dir << "'. Assembling them into a video next.";

			fs::path ffmpeg = "ffmpeg";

#ifdef _WIN32
			// Under Windows, try automatically downloading FFmpeg binaries if they don't exist
			if (system(fmt::format("where {} >nul 2>nul", ffmpeg.str()).c_str()) != 0) {
				fs::path dir = root_dir();
				if ((dir / "external" / "ffmpeg").exists()) {
					for (const auto& path : fs::directory{dir / "external" / "ffmpeg"}) {
						ffmpeg = path / "bin" / "ffmpeg.exe";
					}
				}

				if (!ffmpeg.exists()) {
					tlog::info() << "FFmpeg not found. Downloading FFmpeg...";
					do_system((dir / "scripts" / "download_ffmpeg.bat").str());
				}

				for (const auto& path : fs::directory{dir / "external" / "ffmpeg"}) {
					ffmpeg = path / "bin" / "ffmpeg.exe";
				}

				if (!ffmpeg.exists()) {
					tlog::warning() << "FFmpeg download failed. Trying system-wide FFmpeg.";
				}
			}
#endif

			auto ffmpeg_command = fmt::format(
				"{} -loglevel error -y -framerate {} -i tmp/%06d.jpg -c:v libx264 -preset slow -crf {} -pix_fmt yuv420p \"{}\"",
				ffmpeg.str(),
				m_camera_path.render_settings.fps,
				// Quality goes from 0 to 10. This conversion to CRF means a quality of 10
				// is a CRF of 17 and a quality of 0 a CRF of 27, which covers the "sane"
				// range of x264 quality settings according to the FFmpeg docs:
				// https://trac.ffmpeg.org/wiki/Encode/H.264
				27 - m_camera_path.render_settings.quality,
				m_camera_path.render_settings.filename
			);
			int ffmpeg_result = do_system(ffmpeg_command);
			if (ffmpeg_result == 0) {
				tlog::success() << "Saved video '" << m_camera_path.render_settings.filename << "'";
			} else if (ffmpeg_result == -1) {
				tlog::error() << "Video could not be assembled: FFmpeg not found.";
			} else {
				tlog::error() << "Video could not be assembled: FFmpeg failed";
			}

			clear_tmp_dir();
		}
	}

	const auto& rs = m_camera_path.render_settings;
	m_camera_path.play_time = (float)((double)m_camera_path.render_frame_idx / (double)rs.n_frames());

	if (m_views.front().render_buffer->spp() == 0) {
		set_camera_from_time(m_camera_path.play_time);
		apply_camera_smoothing(rs.frame_milliseconds());

		auto smoothed_camera_backup = m_smoothed_camera;

		// Compute the camera for the next frame in order to be able to compute motion blur
		// between it and the current one.
		set_camera_from_time(m_camera_path.play_time + 1.0f / rs.n_frames());
		apply_camera_smoothing(rs.frame_milliseconds());

		m_camera_path.render_frame_end_camera = m_smoothed_camera;

		// Revert camera such that the next frame will be computed correctly
		// (Start camera of next frame should be the same as end camera of this frame)
		set_camera_from_time(m_camera_path.play_time);
		m_smoothed_camera = smoothed_camera_backup;
	}
}


void Testbed::train_and_render(bool skip_rendering) {
	if (m_train) {
		train(m_training_batch_size);
	}

	// If we don't have a trainer, as can happen when having loaded training data or changed modes without having
	// explicitly loaded a new neural network.
	if (
		m_testbed_mode != ETestbedMode::None &&
		!m_network
	) {
		reload_network_from_file();
		if (!m_network) {
			throw std::runtime_error{"Unable to reload neural network."};
		}
	}

	if (m_mesh.optimize_mesh) {
		optimise_mesh_step(1);
	}

	// Don't do any smoothing here if a camera path is being rendered. It'll take care
	// of the smoothing on its own.
	float frame_ms = m_camera_path.rendering ? 0.0f : m_frame_ms.val();
	apply_camera_smoothing(frame_ms);

	if (!m_render || skip_rendering) {
		return;
	}

	auto start = std::chrono::steady_clock::now();
	ScopeGuard timing_guard{[&]() {
		m_render_ms.update(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count());
	}};

	if (frobenius_norm(m_smoothed_camera - m_camera) < 0.001f) {
		m_smoothed_camera = m_camera;
	} else if (!m_camera_path.rendering) {
		reset_accumulation(true);
	}

	if (m_autofocus) {
		autofocus();
	}

	Lens lens = m_render_with_lens_distortion ? m_render_lens : Lens{};


	CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
}

mat4x3 Testbed::view_camera(size_t view_idx) const {
	if (m_views.size() <= view_idx) {
		throw std::runtime_error{fmt::format("View #{} does not exist.", view_idx)};
	}

	auto& view = m_views.at(view_idx);
	return view.camera0;
}

mat4x3 Testbed::predict_view_camera(size_t view_idx, float dt) const {
	if (m_views.size() <= view_idx) {
		throw std::runtime_error{fmt::format("View #{} does not exist.", view_idx)};
	}

	auto& view = m_views.at(view_idx);
	return view.camera_predictor->predict_camera(dt);
}

void Testbed::set_camera_prediction_mode(ECameraPredictionMode mode) {
	for (auto& view : m_views) {
		view.camera_predictor = create_camera_predictor(mode, 0.01f);
	}

	m_camera_prediction_mode = mode;
}

ECameraPredictionMode Testbed::camera_prediction_mode() const { return m_camera_prediction_mode; }



bool Testbed::frame() {
	// Render against the trained neural network. If we're training and already close to convergence,
	// we can skip rendering if the scene camera doesn't change
	uint32_t n_to_skip = m_train ? clamp(m_training_step / 16u, 15u, 255u) : 0;
	if (m_render_skip_due_to_lack_of_camera_movement_counter > n_to_skip) {
		m_render_skip_due_to_lack_of_camera_movement_counter = 0;
	}
	bool skip_rendering = m_render_skip_due_to_lack_of_camera_movement_counter++ != 0;

	if (!m_dlss && m_max_spp > 0 && !m_views.empty() && m_views.front().render_buffer->spp() >= m_max_spp) {
		skip_rendering = true;
		if (!m_train) {
			std::this_thread::sleep_for(1ms);
		}
	}

	if (
		m_camera_path.rendering
	) {
		prepare_next_camera_path_frame();
		skip_rendering = false;
	}

	if (m_record_camera_path && !m_views.empty()) {
		m_camera_path.spline_order = 1;
		m_camera_path.duration_seconds += m_frame_ms.val() / 1000.0f;
		m_camera_path.add_camera(
			m_views[0].camera0,
			m_slice_plane_z,
			m_scale,
			focal_length_to_fov(1.0f, m_views[0].relative_focal_length[m_fov_axis]),
			m_aperture_size,
			m_bounding_radius,
			m_camera_path.duration_seconds
		);

		m_camera_path.keyframe_subsampling = (int)m_camera_path.keyframes.size();
		m_camera_path.editing_kernel_type = EEditingKernel::Gaussian;
	}

	train_and_render(skip_rendering);
	return true;
}

fs::path Testbed::training_data_path() const { return m_data_path.with_extension("training"); }

void Testbed::apply_camera_smoothing(float elapsed_ms) {
	// Ensure our camera rotation remains an orthogonal matrix as numeric
	// errors accumulate across frames.
	m_camera = orthogonalize(m_camera);

	if (m_camera_smoothing) {
		float decay = std::pow(0.02f, elapsed_ms / 1000.0f);
		m_smoothed_camera = orthogonalize(camera_log_lerp(m_smoothed_camera, m_camera, 1.0f - decay));
	} else {
		m_smoothed_camera = m_camera;
	}
}

CameraKeyframe Testbed::copy_camera_to_keyframe() const {
	return CameraKeyframe(m_camera, m_slice_plane_z, m_scale, fov(), m_aperture_size, 0.0f);
}

void Testbed::set_camera_from_keyframe(const CameraKeyframe& k) {
	m_camera = k.m();
	m_slice_plane_z = k.slice;
	m_scale = k.scale;
	set_fov(k.fov);
	m_aperture_size = k.aperture_size;
}

void Testbed::set_camera_from_time(float t) {
	if (m_camera_path.keyframes.empty()) {
		return;
	}

	set_camera_from_keyframe(m_camera_path.eval_camera_path(t));
}

void Testbed::update_loss_graph() { m_loss_graph[m_loss_graph_samples++ % m_loss_graph.size()] = std::log(m_loss_scalar.val()); }

uint32_t Testbed::n_dimensions_to_visualize() const { return m_network ? m_network->width(m_visualized_layer) : 0; }

float Testbed::fov() const { return focal_length_to_fov(1.0f, m_relative_focal_length[m_fov_axis]); }

void Testbed::set_fov(float val) { m_relative_focal_length = vec2(fov_to_focal_length(1, val)); }

vec2 Testbed::fov_xy() const { return focal_length_to_fov(ivec2(1), m_relative_focal_length); }

void Testbed::set_fov_xy(const vec2& val) { m_relative_focal_length = fov_to_focal_length(ivec2(1), val); }

size_t Testbed::n_params() { return m_network ? m_network->n_params() : 0; }

size_t Testbed::n_encoding_params() { return n_params() - first_encoder_param(); }

size_t Testbed::first_encoder_param() {
	if (!m_network) {
		return 0;
	}

	auto layer_sizes = m_network->layer_sizes();
	size_t first_encoder = 0;
	for (auto size : layer_sizes) {
		first_encoder += size.first * size.second;
	}

	return first_encoder;
}

uint32_t Testbed::network_width(uint32_t layer) const { return m_network ? m_network->width(layer) : 0; }

uint32_t Testbed::network_num_forward_activations() const { return m_network ? m_network->num_forward_activations() : 0; }

void Testbed::set_max_level(float maxlevel) {
	if (!m_network) {
		return;
	}
	auto hg_enc = dynamic_cast<MultiLevelEncoding<network_precision_t>*>(m_encoding.get());
	if (hg_enc) {
		hg_enc->set_max_level(maxlevel);
	}

	reset_accumulation();
}

void Testbed::set_visualized_layer(int layer) {
	m_visualized_layer = layer;
	m_visualized_dimension = std::max(-1, std::min(m_visualized_dimension, (int)network_width(layer) - 1));
	reset_accumulation();
}

ELossType Testbed::string_to_loss_type(const std::string& str) {
	if (equals_case_insensitive(str, "L2")) {
		return ELossType::L2;
	} else if (equals_case_insensitive(str, "RelativeL2")) {
		return ELossType::RelativeL2;
	} else if (equals_case_insensitive(str, "L1")) {
		return ELossType::L1;
	} else if (equals_case_insensitive(str, "Mape")) {
		return ELossType::Mape;
	} else if (equals_case_insensitive(str, "Smape")) {
		return ELossType::Smape;
	} else if (equals_case_insensitive(str, "Huber") || equals_case_insensitive(str, "SmoothL1")) {
		// Legacy: we used to refer to the Huber loss (L2 near zero, L1 further away) as "SmoothL1".
		return ELossType::Huber;
	} else if (equals_case_insensitive(str, "LogL1")) {
		return ELossType::LogL1;
	} else {
		throw std::runtime_error{"Unknown loss type."};
	}
}

Testbed::NetworkDims Testbed::network_dims() const {
	switch (m_testbed_mode) {
		case ETestbedMode::Nerf: return network_dims_nerf(); break;
		// case ETestbedMode::Sdf: return network_dims_sdf(); break;
		// case ETestbedMode::Image: return network_dims_image(); break;
		// case ETestbedMode::Volume: return network_dims_volume(); break;
		default: throw std::runtime_error{"Invalid mode."};
	}
}

void Testbed::reset_network(bool clear_density_grid) {
	m_rng = default_rng_t{m_seed};

	// Start with a low rendering resolution and gradually ramp up
	m_render_ms.set(10000);

	reset_accumulation();


	m_nerf.training.counters_rgb.rays_per_batch = 1 << 12;
	m_nerf.training.counters_rgb.measured_batch_size_before_compaction = 0;
	m_nerf.training.n_steps_since_cam_update = 0;
	m_nerf.training.n_steps_since_error_map_update = 0;
	m_nerf.training.n_rays_since_error_map_update = 0;
	m_nerf.training.n_steps_between_error_map_updates = 128;
	m_nerf.training.error_map.is_cdf_valid = false;
	m_nerf.training.density_grid_rng = default_rng_t{m_rng.next_uint()};

	m_nerf.training.reset_camera_extrinsics();

	if (clear_density_grid) {
		m_nerf.density_grid.memset(0);
		m_nerf.density_grid_bitfield.memset(0);

		set_all_devices_dirty();
	}

	m_loss_graph_samples = 0;

	// Default config
	json config = m_network_config;

	json& encoding_config = config["encoding"];
	json& loss_config = config["loss"];
	json& optimizer_config = config["optimizer"];
	json& network_config = config["network"];

	// If the network config is incomplete, avoid doing further work.
	/*
	if (config.is_null() || encoding_config.is_null() || loss_config.is_null() || optimizer_config.is_null() ||
	network_config.is_null()) { return;
	}
	*/

	auto dims = network_dims();

	if (m_testbed_mode == ETestbedMode::Nerf) {
		m_nerf.training.loss_type = string_to_loss_type(loss_config.value("otype", "L2"));

		// Some of the Nerf-supported losses are not supported by Loss,
		// so just create a dummy L2 loss there. The NeRF code path will bypass
		// the Loss in any case.
		loss_config["otype"] = "L2";
	}

	// Automatically determine certain parameters if we're dealing with the (hash)grid encoding
	std::string encoding_otype = to_lower(encoding_config.value("otype", "OneBlob"));
	if (encoding_otype.find("grid") != std::string::npos || encoding_otype.find("permuto") != std::string::npos) {
		encoding_config["n_pos_dims"] = dims.n_pos;

		m_n_features_per_level = encoding_config.value("n_features_per_level", 2u);

		if (encoding_config.contains("n_features") && encoding_config["n_features"] > 0) {
			m_n_levels = (uint32_t)encoding_config["n_features"] / m_n_features_per_level;
		} else {
			m_n_levels = encoding_config.value("n_levels", 16u);
		}

		m_level_stats.resize(m_n_levels);
		m_first_layer_column_stats.resize(m_n_levels);

		const uint32_t log2_hashmap_size = encoding_config.value("log2_hashmap_size", 15);

		m_base_grid_resolution = encoding_config.value("base_resolution", 0);
		if (!m_base_grid_resolution) {
			m_base_grid_resolution = 1u << ((log2_hashmap_size) / dims.n_pos);
			encoding_config["base_resolution"] = m_base_grid_resolution;
		}

		float desired_resolution = 2048.0f; // Desired resolution of the finest hashgrid level over the unit cube

		// Automatically determine suitable per_level_scale
		m_per_level_scale = encoding_config.value("per_level_scale", 0.0f);
		if (m_per_level_scale <= 0.0f && m_n_levels > 1) {
			m_per_level_scale = std::exp(
				std::log(desired_resolution * (float)m_nerf.training.dataset.aabb_scale / (float)m_base_grid_resolution) / (m_n_levels - 1)
			);
			encoding_config["per_level_scale"] = m_per_level_scale;
		}

		tlog::info() << "MultiLevelEncoding:"
					 << " type=" << encoding_otype << " Nmin=" << m_base_grid_resolution << " b=" << m_per_level_scale
					 << " F=" << m_n_features_per_level << " T=2^" << log2_hashmap_size << " L=" << m_n_levels;
	}

	m_loss.reset(create_loss<network_precision_t>(loss_config));
	m_optimizer.reset(create_optimizer<network_precision_t>(optimizer_config));

	size_t n_encoding_params = 0;
	if (m_testbed_mode == ETestbedMode::Nerf) {
		m_nerf.training.cam_exposure.resize(m_nerf.training.dataset.n_images, AdamOptimizer<vec3>(1e-3f));
		m_nerf.training.cam_pos_offset.resize(m_nerf.training.dataset.n_images, AdamOptimizer<vec3>(1e-4f));
		m_nerf.training.cam_rot_offset.resize(m_nerf.training.dataset.n_images, RotationAdamOptimizer(1e-4f));
		m_nerf.training.cam_focal_length_offset = AdamOptimizer<vec2>(1e-5f);

		m_nerf.reset_extra_dims(m_rng);

		json& dir_encoding_config = config["dir_encoding"];
		json& rgb_network_config = config["rgb_network"];

		uint32_t n_dir_dims = 3;
		uint32_t n_extra_dims = m_nerf.training.dataset.n_extra_dims();

		// Instantiate an additional model for each auxiliary GPU
		for (auto& device : m_devices) {
			device.set_nerf_network(
				std::make_shared<NerfNetwork<network_precision_t>>(
					dims.n_pos,
					n_dir_dims,
					n_extra_dims,
					dims.n_pos + 1, // The offset of 1 comes from the dt member variable of NerfCoordinate. HACKY
					encoding_config,
					dir_encoding_config,
					network_config,
					rgb_network_config
				)
			);
		}

		m_network = m_nerf_network = primary_device().nerf_network();

		m_encoding = m_nerf_network->pos_encoding();
		n_encoding_params = m_encoding->n_params() + m_nerf_network->dir_encoding()->n_params();

		tlog::info() << "Density model: " << dims.n_pos << "--[" << std::string(encoding_config["otype"]) << "]-->"
					 << m_nerf_network->pos_encoding()->padded_output_width() << "--[" << std::string(network_config["otype"])
					 << "(neurons=" << (int)network_config["n_neurons"] << ",layers=" << ((int)network_config["n_hidden_layers"] + 2) << ")"
					 << "]-->" << 1;

		tlog::info() << "Color model:   " << n_dir_dims << "--[" << std::string(dir_encoding_config["otype"]) << "]-->"
					 << m_nerf_network->dir_encoding()->padded_output_width() << "+" << network_config.value("n_output_dims", 16u) << "--["
					 << std::string(rgb_network_config["otype"]) << "(neurons=" << (int)rgb_network_config["n_neurons"]
					 << ",layers=" << ((int)rgb_network_config["n_hidden_layers"] + 2) << ")"
					 << "]-->" << 3;

		// Create distortion map model
		{
			json& distortion_map_optimizer_config = config.contains("distortion_map") && config["distortion_map"].contains("optimizer") ?
				config["distortion_map"]["optimizer"] :
				optimizer_config;

			m_distortion.resolution = ivec2(32);
			if (config.contains("distortion_map") && config["distortion_map"].contains("resolution")) {
				from_json(config["distortion_map"]["resolution"], m_distortion.resolution);
			}
			m_distortion.map = std::make_shared<TrainableBuffer<2, 2, float>>(m_distortion.resolution);
			m_distortion.optimizer.reset(create_optimizer<float>(distortion_map_optimizer_config));
			m_distortion.trainer = std::make_shared<Trainer<float, float>>(
				m_distortion.map, m_distortion.optimizer, std::shared_ptr<Loss<float>>{create_loss<float>(loss_config)}, m_seed
			);
		}
	} else {
		uint32_t alignment = network_config.contains("otype") &&
				(equals_case_insensitive(network_config["otype"], "FullyFusedMLP") ||
				 equals_case_insensitive(network_config["otype"], "MegakernelMLP")) ?
			16u :
			8u;

		m_encoding.reset(create_encoding<network_precision_t>(dims.n_input, encoding_config));

		for (auto& device : m_devices) {
			device.set_network(std::make_shared<NetworkWithInputEncoding<network_precision_t>>(m_encoding, dims.n_output, network_config));
		}

		m_network = primary_device().network();

		n_encoding_params = m_encoding->n_params();

		tlog::info() << "Model:         " << dims.n_input << "--[" << std::string(encoding_config["otype"]) << "]-->"
					 << m_encoding->padded_output_width() << "--[" << std::string(network_config["otype"])
					 << "(neurons=" << (int)network_config["n_neurons"] << ",layers=" << ((int)network_config["n_hidden_layers"] + 2) << ")"
					 << "]-->" << dims.n_output;
	}

	// Internally calls `model->set_jit_fusion` on every device.
	set_jit_fusion(m_jit_fusion);

	size_t n_network_params = m_network->n_params() - n_encoding_params;

	tlog::info() << "  total_encoding_params=" << n_encoding_params << " total_network_params=" << n_network_params;

	m_trainer = std::make_shared<Trainer<float, network_precision_t, network_precision_t>>(m_network, m_optimizer, m_loss, m_seed);

	auto rtc_cache_dir = root_dir() / "rtc" / "cache";
	rtc_set_cache_dir(rtc_cache_dir.exists() ? rtc_cache_dir.str() : "");

	m_training_step = 0;
	m_training_start_time_point = std::chrono::steady_clock::now();

	// Create envmap model
	{
		json& envmap_loss_config = config.contains("envmap") && config["envmap"].contains("loss") ? config["envmap"]["loss"] : loss_config;
		json& envmap_optimizer_config = config.contains("envmap") && config["envmap"].contains("optimizer") ? config["envmap"]["optimizer"] :
																											  optimizer_config;

		m_envmap.loss_type = string_to_loss_type(envmap_loss_config.value("otype", "L2"));

		m_envmap.resolution = m_nerf.training.dataset.envmap_resolution;
		m_envmap.envmap = std::make_shared<TrainableBuffer<4, 2, float>>(m_envmap.resolution);
		m_envmap.optimizer.reset(create_optimizer<float>(envmap_optimizer_config));
		m_envmap.trainer = std::make_shared<Trainer<float, float, float>>(
			m_envmap.envmap, m_envmap.optimizer, std::shared_ptr<Loss<float>>{create_loss<float>(envmap_loss_config)}, m_seed
		);

		if (m_nerf.training.dataset.envmap_data.data()) {
			m_envmap.trainer->set_params_full_precision(m_nerf.training.dataset.envmap_data.data(), m_nerf.training.dataset.envmap_data.size());
		}
	}

	set_all_devices_dirty();
}

Testbed::Testbed(ETestbedMode mode) {
	tcnn::set_log_callback([](LogSeverity severity, const std::string& msg) {
		tlog::ESeverity s = tlog::ESeverity::Info;
		switch (severity) {
			case LogSeverity::Info: s = tlog::ESeverity::Info; break;
			case LogSeverity::Debug: s = tlog::ESeverity::Debug; break;
			case LogSeverity::Warning: s = tlog::ESeverity::Warning; break;
			case LogSeverity::Error: s = tlog::ESeverity::Error; break;
			case LogSeverity::Success: s = tlog::ESeverity::Success; break;
			default: break;
		}
		tlog::log(s) << msg;
	});

	if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
		throw std::runtime_error{"Testbed requires CUDA 10.2 or later."};
	}


	// Reset our stream, which was allocated on the originally active device,
	// to make sure it corresponds to the now active device.
	m_stream = {};

	int active_device = cuda_device();
	int active_compute_capability = cuda_compute_capability();
	tlog::success() << fmt::format(
		"Initialized CUDA {}. Active GPU is #{}: {} [{}]", cuda_runtime_version_string(), active_device, cuda_device_name(), active_compute_capability
	);

	if (active_compute_capability < MIN_GPU_ARCH) {
		tlog::warning() << "Insufficient compute capability " << active_compute_capability << " detected.";
		tlog::warning() << "This program was compiled for >=" << MIN_GPU_ARCH << " and may thus behave unexpectedly.";
	}

	m_devices.emplace_back(active_device, true);

	// Multi-GPU is only supported in NeRF mode for now
	int n_devices = cuda_device_count();
	for (int i = 0; i < n_devices; ++i) {
		if (i == active_device) {
			continue;
		}

		if (cuda_compute_capability(i) >= MIN_GPU_ARCH) {
			m_devices.emplace_back(i, false);
		}
	}

	if (m_devices.size() > 1) {
		tlog::success() << "Detected auxiliary GPUs:";
		for (size_t i = 1; i < m_devices.size(); ++i) {
			const auto& device = m_devices[i];
			tlog::success() << "  #" << device.id() << ": " << device.name() << " [" << device.compute_capability() << "]";
		}
	}

	m_network_config = {
		{"loss",      {{"otype", "L2"}}},
		{"optimizer",
		 {
			 {"otype", "Adam"},
			 {"learning_rate", 1e-3},
			 {"beta1", 0.9f},
			 {"beta2", 0.99f},
			 {"epsilon", 1e-15f},
			 {"l2_reg", 1e-6f},
		 }							 },
		{"encoding",
		 {
			 {"otype", "HashGrid"},
			 {"n_levels", 16},
			 {"n_features_per_level", 2},
			 {"log2_hashmap_size", 19},
			 {"base_resolution", 16},
		 }							 },
		{"network",
		 {
			 {"otype", "FullyFusedMLP"},
			 {"n_neurons", 64},
			 {"n_layers", 2},
			 {"activation", "ReLU"},
			 {"output_activation", "None"},
		 }							 },
	};

	set_mode(mode);
	set_exposure(0);
	set_max_level(1.f);

	reset_camera();
}

Testbed::~Testbed() {

	// If any temporary file was created, make sure it's deleted
	clear_tmp_dir();
}

bool Testbed::clear_tmp_dir() {
	wait_all(m_render_futures);
	m_render_futures.clear();

	bool success = true;
	auto tmp_dir = fs::path{"tmp"};
	if (tmp_dir.exists()) {
		if (tmp_dir.is_directory()) {
			for (const auto& path : fs::directory{tmp_dir}) {
				if (path.is_file()) {
					success &= path.remove_file();
				}
			}
		}

		success &= tmp_dir.remove_file();
	}

	return success;
}

void Testbed::train(uint32_t batch_size) {
	if (!m_training_data_available || m_camera_path.rendering) {
		m_train = false;
		return;
	}

	if (m_testbed_mode == ETestbedMode::None) {
		throw std::runtime_error{"Cannot train without a mode."};
	}

	set_all_devices_dirty();

	// If we don't have a trainer, as can happen when having loaded training data or changed modes without having
	// explicitly loaded a new neural network.
	if (!m_trainer) {
		reload_network_from_file();
		if (!m_trainer) {
			throw std::runtime_error{"Unable to create a neural network trainer."};
		}
	}

	if (m_testbed_mode == ETestbedMode::Nerf) {
		if (m_nerf.training.optimize_extra_dims) {
			if (m_nerf.training.dataset.n_extra_learnable_dims == 0) {
				m_nerf.training.dataset.n_extra_learnable_dims = 16;
				reset_network();
			}
		}
	}

	if (!m_dlss) {
		// No immediate redraw necessary
		reset_accumulation(false, false);
	}

	uint32_t n_prep_to_skip = m_testbed_mode == ETestbedMode::Nerf ? clamp(m_training_step / 16u, 1u, 16u) : 1u;
	if (m_training_step % n_prep_to_skip == 0) {
		auto start = std::chrono::steady_clock::now();
		ScopeGuard timing_guard{[&]() {
			m_training_prep_ms.update(
				std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count() / n_prep_to_skip
			);
		}};

		if (m_testbed_mode == ETestbedMode::Nerf) {
			training_prep_nerf(batch_size, m_stream.get());
		} else {
			throw std::runtime_error{"Invalid training mode."};
		}

		CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
	}

	// Find leaf optimizer and update its settings
	json* leaf_optimizer_config = &m_network_config["optimizer"];
	while (leaf_optimizer_config->contains("nested")) {
		leaf_optimizer_config = &(*leaf_optimizer_config)["nested"];
	}
	(*leaf_optimizer_config)["optimize_matrix_params"] = m_train_network;
	(*leaf_optimizer_config)["optimize_non_matrix_params"] = m_train_encoding;
	m_optimizer->update_hyperparams(m_network_config["optimizer"]);

	bool get_loss_scalar = m_training_step % 16 == 0;

	{
		auto start = std::chrono::steady_clock::now();
		ScopeGuard timing_guard{[&]() {
			m_training_ms.update(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count());
		}};

		if (m_testbed_mode == ETestbedMode::Nerf) {
			train_nerf(batch_size, get_loss_scalar, m_stream.get());
		} else {
			throw std::runtime_error{"Invalid training mode."};
		}

		CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
	}

	if (get_loss_scalar) {
		update_loss_graph();
	}
}

vec2 Testbed::calc_focal_length(const ivec2& resolution, const vec2& relative_focal_length, int fov_axis, float zoom) const {
	return relative_focal_length * (float)resolution[fov_axis] * zoom;
}

vec2 Testbed::render_screen_center(const vec2& screen_center) const {
	// see pixel_to_ray for how screen center is used; 0.5, 0.5 is 'normal'. we flip so that it becomes the point in the
	// original image we want to center on.
	return (0.5f - screen_center) * m_zoom + 0.5f;
}

__global__ void dlss_prep_kernel(
	ivec2 resolution,
	uint32_t sample_index,
	vec2 focal_length,
	vec2 screen_center,
	vec3 parallax_shift,
	bool snap_to_pixel_centers,
	float* depth_buffer,
	const float znear,
	const float zfar,
	mat4x3 camera,
	mat4x3 prev_camera,
	cudaSurfaceObject_t depth_surface,
	cudaSurfaceObject_t mvec_surface,
	cudaSurfaceObject_t exposure_surface,
	Foveation foveation,
	Foveation prev_foveation,
	Lens lens
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	uint32_t idx = x + resolution.x * y;

	uint32_t x_orig = x;
	uint32_t y_orig = y;

	const float depth = depth_buffer[idx];
	vec2 mvec = motion_vector(
		sample_index,
		{(int)x, (int)y},
		resolution,
		focal_length,
		camera,
		prev_camera,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		depth,
		foveation,
		prev_foveation,
		lens
	);

	surf2Dwrite(make_float2(mvec.x, mvec.y), mvec_surface, x_orig * sizeof(float2), y_orig);

	// DLSS was trained on games, which presumably used standard normalized device coordinates (ndc)
	// depth buffers. So: convert depth to NDC with reasonable near- and far planes.
	surf2Dwrite(to_ndc_depth(depth, znear, zfar), depth_surface, x_orig * sizeof(float), y_orig);

	// First thread write an exposure factor of 1. Since DLSS will run on tonemapped data,
	// exposure is assumed to already have been applied to DLSS' inputs.
	if (x_orig == 0 && y_orig == 0) {
		surf2Dwrite(1.0f, exposure_surface, 0, 0);
	}
}

__global__ void spherical_checkerboard_kernel(
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera,
	vec2 screen_center,
	vec3 parallax_shift,
	Foveation foveation,
	Lens lens,
	vec4 background_color,
	vec4* frame_buffer
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	Ray ray = pixel_to_ray(
		0,
		{(int)x, (int)y},
		resolution,
		focal_length,
		camera,
		screen_center,
		parallax_shift,
		false,
		0.0f,
		1.0f,
		0.0f,
		foveation,
		{}, // No need for hidden area mask
		lens
	);

	// Blend with checkerboard to break up reprojection weirdness in some VR runtimes
	host_device_swap(ray.d.z, ray.d.y);
	vec2 spherical = dir_to_spherical(normalize(ray.d)) * 32.0f / PI();
	const vec4 dark_gray = {0.5f, 0.5f, 0.5f, 1.0f};
	const vec4 light_gray = {0.55f, 0.55f, 0.55f, 1.0f};
	vec4 checker = fabsf(fmodf(floorf(spherical.x) + floorf(spherical.y), 2.0f)) < 0.5f ? dark_gray : light_gray;

	// Blend background color on top of checkerboard first (checkerboard is meant to be "behind" the background,
	// representing transparency), and then blend the result behind the frame buffer.
	background_color.rgb() = srgb_to_linear(background_color.rgb());
	background_color += (1.0f - background_color.a) * checker;

	uint32_t idx = x + resolution.x * y;
	frame_buffer[idx] += (1.0f - frame_buffer[idx].a) * background_color;
}

void Testbed::render_frame(
	cudaStream_t stream,
	const mat4x3& camera_matrix0,
	const mat4x3& camera_matrix1,
	const mat4x3& prev_camera_matrix,
	const vec2& orig_screen_center,
	const vec2& relative_focal_length,
	const vec4& nerf_rolling_shutter,
	const Foveation& foveation,
	const Foveation& prev_foveation,
	const Lens& lens,
	int visualized_dimension,
	CudaRenderBuffer& render_buffer,
	bool to_srgb,
	CudaDevice* device
) {
	if (!device) {
		device = &primary_device();
	}

	sync_device(render_buffer, *device);

	{
		auto device_guard = use_device(stream, render_buffer, *device);
		render_frame_main(
			*device, camera_matrix0, camera_matrix1, orig_screen_center, relative_focal_length, nerf_rolling_shutter, foveation, lens, visualized_dimension
		);
	}

	render_frame_epilogue(
		stream, camera_matrix0, prev_camera_matrix, orig_screen_center, relative_focal_length, foveation, prev_foveation, lens, render_buffer, to_srgb
	);
}

void Testbed::render_frame_main(
	CudaDevice& device,
	const mat4x3& camera_matrix0,
	const mat4x3& camera_matrix1,
	const vec2& orig_screen_center,
	const vec2& relative_focal_length,
	const vec4& nerf_rolling_shutter,
	const Foveation& foveation,
	const Lens& lens,
	int visualized_dimension
) {
	device.render_buffer_view().clear(device.stream());

	if (!m_network) {
		return;
	}

	vec2 focal_length = calc_focal_length(device.render_buffer_view().resolution, relative_focal_length, m_fov_axis, m_zoom);
	vec2 screen_center = render_screen_center(orig_screen_center);

	switch (m_testbed_mode) {
		case ETestbedMode::Nerf:
			if (!m_render_ground_truth || m_ground_truth_alpha < 1.0f) {
				render_nerf(
					device.stream(),
					device,
					device.render_buffer_view(),
					device.nerf_network(),
					device.data().density_grid_bitfield_ptr,
					focal_length,
					camera_matrix0,
					camera_matrix1,
					nerf_rolling_shutter,
					screen_center,
					foveation,
					lens,
					visualized_dimension
				);
			}
		// 		focal_length,
		// 		camera_matrix0,
		// 		screen_center,
		// 		foveation,
		// 		lens,
		// 		visualized_dimension
		// 	);
		// } break;
		// case ETestbedMode::Image:
		// 	render_image(
		// 		device.stream(), device.render_buffer_view(), focal_length, camera_matrix0, screen_center, foveation, lens, visualized_dimension
		// 	);
		// 	break;
		// case ETestbedMode::Volume:
		// 	render_volume(device.stream(), device.render_buffer_view(), focal_length, camera_matrix0, screen_center, foveation, lens);
		// 	break;
		default:
			// No-op if no mode is active
			break;
	}
}

void Testbed::render_frame_epilogue(
	cudaStream_t stream,
	const mat4x3& camera_matrix0,
	const mat4x3& prev_camera_matrix,
	const vec2& orig_screen_center,
	const vec2& relative_focal_length,
	const Foveation& foveation,
	const Foveation& prev_foveation,
	const Lens& lens,
	CudaRenderBuffer& render_buffer,
	bool to_srgb
) {
	vec2 focal_length = calc_focal_length(render_buffer.in_resolution(), relative_focal_length, m_fov_axis, m_zoom);
	vec2 screen_center = render_screen_center(orig_screen_center);

	render_buffer.set_color_space(m_color_space);
	render_buffer.set_tonemap_curve(m_tonemap_curve);

	// Prepare DLSS data: motion vectors, scaled depth, exposure
	if (render_buffer.dlss()) {
		auto res = render_buffer.in_resolution();

		const dim3 threads = {16, 8, 1};
		const dim3 blocks = {div_round_up((uint32_t)res.x, threads.x), div_round_up((uint32_t)res.y, threads.y), 1};

		dlss_prep_kernel<<<blocks, threads, 0, stream>>>(
			res,
			render_buffer.spp(),
			focal_length,
			screen_center,
			m_parallax_shift,
			m_snap_to_pixel_centers,
			render_buffer.depth_buffer(),
			m_ndc_znear,
			m_ndc_zfar,
			camera_matrix0,
			prev_camera_matrix,
			render_buffer.dlss()->depth(),
			render_buffer.dlss()->mvec(),
			render_buffer.dlss()->exposure(),
			foveation,
			prev_foveation,
			lens
		);

		render_buffer.set_dlss_sharpening(m_dlss_sharpening);
	}

	EColorSpace output_color_space = to_srgb ? EColorSpace::SRGB : EColorSpace::Linear;

	if (m_render_transparency_as_checkerboard) {
		mat4x3 checkerboard_transform = mat4x3::identity();


		auto res = render_buffer.in_resolution();
		const dim3 threads = {16, 8, 1};
		const dim3 blocks = {div_round_up((uint32_t)res.x, threads.x), div_round_up((uint32_t)res.y, threads.y), 1};
		spherical_checkerboard_kernel<<<blocks, threads, 0, stream>>>(
			res,
			focal_length,
			checkerboard_transform,
			screen_center,
			m_parallax_shift,
			foveation,
			lens,
			m_background_color,
			render_buffer.frame_buffer()
		);
	}

	render_buffer.accumulate(m_exposure, stream);
	render_buffer.tonemap(m_exposure, m_background_color, output_color_space, m_ndc_znear, m_ndc_zfar, m_snap_to_pixel_centers, stream);

	if (m_testbed_mode == ETestbedMode::Nerf) {
		// Overlay the ground truth image if requested
		if (m_render_ground_truth) {
			const auto& metadata = m_nerf.training.dataset.metadata[m_nerf.training.view];
			if (m_ground_truth_render_mode == EGroundTruthRenderMode::Shade) {
				render_buffer.overlay_image(
					m_ground_truth_alpha,
					vec3(m_exposure) + m_nerf.training.cam_exposure[m_nerf.training.view].variable(),
					m_background_color,
					output_color_space,
					metadata.pixels,
					metadata.image_data_type,
					metadata.resolution,
					m_fov_axis,
					m_zoom,
					vec2(0.5f),
					stream
				);
			} else if (m_ground_truth_render_mode == EGroundTruthRenderMode::Depth && metadata.depth) {
				render_buffer.overlay_depth(
					m_ground_truth_alpha,
					metadata.depth,
					1.0f / m_nerf.training.dataset.scale,
					metadata.resolution,
					m_fov_axis,
					m_zoom,
					vec2(0.5f),
					stream
				);
			}
		}

		// Visualize the accumulated error map if requested
		if (m_nerf.training.render_error_overlay) {
			const float* err_data = m_nerf.training.error_map.data.data();
			ivec2 error_map_res = m_nerf.training.error_map.resolution;
			if (m_render_ground_truth) {
				err_data = m_nerf.training.dataset.sharpness_data.data();
				error_map_res = m_nerf.training.dataset.sharpness_resolution;
			}
			size_t emap_size = error_map_res.x * error_map_res.y;
			err_data += emap_size * m_nerf.training.view;

			GPUMemory<float> average_error;
			average_error.enlarge(1);
			average_error.memset(0);
			const float* aligned_err_data_s = (const float*)(((size_t)err_data) & ~15);
			const float* aligned_err_data_e = (const float*)(((size_t)(err_data + emap_size)) & ~15);
			size_t reduce_size = aligned_err_data_e - aligned_err_data_s;
			reduce_sum(
				aligned_err_data_s,
				[reduce_size] __device__(float val) { return max(val, 0.f) / (reduce_size); },
				average_error.data(),
				reduce_size,
				stream
			);
			const auto& metadata = m_nerf.training.dataset.metadata[m_nerf.training.view];
			render_buffer.overlay_false_color(
				metadata.resolution,
				to_srgb,
				m_fov_axis,
				stream,
				err_data,
				error_map_res,
				average_error.data(),
				m_nerf.training.error_overlay_brightness,
				m_render_ground_truth
			);
		}
	}

}

float Testbed::get_depth_from_renderbuffer(const CudaRenderBuffer& render_buffer, const vec2& uv) {
	if (!render_buffer.depth_buffer()) {
		return m_scale;
	}

	float depth;
	auto res = render_buffer.in_resolution();
	ivec2 depth_pixel = clamp(ivec2(uv * vec2(res)), 0, res - 1);

	CUDA_CHECK_THROW(
		cudaMemcpy(&depth, render_buffer.depth_buffer() + depth_pixel.x + depth_pixel.y * res.x, sizeof(float), cudaMemcpyDeviceToHost)
	);
	return depth;
}

vec3 Testbed::get_3d_pos_from_pixel(const CudaRenderBuffer& render_buffer, const vec2& pixel) {
	float depth = get_depth_from_renderbuffer(render_buffer, pixel / vec2(m_window_res));
	auto ray = pixel_to_ray_pinhole(
		0,
		ivec2(pixel),
		m_window_res,
		calc_focal_length(m_window_res, m_relative_focal_length, m_fov_axis, m_zoom),
		m_smoothed_camera,
		render_screen_center(m_screen_center)
	);
	return ray(depth);
}

void Testbed::autofocus() {
	float new_slice_plane_z = std::max(dot(view_dir(), m_autofocus_target - view_pos()), 0.1f) - m_scale;
	if (new_slice_plane_z != m_slice_plane_z) {
		m_slice_plane_z = new_slice_plane_z;
		if (m_aperture_size != 0.0f) {
			reset_accumulation();
		}
	}
}

Testbed::LevelStats compute_level_stats(const float* params, size_t n_params) {
	Testbed::LevelStats s = {};
	for (size_t i = 0; i < n_params; ++i) {
		float v = params[i];
		float av = fabsf(v);
		if (av < 0.00001f) {
			s.numzero++;
		} else {
			if (s.count == 0) {
				s.min = s.max = v;
			}
			s.count++;
			s.x += v;
			s.xsquared += v * v;
			s.min = min(s.min, v);
			s.max = max(s.max, v);
		}
	}
	return s;
}

// Increment this number when making a change to the snapshot format
static const size_t SNAPSHOT_FORMAT_VERSION = 1;

void Testbed::save_snapshot(const fs::path& path, bool include_optimizer_state, bool compress) {
	m_network_config["snapshot"] = m_trainer->serialize(include_optimizer_state);

	auto& snapshot = m_network_config["snapshot"];
	snapshot["version"] = SNAPSHOT_FORMAT_VERSION;
	snapshot["mode"] = to_string(m_testbed_mode);

	if (m_testbed_mode == ETestbedMode::Nerf) {
		snapshot["density_grid_size"] = NERF_GRIDSIZE();

		GPUMemory<__half> density_grid_fp16(m_nerf.density_grid.size());
		parallel_for_gpu(
			density_grid_fp16.size(),
			[density_grid = m_nerf.density_grid.data(), density_grid_fp16 = density_grid_fp16.data()] __device__(size_t i) {
				density_grid_fp16[i] = (__half)density_grid[i];
			}
		);

		snapshot["density_grid_binary"] = density_grid_fp16;
		snapshot["nerf"]["aabb_scale"] = m_nerf.training.dataset.aabb_scale;

		snapshot["nerf"]["cam_pos_offset"] = m_nerf.training.cam_pos_offset;
		snapshot["nerf"]["cam_rot_offset"] = m_nerf.training.cam_rot_offset;
		snapshot["nerf"]["extra_dims_opt"] = m_nerf.training.extra_dims_opt;
	}

	snapshot["training_step"] = m_training_step;
	snapshot["loss"] = m_loss_scalar.val();
	snapshot["aabb"] = m_aabb;
	snapshot["bounding_radius"] = m_bounding_radius;
	snapshot["render_aabb_to_local"] = m_render_aabb_to_local;
	snapshot["render_aabb"] = m_render_aabb;
	snapshot["up_dir"] = m_up_dir;
	snapshot["sun_dir"] = m_sun_dir;
	snapshot["exposure"] = m_exposure;
	snapshot["background_color"] = m_background_color;

	snapshot["camera"]["matrix"] = m_camera;
	snapshot["camera"]["fov_axis"] = m_fov_axis;
	snapshot["camera"]["relative_focal_length"] = m_relative_focal_length;
	snapshot["camera"]["screen_center"] = m_screen_center;
	snapshot["camera"]["zoom"] = m_zoom;
	snapshot["camera"]["scale"] = m_scale;

	snapshot["camera"]["aperture_size"] = m_aperture_size;
	snapshot["camera"]["autofocus"] = m_autofocus;
	snapshot["camera"]["autofocus_target"] = m_autofocus_target;
	snapshot["camera"]["autofocus_depth"] = m_slice_plane_z;

	if (m_testbed_mode == ETestbedMode::Nerf) {
		snapshot["nerf"]["rgb"]["rays_per_batch"] = m_nerf.training.counters_rgb.rays_per_batch;
		snapshot["nerf"]["rgb"]["measured_batch_size"] = m_nerf.training.counters_rgb.measured_batch_size;
		snapshot["nerf"]["rgb"]["measured_batch_size_before_compaction"] = m_nerf.training.counters_rgb.measured_batch_size_before_compaction;
		snapshot["nerf"]["dataset"] = m_nerf.training.dataset;
	}

	m_network_config_path = path;
	std::ofstream f{native_string(m_network_config_path), std::ios::out | std::ios::binary};
	if (equals_case_insensitive(m_network_config_path.extension(), "ingp")) {
		// zstr::ofstream applies zlib compression.
		zstr::ostream zf{f, zstr::default_buff_size, compress ? Z_DEFAULT_COMPRESSION : Z_NO_COMPRESSION};
		json::to_msgpack(m_network_config, zf);
	} else {
		json::to_msgpack(m_network_config, f);
	}

	tlog::success() << "Saved snapshot '" << path.str() << "'";
}

void Testbed::load_snapshot(nlohmann::json config) {
	const auto& snapshot = config["snapshot"];
	if (snapshot.value("version", 0) < SNAPSHOT_FORMAT_VERSION) {
		throw std::runtime_error{"Snapshot uses an old format and can not be loaded."};
	}

	if (snapshot.contains("mode")) {
		set_mode(mode_from_string(snapshot["mode"]));
	} else if (snapshot.contains("nerf")) {
		// To be able to load old NeRF snapshots that don't specify their mode yet
		set_mode(ETestbedMode::Nerf);
	} else if (m_testbed_mode == ETestbedMode::None) {
		throw std::runtime_error{"Unknown snapshot mode. Snapshot must be regenerated with a new version of instant-ngp."};
	}

	m_aabb = snapshot.value("aabb", m_aabb);
	m_bounding_radius = snapshot.value("bounding_radius", m_bounding_radius);

	if (m_testbed_mode == ETestbedMode::Nerf) {
		if (snapshot["density_grid_size"] != NERF_GRIDSIZE()) {
			throw std::runtime_error{"Incompatible grid size."};
		}

		m_nerf.training.counters_rgb.rays_per_batch = snapshot["nerf"]["rgb"]["rays_per_batch"];
		m_nerf.training.counters_rgb.measured_batch_size = snapshot["nerf"]["rgb"]["measured_batch_size"];
		m_nerf.training.counters_rgb.measured_batch_size_before_compaction = snapshot["nerf"]["rgb"]["measured_batch_size_before_compaction"];

		// If we haven't got a nerf dataset loaded, load dataset metadata from the snapshot
		// and render using just that.
		if (m_data_path.empty() && snapshot["nerf"].contains("dataset")) {
			m_nerf.training.dataset = snapshot["nerf"]["dataset"];
			load_nerf(m_data_path);
		} else {
			if (snapshot["nerf"].contains("aabb_scale")) {
				m_nerf.training.dataset.aabb_scale = snapshot["nerf"]["aabb_scale"];
			}

			if (snapshot["nerf"].contains("dataset")) {
				m_nerf.training.dataset.n_extra_learnable_dims =
					snapshot["nerf"]["dataset"].value("n_extra_learnable_dims", m_nerf.training.dataset.n_extra_learnable_dims);
			}
		}

		load_nerf_post();

		GPUMemory<__half> density_grid_fp16 = snapshot["density_grid_binary"];
		m_nerf.density_grid.resize(density_grid_fp16.size());

		parallel_for_gpu(
			density_grid_fp16.size(),
			[density_grid = m_nerf.density_grid.data(), density_grid_fp16 = density_grid_fp16.data()] __device__(size_t i) {
				density_grid[i] = (float)density_grid_fp16[i];
			}
		);

		if (m_nerf.density_grid.size() == NERF_GRID_N_CELLS() * (m_nerf.max_cascade + 1)) {
			update_density_grid_mean_and_bitfield(nullptr);
		} else if (m_nerf.density_grid.size() != 0) {
			// A size of 0 indicates that the density grid was never populated, which is a valid state of a (yet) untrained model.
			throw std::runtime_error{"Incompatible number of grid cascades."};
		}
	}

	// Needs to happen after `load_nerf_post()`
	m_sun_dir = snapshot.value("sun_dir", m_sun_dir);
	m_exposure = snapshot.value("exposure", m_exposure);

		m_background_color = snapshot.value("background_color", m_background_color);

	if (snapshot.contains("camera")) {
		m_camera = snapshot["camera"].value("matrix", m_camera);
		m_fov_axis = snapshot["camera"].value("fov_axis", m_fov_axis);
		if (snapshot["camera"].contains("relative_focal_length")) {
			from_json(snapshot["camera"]["relative_focal_length"], m_relative_focal_length);
		}
		if (snapshot["camera"].contains("screen_center")) {
			from_json(snapshot["camera"]["screen_center"], m_screen_center);
		}
		m_zoom = snapshot["camera"].value("zoom", m_zoom);
		m_scale = snapshot["camera"].value("scale", m_scale);

		m_aperture_size = snapshot["camera"].value("aperture_size", m_aperture_size);
		if (m_aperture_size != 0) {
			m_dlss = false;
		}

		m_autofocus = snapshot["camera"].value("autofocus", m_autofocus);
		if (snapshot["camera"].contains("autofocus_target")) {
			from_json(snapshot["camera"]["autofocus_target"], m_autofocus_target);
		}
		m_slice_plane_z = snapshot["camera"].value("autofocus_depth", m_slice_plane_z);
	}

	if (snapshot.contains("render_aabb_to_local")) {
		from_json(snapshot.at("render_aabb_to_local"), m_render_aabb_to_local);
	}
	m_render_aabb = snapshot.value("render_aabb", m_render_aabb);
	if (snapshot.contains("up_dir")) {
		from_json(snapshot.at("up_dir"), m_up_dir);
	}

	m_network_config = std::move(config);

	reset_network(false);

	m_training_step = m_network_config["snapshot"]["training_step"];
	m_loss_scalar.set(m_network_config["snapshot"]["loss"]);

	m_trainer->deserialize(m_network_config["snapshot"]);

	if (m_testbed_mode == ETestbedMode::Nerf) {
		// If the snapshot appears to come from the same dataset as was already present
		// (or none was previously present, in which case it came from the snapshot
		// in the first place), load dataset-specific optimized quantities, such as
		// extrinsics, exposure, latents.
		if (snapshot["nerf"].contains("dataset") && m_nerf.training.dataset.is_same(snapshot["nerf"]["dataset"])) {
			if (snapshot["nerf"].contains("cam_pos_offset")) {
				m_nerf.training.cam_pos_offset = snapshot["nerf"].at("cam_pos_offset").get<std::vector<AdamOptimizer<vec3>>>();
			}
			if (snapshot["nerf"].contains("cam_rot_offset")) {
				m_nerf.training.cam_rot_offset = snapshot["nerf"].at("cam_rot_offset").get<std::vector<RotationAdamOptimizer>>();
			}
			if (snapshot["nerf"].contains("extra_dims_opt")) {
				m_nerf.training.extra_dims_opt = snapshot["nerf"].at("extra_dims_opt").get<std::vector<VarAdamOptimizer>>();
			}
			m_nerf.training.update_transforms();
			m_nerf.training.update_extra_dims();
		}
	}

	set_all_devices_dirty();
}

void Testbed::load_snapshot(const fs::path& path) {
	auto config = load_network_config(path);
	if (!config.contains("snapshot")) {
		throw std::runtime_error{fmt::format("File '{}' does not contain a snapshot.", path.str())};
	}

	load_snapshot(std::move(config));

	m_network_config_path = path;
}

void Testbed::load_snapshot(std::istream& stream, bool is_compressed) {
	auto config = load_network_config(stream, is_compressed);
	if (!config.contains("snapshot")) {
		throw std::runtime_error{"Given stream does not contain a snapshot."};
	}

	load_snapshot(std::move(config));

	// Network config path is unknown.
	m_network_config_path = "";
}

Testbed::CudaDevice::CudaDevice(int id, bool is_primary) : m_id{id}, m_is_primary{is_primary} {
	auto guard = device_guard();
	m_stream = std::make_unique<StreamAndEvent>();
	m_data = std::make_unique<Data>();
	m_render_worker = std::make_unique<ThreadPool>(is_primary ? 0u : 1u);
}

ScopeGuard Testbed::CudaDevice::device_guard() {
	int prev_device = cuda_device();
	if (prev_device == m_id) {
		return {};
	}

	set_cuda_device(m_id);
	return ScopeGuard{[prev_device]() { set_cuda_device(prev_device); }};
}

void Testbed::CudaDevice::set_network(const std::shared_ptr<Network<float, network_precision_t>>& network) {
	m_network = network;
	m_fused_render_kernel = nullptr;
}

void Testbed::CudaDevice::set_nerf_network(const std::shared_ptr<NerfNetwork<network_precision_t>>& nerf_network) {
	m_nerf_network = nerf_network;
	set_network(nerf_network);
}

CudaRtcKernel* Testbed::CudaDevice::fused_render_kernel() { return m_fused_render_kernel.get(); }

void Testbed::CudaDevice::set_fused_render_kernel(std::unique_ptr<CudaRtcKernel>&& kernel) { m_fused_render_kernel = std::move(kernel); }

void Testbed::sync_device(CudaRenderBuffer& render_buffer, Testbed::CudaDevice& device) {
	if (!device.dirty()) {
		return;
	}

	if (device.is_primary()) {
		device.data().density_grid_bitfield_ptr = m_nerf.density_grid_bitfield.data();
		device.data().hidden_area_mask = render_buffer.hidden_area_mask();
		device.set_dirty(false);
		return;
	}

	m_stream.signal(device.stream());

	int active_device = cuda_device();
	auto guard = device.device_guard();

	device.data().density_grid_bitfield.resize(m_nerf.density_grid_bitfield.size());
	if (m_nerf.density_grid_bitfield.size() > 0) {
		CUDA_CHECK_THROW(cudaMemcpyPeerAsync(
			device.data().density_grid_bitfield.data(),
			device.id(),
			m_nerf.density_grid_bitfield.data(),
			active_device,
			m_nerf.density_grid_bitfield.bytes(),
			device.stream()
		));
	}

	device.data().density_grid_bitfield_ptr = device.data().density_grid_bitfield.data();

	if (m_network) {
		device.data().params.resize(m_network->n_params());
		CUDA_CHECK_THROW(cudaMemcpyPeerAsync(
			device.data().params.data(), device.id(), m_network->inference_params(), active_device, device.data().params.bytes(), device.stream()
		));
		device.nerf_network()->set_params(device.data().params.data(), device.data().params.data(), nullptr);
	}

	if (render_buffer.hidden_area_mask()) {
		auto ham = std::make_shared<Buffer2D<uint8_t>>(render_buffer.hidden_area_mask()->resolution());
		CUDA_CHECK_THROW(cudaMemcpyPeerAsync(
			ham->data(), device.id(), render_buffer.hidden_area_mask()->data(), active_device, ham->bytes(), device.stream()
		));
		device.data().hidden_area_mask = ham;
	} else {
		device.data().hidden_area_mask = nullptr;
	}

	device.set_dirty(false);
	device.signal(m_stream.get());
}

// From https://stackoverflow.com/questions/20843271/passing-a-non-copyable-closure-object-to-stdfunction-parameter
template <class F> auto make_copyable_function(F&& f) {
	using dF = std::decay_t<F>;
	auto spf = std::make_shared<dF>(std::forward<F>(f));
	return [spf](auto&&... args) -> decltype(auto) { return (*spf)(decltype(args)(args)...); };
}

ScopeGuard Testbed::use_device(cudaStream_t stream, CudaRenderBuffer& render_buffer, Testbed::CudaDevice& device) {
	device.wait_for(stream);

	if (device.is_primary()) {
		device.set_render_buffer_view(render_buffer.view());
		return ScopeGuard{[&device, stream]() {
			device.set_render_buffer_view({});
			device.signal(stream);
		}};
	}

	int active_device = cuda_device();
	auto guard = device.device_guard();

	size_t n_pixels = product(render_buffer.in_resolution());

	GPUMemoryArena::Allocation alloc;
	auto scratch = allocate_workspace_and_distribute<vec4, float>(device.stream(), &alloc, n_pixels, n_pixels);

	device.set_render_buffer_view({
		std::get<0>(scratch),
		std::get<1>(scratch),
		render_buffer.in_resolution(),
		render_buffer.spp(),
		device.data().hidden_area_mask,
	});

	return ScopeGuard{
		make_copyable_function([&render_buffer, &device, guard = std::move(guard), alloc = std::move(alloc), active_device, stream]() {
			// Copy device's render buffer's data onto the original render buffer
			CUDA_CHECK_THROW(cudaMemcpyPeerAsync(
				render_buffer.frame_buffer(),
				active_device,
				device.render_buffer_view().frame_buffer,
				device.id(),
				product(render_buffer.in_resolution()) * sizeof(vec4),
				device.stream()
			));
			CUDA_CHECK_THROW(cudaMemcpyPeerAsync(
				render_buffer.depth_buffer(),
				active_device,
				device.render_buffer_view().depth_buffer,
				device.id(),
				product(render_buffer.in_resolution()) * sizeof(float),
				device.stream()
			));

			device.set_render_buffer_view({});
			device.signal(stream);
		})
	};
}

void Testbed::set_all_devices_dirty() {
	for (auto& device : m_devices) {
		device.set_dirty(true);
	}
}

void Testbed::load_camera_path(const fs::path& path) { m_camera_path.load(path, mat4x3::identity()); }

bool Testbed::loop_animation() { return m_camera_path.loop; }

void Testbed::set_loop_animation(bool value) { m_camera_path.loop = value; }

} // namespace ngp
