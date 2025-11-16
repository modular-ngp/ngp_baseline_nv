/*
 * ngp-minimal: NeRF dataset loader implementation
 * Simplified loader for NeRF-synthetic (Blender) datasets
 */

#include <ngp-minimal/nerf_loader.h>
#include <ngp-minimal/common_device.cuh>

#include <json/json.hpp>
#include <tinylogger/tinylogger.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image/stb_image.h>

#include <fstream>
#include <algorithm>
#include <cmath>

using json = nlohmann::json;

namespace ngp {

// CUDA kernel for converting RGBA to device format
__global__ void convert_rgba32_kernel(
    uint64_t num_pixels,
    const uint8_t* __restrict__ pixels,
    uint8_t* __restrict__ out
) {
    const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_pixels) return;

    // Simple copy for now (can add transparency handling later)
    uint32_t* out_ptr = reinterpret_cast<uint32_t*>(out);
    const uint32_t* in_ptr = reinterpret_cast<const uint32_t*>(pixels);
    out_ptr[i] = in_ptr[i];
}

void NerfDataset::update_metadata(int first, int last) {
    if (last < 0) {
        last = (int)metadata.size();
    }

    if (metadata.empty()) {
        return;
    }

    // Allocate GPU memory for metadata if needed
    if (metadata_gpu.size() < metadata.size()) {
        metadata_gpu.resize(metadata.size());
    }

    // Copy metadata to GPU
    size_t count = last - first;
    if (count > 0) {
        cudaMemcpy(
            metadata_gpu.data() + first,
            metadata.data() + first,
            count * sizeof(TrainingImageMetadata),
            cudaMemcpyHostToDevice
        );
    }
}

void NerfDataset::set_training_image(
    int frame_idx,
    const ivec2& image_resolution,
    const void* pixels,
    const void* depth_pixels,
    float depth_scale,
    bool image_data_on_gpu,
    EImageDataType image_type,
    EDepthDataType depth_type,
    float sharpen_amount,
    bool white_transparent,
    bool black_transparent,
    uint32_t mask_color,
    const Ray* rays
) {
    if (frame_idx < 0 || frame_idx >= (int)metadata.size()) {
        return;
    }

    auto& m = metadata[frame_idx];
    m.resolution = image_resolution;
    m.image_data_type = image_type;

    // Allocate and upload pixel data
    size_t n_pixels = (size_t)image_resolution.x * image_resolution.y;
    size_t img_size = n_pixels * 4; // RGBA

    if (image_type == EImageDataType::Byte) {
        pixelmemory[frame_idx].resize(img_size);

        if (image_data_on_gpu) {
            // Data already on GPU
            cudaMemcpy(
                pixelmemory[frame_idx].data(),
                pixels,
                img_size,
                cudaMemcpyDeviceToDevice
            );
        } else {
            // Upload from CPU
            cudaMemcpy(
                pixelmemory[frame_idx].data(),
                pixels,
                img_size,
                cudaMemcpyHostToDevice
            );
        }

        m.pixels = pixelmemory[frame_idx].data();
    }
}

// Helper function to read focal length from JSON
static bool read_focal_length(
    const json& json_data,
    const ivec2& res,
    vec2& focal_length
) {
    auto read_fl = [&](int resolution, const char* axis) -> float {
        std::string fov_key = std::string("camera_angle_") + axis;
        std::string fl_key = std::string("fl_") + axis;

        if (json_data.contains(fl_key)) {
            return json_data[fl_key];
        } else if (json_data.contains(fov_key)) {
            float angle = json_data[fov_key];
            return 0.5f * resolution / std::tan(0.5f * angle);
        } else {
            return 0.0f;
        }
    };

    float x_fl = read_fl(res.x, "x");
    float y_fl = read_fl(res.y, "y");

    if (x_fl != 0) {
        focal_length = vec2(x_fl);
        if (y_fl != 0) {
            focal_length.y = y_fl;
        }
    } else if (y_fl != 0) {
        focal_length = vec2(y_fl);
    } else {
        return false;
    }

    return true;
}

// Helper to load an image from disk
static void* load_image_from_disk(
    const fs::path& path,
    ivec2& resolution,
    int& n_channels
) {
    if (!path.exists()) {
        tlog::warning() << "Image file does not exist: " << path.str();
        return nullptr;
    }

    int width, height, comp;
    stbi_set_flip_vertically_on_load(false);

    unsigned char* data = stbi_load(
        path.str().c_str(),
        &width,
        &height,
        &comp,
        4  // Force RGBA
    );

    if (!data) {
        tlog::error() << "Failed to load image: " << path.str();
        return nullptr;
    }

    resolution = ivec2(width, height);
    n_channels = 4;
    return data;
}

NerfDataset load_nerf(
    const std::vector<fs::path>& jsonpaths,
    float sharpen_amount
) {
    if (jsonpaths.empty()) {
        throw std::runtime_error("Cannot load NeRF data from empty path list");
    }

    tlog::info() << "Loading NeRF-synthetic dataset from " << jsonpaths.size() << " JSON file(s)";

    NerfDataset result;

    // Parse all JSON files
    std::vector<json> jsons;
    for (const auto& path : jsonpaths) {
        tlog::info() << "  Reading: " << path.str();
        std::ifstream f(path.str());
        if (!f.is_open()) {
            tlog::error() << "Failed to open JSON file: " << path.str();
            continue;
        }
        jsons.push_back(json::parse(f, nullptr, true, true));
    }

    // Count total frames
    result.n_images = 0;
    for (size_t i = 0; i < jsons.size(); ++i) {
        auto& j = jsons[i];
        if (j.contains("frames") && j["frames"].is_array()) {
            result.n_images += j["frames"].size();
        }
    }

    if (result.n_images == 0) {
        throw std::runtime_error("No frames found in JSON files");
    }

    tlog::info() << "Total images to load: " << result.n_images;

    // Allocate storage
    result.metadata.resize(result.n_images);
    result.xforms.resize(result.n_images);
    result.paths.resize(result.n_images);
    result.pixelmemory.resize(result.n_images);

    // Set defaults
    result.scale = NERF_SCALE;
    result.offset = vec3(0.5f, 0.5f, 0.5f);
    result.aabb_scale = 1;

    // Read aabb_scale from first JSON if available
    if (jsons[0].contains("aabb_scale")) {
        result.aabb_scale = jsons[0]["aabb_scale"];
    }

    // Process all frames
    size_t frame_idx = 0;
    for (size_t json_idx = 0; json_idx < jsons.size(); ++json_idx) {
        auto& j = jsons[json_idx];
        fs::path base_path = jsonpaths[json_idx].parent_path();

        if (!j.contains("frames") || !j["frames"].is_array()) {
            continue;
        }

        auto& frames = j["frames"];

        for (size_t i = 0; i < frames.size(); ++i) {
            auto& frame = frames[i];

            // Get image path
            std::string file_path = frame["file_path"];
            // Handle Windows paths on any platform
            std::replace(file_path.begin(), file_path.end(), '\\', '/');

            fs::path img_path = base_path / file_path;

            // Try adding .png extension if path doesn't exist
            if (!img_path.exists() && img_path.extension().empty()) {
                img_path = img_path.with_extension("png");
            }

            result.paths[frame_idx] = img_path.str();

            // Read transform matrix (4x4 camera-to-world)
            if (frame.contains("transform_matrix")) {
                auto& matrix = frame["transform_matrix"];
                mat4x3 xform;
                for (int row = 0; row < 3; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        xform[col][row] = matrix[row][col];
                    }
                }

                // Convert from NeRF to NGP coordinate system
                result.xforms[frame_idx].start = result.nerf_matrix_to_ngp(xform);
                result.xforms[frame_idx].end = result.xforms[frame_idx].start;
            }

            // Load image
            ivec2 resolution;
            int n_channels;
            void* image_data = load_image_from_disk(img_path, resolution, n_channels);

            if (!image_data) {
                tlog::warning() << "Skipping frame " << frame_idx << " (failed to load image)";
                frame_idx++;
                continue;
            }

            // Get focal length
            vec2 focal_length(0.0f);
            if (!read_focal_length(j, resolution, focal_length)) {
                tlog::error() << "Failed to read focal length for frame " << frame_idx;
                stbi_image_free(image_data);
                frame_idx++;
                continue;
            }

            // Set metadata
            auto& meta = result.metadata[frame_idx];
            meta.resolution = resolution;
            meta.focal_length = focal_length;
            meta.principal_point = vec2(0.5f, 0.5f); // Centered by default
            meta.lens = Lens(ELensMode::Perspective);

            // Upload image data to GPU
            result.set_training_image(
                frame_idx,
                resolution,
                image_data,
                nullptr,  // no depth
                0.0f,
                false,    // data on CPU
                EImageDataType::Byte,
                EDepthDataType::Float
            );

            stbi_image_free(image_data);

            if ((frame_idx + 1) % 10 == 0 || frame_idx == result.n_images - 1) {
                tlog::info() << "Loaded " << (frame_idx + 1) << "/" << result.n_images << " images";
            }

            frame_idx++;
        }
    }

    // Compute bounding box from camera positions
    BoundingBox aabb;
    for (const auto& xform : result.xforms) {
        // Camera position is in the 4th column (translation)
        vec3 pos = xform.start[3];
        aabb.enlarge(pos);
    }

    // Set render AABB
    float scene_scale = 1.0f / result.aabb_scale;
    result.render_aabb = BoundingBox(
        vec3(0.5f - scene_scale),
        vec3(0.5f + scene_scale)
    );
    result.render_aabb_to_local = mat3::identity();

    // Update GPU metadata
    result.update_metadata();

    tlog::success() << "Successfully loaded " << result.n_images << " images";
    tlog::info() << "  Resolution: " << result.metadata[0].resolution.x
                 << "x" << result.metadata[0].resolution.y;
    tlog::info() << "  Focal length: " << result.metadata[0].focal_length.x;
    tlog::info() << "  AABB scale: " << result.aabb_scale;
    tlog::info() << "  Render AABB: [" << result.render_aabb.min.x << ", "
                 << result.render_aabb.min.y << ", " << result.render_aabb.min.z
                 << "] - [" << result.render_aabb.max.x << ", "
                 << result.render_aabb.max.y << ", " << result.render_aabb.max.z << "]";

    return result;
}

} // namespace ngp

