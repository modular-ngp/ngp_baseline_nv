/*
 * ngp-minimal: NeRF dataset loader
 * Structures and functions for loading NeRF-synthetic datasets
 */

#pragma once

#include <ngp-minimal/common.h>
#include <ngp-minimal/common_device.cuh>
#include <ngp-minimal/gpu_memory.h>

#include <filesystem/path.h>
#include <vector>
#include <string>

namespace ngp {

namespace fs = ::filesystem;

// Training image metadata
struct TrainingImageMetadata {
    ivec2 resolution = ivec2(0);
    vec2 focal_length = vec2(0.0f);
    vec2 principal_point = vec2(0.0f);
    Lens lens;
    vec4 rolling_shutter = vec4(0.0f);

    void* pixels = nullptr;  // Device pointer to pixel data
    void* depth_pixels = nullptr;  // Optional depth data

    EImageDataType image_data_type = EImageDataType::Byte;
    EDepthDataType depth_data_type = EDepthDataType::Float;
};

// Camera transform (for interpolation support)
struct TrainingXForm {
    mat4x3 start = mat4x3::identity();
    mat4x3 end = mat4x3::identity();
};

// NeRF dataset structure (minimal version - only fields needed for training)
struct NerfDataset {
    // Core data
    std::vector<GPUMemory<uint8_t>> pixelmemory;
    std::vector<TrainingImageMetadata> metadata;
    GPUMemory<TrainingImageMetadata> metadata_gpu;

    std::vector<TrainingXForm> xforms;
    std::vector<std::string> paths;

    // Scene bounds and transforms
    BoundingBox render_aabb = BoundingBox::unit_cube();
    mat3 render_aabb_to_local = mat3::identity();
    vec3 up = vec3(0.0f, 1.0f, 0.0f);
    vec3 offset = vec3(0.0f, 0.0f, 0.0f);
    float scale = 1.0f;
    int aabb_scale = 1;

    size_t n_images = 0;

    // Update GPU metadata from CPU
    void update_metadata(int first = 0, int last = -1);

    // Set a training image (allocate GPU memory and upload)
    void set_training_image(
        int frame_idx,
        const ivec2& image_resolution,
        const void* pixels,
        const void* depth_pixels,
        float depth_scale,
        bool image_data_on_gpu,
        EImageDataType image_type,
        EDepthDataType depth_type,
        float sharpen_amount = 0.0f,
        bool white_transparent = false,
        bool black_transparent = false,
        uint32_t mask_color = 0,
        const Ray* rays = nullptr
    );

    // Coordinate conversion helpers for NeRF-synthetic (Blender) format
    vec3 nerf_direction_to_ngp(const vec3& nerf_dir) const {
        // Cycle axes xyz <- yzx
        return vec3(nerf_dir.y, nerf_dir.z, nerf_dir.x);
    }

    mat4x3 nerf_matrix_to_ngp(const mat4x3& nerf_matrix, bool scale_columns = false) const {
        mat4x3 result = nerf_matrix;
        result[0] *= scale_columns ? scale : 1.0f;
        result[1] *= scale_columns ? -scale : -1.0f;
        result[2] *= scale_columns ? -scale : -1.0f;
        result[3] = result[3] * scale + offset;

        // Cycle axes
        mat4x3 cycled;
        cycled[0] = result[1];
        cycled[1] = result[2];
        cycled[2] = result[0];
        cycled[3] = result[3];
        return cycled;
    }

    vec3 ngp_position_to_nerf(const vec3& pos) const {
        vec3 p = (pos - offset) / scale;
        return vec3(p.z, p.x, p.y);
    }
};

// Load NeRF-synthetic dataset from JSON files
NerfDataset load_nerf(
    const std::vector<fs::path>& json_paths,
    float sharpen_amount = 0.0f
);

} // namespace ngp

