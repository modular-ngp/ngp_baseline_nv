/*
 * ngp-minimal: NeRF device-side structures and constants
 * CUDA device code for NeRF training
 */

#pragma once

#include <ngp-minimal/common_device.cuh>

namespace ngp {

// NeRF grid constants
inline constexpr NGP_HOST_DEVICE uint32_t NERF_GRIDSIZE() { return 128; }
inline constexpr NGP_HOST_DEVICE uint32_t NERF_GRID_N_CELLS() {
    return NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE();
}

// Ray marching constants
inline constexpr NGP_HOST_DEVICE float NERF_RENDERING_NEAR_DISTANCE() { return 0.05f; }
inline constexpr NGP_HOST_DEVICE uint32_t NERF_STEPS() { return 1024; }
inline constexpr NGP_HOST_DEVICE uint32_t NERF_CASCADES() { return 8; }

inline constexpr NGP_HOST_DEVICE float SQRT3() { return 1.73205080757f; }
inline constexpr NGP_HOST_DEVICE float STEPSIZE() { return (SQRT3() / NERF_STEPS()); }
inline constexpr NGP_HOST_DEVICE float MIN_CONE_STEPSIZE() { return STEPSIZE(); }
inline constexpr NGP_HOST_DEVICE float MAX_CONE_STEPSIZE() {
    return STEPSIZE() * (1<<(NERF_CASCADES()-1)) * NERF_STEPS() / NERF_GRIDSIZE();
}

inline constexpr NGP_HOST_DEVICE uint32_t N_MAX_RANDOM_SAMPLES_PER_RAY() { return 16; }
inline constexpr NGP_HOST_DEVICE float NERF_MIN_OPTICAL_THICKNESS() { return 0.01f; }

// Loss functions
struct LossAndGradient {
    vec3 loss;
    vec3 gradient;

    NGP_HOST_DEVICE LossAndGradient operator*(float scalar) {
        return {loss * scalar, gradient * scalar};
    }

    NGP_HOST_DEVICE LossAndGradient operator/(float scalar) {
        return {loss / scalar, gradient / scalar};
    }
};

inline NGP_HOST_DEVICE LossAndGradient l2_loss(const vec3& target, const vec3& prediction) {
    vec3 difference = prediction - target;
    return {
        difference * difference,
        2.0f * difference
    };
}

inline NGP_HOST_DEVICE LossAndGradient l1_loss(const vec3& target, const vec3& prediction) {
    vec3 difference = prediction - target;
    return {
        abs(difference),
        copysign(vec3(1.0f), difference),
    };
}

inline NGP_HOST_DEVICE LossAndGradient huber_loss(const vec3& target, const vec3& prediction, float alpha = 1) {
    vec3 difference = prediction - target;
    vec3 abs_diff = abs(difference);
    vec3 square = 0.5f/alpha * difference * difference;
    return {
        {
            abs_diff.x > alpha ? (abs_diff.x - 0.5f * alpha) : square.x,
            abs_diff.y > alpha ? (abs_diff.y - 0.5f * alpha) : square.y,
            abs_diff.z > alpha ? (abs_diff.z - 0.5f * alpha) : square.z,
        },
        {
            abs_diff.x > alpha ? (difference.x > 0 ? 1.0f : -1.0f) : (difference.x / alpha),
            abs_diff.y > alpha ? (difference.y > 0 ? 1.0f : -1.0f) : (difference.y / alpha),
            abs_diff.z > alpha ? (difference.z > 0 ? 1.0f : -1.0f) : (difference.z / alpha),
        },
    };
}

// NeRF payload for ray marching
struct NerfPayload {
    vec3 origin;
    vec3 dir;
    float t;
    float max_weight;
    uint32_t idx;
    uint16_t n_steps;
    uint16_t padding;
};

// Coordinate system helpers
struct NerfCoordinate {
    NGP_HOST_DEVICE vec3 pos_to_unit_cube(const vec3& pos, const BoundingBox& aabb) const {
        return (pos - aabb.min) / (aabb.max - aabb.min);
    }

    NGP_HOST_DEVICE vec3 unit_cube_to_pos(const vec3& unit, const BoundingBox& aabb) const {
        return unit * (aabb.max - aabb.min) + aabb.min;
    }
};

} // namespace ngp

