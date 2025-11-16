/*
 * ngp-minimal: Device-side utilities
 * CUDA device code for math and common structures
 */

#pragma once

#include <ngp-minimal/common.h>

namespace ngp {

// Bounding box structure
struct BoundingBox {
    vec3 min = vec3(std::numeric_limits<float>::infinity());
    vec3 max = vec3(-std::numeric_limits<float>::infinity());

    NGP_HOST_DEVICE BoundingBox() = default;

    NGP_HOST_DEVICE BoundingBox(const vec3& a, const vec3& b) : min(a), max(b) {}

    NGP_HOST_DEVICE void enlarge(const vec3& point) {
        min = tcnn::min(min, point);
        max = tcnn::max(max, point);
    }

    NGP_HOST_DEVICE void enlarge(const BoundingBox& other) {
        enlarge(other.min);
        enlarge(other.max);
    }

    NGP_HOST_DEVICE void inflate(float amount) {
        min -= vec3(amount);
        max += vec3(amount);
    }

    NGP_HOST_DEVICE vec3 diag() const {
        return max - min;
    }

    NGP_HOST_DEVICE vec3 relative_pos(const vec3& pos) const {
        return (pos - min) / diag();
    }

    NGP_HOST_DEVICE vec3 center() const {
        return 0.5f * (max + min);
    }

    NGP_HOST_DEVICE bool contains(const vec3& p) const {
        return p.x >= min.x && p.x <= max.x &&
               p.y >= min.y && p.y <= max.y &&
               p.z >= min.z && p.z <= max.z;
    }

    NGP_HOST_DEVICE bool intersects(const BoundingBox& other) const {
        return min.x <= other.max.x && max.x >= other.min.x &&
               min.y <= other.max.y && max.y >= other.min.y &&
               min.z <= other.max.z && max.z >= other.min.z;
    }

    NGP_HOST_DEVICE BoundingBox intersection(const BoundingBox& other) const {
        BoundingBox result;
        result.min = tcnn::max(min, other.min);
        result.max = tcnn::min(max, other.max);
        return result;
    }

    static NGP_HOST_DEVICE BoundingBox unit_cube() {
        return BoundingBox(vec3(0.0f), vec3(1.0f));
    }
};

// Ray structure for ray tracing
struct Ray {
    vec3 o;  // origin
    vec3 d;  // direction

    NGP_HOST_DEVICE Ray() = default;

    NGP_HOST_DEVICE Ray(const vec3& origin, const vec3& direction)
        : o(origin), d(direction) {}

    NGP_HOST_DEVICE vec3 operator()(float t) const {
        return o + t * d;
    }

    NGP_HOST_DEVICE void advance(float t) {
        o += d * t;
    }
};

// Camera lens structure
struct Lens {
    ELensMode mode = ELensMode::Perspective;
    float params[7] = {};

    NGP_HOST_DEVICE Lens() = default;

    NGP_HOST_DEVICE Lens(ELensMode m) : mode(m) {}
};

// Coordinate transformation helpers
NGP_HOST_DEVICE inline vec3 warp_position(const vec3& pos, const BoundingBox& aabb) {
    return aabb.min + pos * aabb.diag();
}

NGP_HOST_DEVICE inline vec3 unwarp_position(const vec3& pos, const BoundingBox& aabb) {
    return (pos - aabb.min) / aabb.diag();
}

NGP_HOST_DEVICE inline vec3 warp_direction(const vec3& dir) {
    return (dir + vec3(1.0f)) * 0.5f;
}

NGP_HOST_DEVICE inline vec3 unwarp_direction(const vec3& dir) {
    return dir * 2.0f - vec3(1.0f);
}

NGP_HOST_DEVICE inline float warp_dt(float dt) {
    return dt * NERF_SCALE;
}

} // namespace ngp

