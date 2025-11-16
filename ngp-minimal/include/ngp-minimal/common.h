/*
 * ngp-minimal: Common types and utilities
 * Basic math types, enums, and constants
 */

#pragma once

#ifdef _WIN32
#  define NOMINMAX
#endif

#include <tiny-cuda-nn/common.h>

using namespace tcnn;

namespace ngp {

// Training modes
enum class ETrainMode : int {
    Nerf,
    Rfl,
    RflRelax,
};

// Testbed modes
enum class ETestbedMode : int {
    None,
    Nerf,
};

// Loss types
enum class ELossType : int {
    L2,
    L1,
    Huber,
    LogL1,
    RelativeL2,
};

// Image data types
enum class EImageDataType : int {
    None,
    Byte,
    Half,
    Float,
};

// Depth data types
enum class EDepthDataType : int {
    UShort,
    Float,
};

// Camera lens types
enum class ELensMode : int {
    Perspective,
    OpenCV,
    FTheta,
    LatLong,
    OpenCVFisheye,
};

// NeRF coordinate system scale factor
static constexpr float NERF_SCALE = 0.33f;

// Helper functions
inline size_t image_type_size(EImageDataType type) {
    switch (type) {
        case EImageDataType::None: return 0;
        case EImageDataType::Byte: return 1;
        case EImageDataType::Half: return 2;
        case EImageDataType::Float: return 4;
        default: return 0;
    }
}

inline size_t depth_type_size(EDepthDataType type) {
    switch (type) {
        case EDepthDataType::UShort: return 2;
        case EDepthDataType::Float: return 4;
        default: return 0;
    }
}

// Common CUDA macros
#if defined(__CUDA_ARCH__)
#  define NGP_PRAGMA_UNROLL _Pragma("unroll")
#  define NGP_PRAGMA_NO_UNROLL _Pragma("unroll 1")
#else
#  define NGP_PRAGMA_UNROLL
#  define NGP_PRAGMA_NO_UNROLL
#endif

#if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))
#  define NGP_HOST_DEVICE __host__ __device__
#else
#  define NGP_HOST_DEVICE
#endif

} // namespace ngp

