cmake_minimum_required(VERSION 3.26)
if (WIN32)
    include(cmake/setup_vcpkg.cmake)
endif ()
project(ngp-baseline-nv VERSION 1.0.0 LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_CUDA_ARCHITECTURES 75;86 CACHE STRING "CUDA archs" FORCE)
list(GET CMAKE_CUDA_ARCHITECTURES 0 _min_arch)
set(TCNN_MIN_GPU_ARCH "${_min_arch}" CACHE STRING "" FORCE)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)


add_library(ngp-baseline-nv
        STATIC
        src/camera_path.cu
        src/common_host.cu
        src/marching_cubes.cu
        src/nerf_loader.cu
        src/render_buffer.cu
        src/testbed.cu
        src/testbed_nerf.cu
        src/thread_pool.cpp
        src/tinyexr_wrapper.cu
)
find_package(ZLIB REQUIRED)
find_path(ZSTR_INCLUDE_DIRS "zstr.hpp")
target_link_libraries(ngp-baseline-nv PRIVATE ZLIB::ZLIB)
target_include_directories(ngp-baseline-nv
        PUBLIC
        include
        ${ZSTR_INCLUDE_DIRS}
        vendor
        vendor/filesystem
        vendor/tinylogger
        vendor/NaturalSort
        vendor/nanovdb
)


include(FetchContent)
set(TCNN_ALLOW_CUBLAS_CUSOLVER OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_NO_FWD_BWD OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_WITH_RTC ON CACHE BOOL "" FORCE)
set(TCNN_BUILD_USE_FAST_MATH ON CACHE BOOL "" FORCE)
set(TCNN_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "" FORCE)
FetchContent_Declare(
        tiny-cuda-nn
        GIT_REPOSITORY https://github.com/NVlabs/tiny-cuda-nn.git
        GIT_TAG master
)
FetchContent_MakeAvailable(tiny-cuda-nn)


cmrc_add_resource_library(ngp-resources NAMESPACE ngp)
list(APPEND NGP_LIBRARIES ngp-resources)
set(NGP_BUILD_WITH_RTC ON)
if (NGP_BUILD_WITH_RTC)
    # Create local directory that can serve as cache for the runtime compiled artifacts
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/rtc/cache")

    file(GLOB_RECURSE NGP_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/neural-graphics-primitives/*")
    cmrc_add_resources(ngp-resources WHENCE "${CMAKE_CURRENT_SOURCE_DIR}/include" ${NGP_HEADERS})
endif()
target_link_libraries(ngp-baseline-nv PUBLIC tiny-cuda-nn ngp-resources)

#FetchContent_Declare(
#        dataset
#        GIT_REPOSITORY https://github.com/modular-ngp/dataset.git
#        GIT_TAG master
#)
#FetchContent_MakeAvailable(dataset)
#target_link_libraries(ngp-baseline-nv PRIVATE dataset::dataset)

add_executable(ngp-baseline-nv-app main.cu)
target_link_libraries(ngp-baseline-nv-app PRIVATE ngp-baseline-nv)

add_subdirectory(ngp-minimal)
