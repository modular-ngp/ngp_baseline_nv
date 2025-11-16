cmake_minimum_required(VERSION 3.26)

# ngp-minimal: Standalone minimal NeRF-synthetic training implementation
# No dependencies on the full ngp-baseline-nv library

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Use the same CUDA architectures as parent project
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75;86)
endif()

# Create the minimal app executable
add_executable(ngp-minimal-app
    src/main.cu
    src/nerf_loader.cu
    src/nerf_network.cu
    src/testbed.cu
    src/testbed_nerf.cu
)

# Include directories - ONLY from ngp-minimal and vendors
target_include_directories(ngp-minimal-app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/vendor
    ${CMAKE_SOURCE_DIR}/vendor/filesystem
    ${CMAKE_SOURCE_DIR}/vendor/tinylogger
    ${CMAKE_SOURCE_DIR}/vendor/args
)

# Link tiny-cuda-nn (which is already fetched by parent CMakeLists.txt)
target_link_libraries(ngp-minimal-app PRIVATE
    tiny-cuda-nn
)

# Set CUDA properties
set_target_properties(ngp-minimal-app PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Compiler options
if(MSVC)
    target_compile_options(ngp-minimal-app PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>
    )
else()
    target_compile_options(ngp-minimal-app PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    )
endif()

message(STATUS "ngp-minimal: Configured standalone minimal NeRF training app")
message(STATUS "ngp-minimal: CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

