/*
 * ngp-minimal: GPU memory management
 * Simple CUDA memory wrapper
 */

#pragma once

#include <ngp-minimal/common.h>
#include <stdexcept>
#include <cuda_runtime.h>

namespace ngp {

template <typename T>
class GPUMemory {
public:
    GPUMemory() : m_data(nullptr), m_size(0) {}

    explicit GPUMemory(size_t size) : m_data(nullptr), m_size(0) {
        resize(size);
    }

    ~GPUMemory() {
        free_memory();
    }

    // Disable copy
    GPUMemory(const GPUMemory&) = delete;
    GPUMemory& operator=(const GPUMemory&) = delete;

    // Enable move
    GPUMemory(GPUMemory&& other) noexcept
        : m_data(other.m_data), m_size(other.m_size) {
        other.m_data = nullptr;
        other.m_size = 0;
    }

    GPUMemory& operator=(GPUMemory&& other) noexcept {
        if (this != &other) {
            free_memory();
            m_data = other.m_data;
            m_size = other.m_size;
            other.m_data = nullptr;
            other.m_size = 0;
        }
        return *this;
    }

    void resize(size_t size) {
        if (size == m_size) {
            return;
        }

        free_memory();

        if (size > 0) {
            cudaError_t err = cudaMalloc(&m_data, size * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("Failed to allocate GPU memory: ") +
                    cudaGetErrorString(err)
                );
            }
            m_size = size;
        }
    }

    void enlarge(size_t size) {
        if (size > m_size) {
            resize(size);
        }
    }

    void free_memory() {
        if (m_data) {
            cudaFree(m_data);
            m_data = nullptr;
        }
        m_size = 0;
    }

    void memset(int value) {
        if (m_data && m_size > 0) {
            cudaMemset(m_data, value, m_size * sizeof(T));
        }
    }

    void memset_async(cudaStream_t stream, int value) {
        if (m_data && m_size > 0) {
            cudaMemsetAsync(m_data, value, m_size * sizeof(T), stream);
        }
    }

    void copy_from_host(const T* host_data, size_t count) {
        if (count > m_size) {
            resize(count);
        }
        if (m_data && host_data && count > 0) {
            cudaError_t err = cudaMemcpy(m_data, host_data, count * sizeof(T),
                                        cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("Failed to copy to GPU: ") +
                    cudaGetErrorString(err)
                );
            }
        }
    }

    void copy_to_host(T* host_data, size_t count) const {
        if (m_data && host_data && count > 0) {
            size_t copy_count = std::min(count, m_size);
            cudaError_t err = cudaMemcpy(host_data, m_data, copy_count * sizeof(T),
                                        cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("Failed to copy from GPU: ") +
                    cudaGetErrorString(err)
                );
            }
        }
    }

    void copy_from_device(const T* device_data, size_t count) {
        if (count > m_size) {
            resize(count);
        }
        if (m_data && device_data && count > 0) {
            cudaMemcpy(m_data, device_data, count * sizeof(T),
                      cudaMemcpyDeviceToDevice);
        }
    }

    T* data() { return m_data; }
    const T* data() const { return m_data; }

    size_t size() const { return m_size; }
    size_t bytes() const { return m_size * sizeof(T); }

    bool empty() const { return m_size == 0; }

    T* operator->() { return m_data; }
    const T* operator->() const { return m_data; }

private:
    T* m_data;
    size_t m_size;
};

} // namespace ngp

