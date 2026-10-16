#include "cuda_visibility/cuda_raycaster.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cuda_visibility {
namespace {

using DeviceKey = unsigned long long;
constexpr DeviceKey kEmpty = std::numeric_limits<DeviceKey>::max();
constexpr int kTreeMaxVal = 32768;

struct DevicePoint {
  float x;
  float y;
  float z;
};

void check_cuda(cudaError_t status, const char *operation) {
  if (status == cudaSuccess) return;
  std::ostringstream message;
  message << operation << ": " << cudaGetErrorString(status);
  throw std::runtime_error(message.str());
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ != 0) {
      check_cuda(cudaMalloc(reinterpret_cast<void **>(&data_),
                            count_ * sizeof(T)),
                 "cudaMalloc");
    }
  }
  ~DeviceBuffer() {
    if (data_ != nullptr) cudaFree(data_);
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  T *get() const { return data_; }
  std::size_t size_bytes() const { return count_ * sizeof(T); }

 private:
  T *data_{nullptr};
  std::size_t count_{0};
};

__host__ __device__ DeviceKey encode_key(
    std::uint16_t x, std::uint16_t y, std::uint16_t z) {
  return (static_cast<DeviceKey>(x) << 32) |
         (static_cast<DeviceKey>(y) << 16) | static_cast<DeviceKey>(z);
}

__host__ __device__ DeviceKey hash_key(DeviceKey key,
                                       DeviceKey capacity_mask) {
  key ^= key >> 16;
  key *= 0x85ebca6bULL;
  key ^= key >> 13;
  key *= 0xc2b2ae35ULL;
  key ^= key >> 16;
  return key & capacity_mask;
}

__device__ DeviceKey query(const DeviceKey *table, DeviceKey capacity_mask,
                           DeviceKey key) {
  DeviceKey slot = hash_key(key, capacity_mask);
  while (true) {
    const DeviceKey stored = table[slot];
    if (stored == kEmpty || stored == key) return stored;
    slot = (slot + 1) & capacity_mask;
  }
}

__global__ void insert_keys(DeviceKey *table, DeviceKey capacity_mask,
                            const DeviceKey *keys, std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const DeviceKey key = keys[index];
  DeviceKey slot = hash_key(key, capacity_mask);
  while (true) {
    auto *address = reinterpret_cast<unsigned long long *>(&table[slot]);
    const DeviceKey previous = atomicCAS(address, kEmpty, key);
    if (previous == kEmpty || previous == key) return;
    slot = (slot + 1) & capacity_mask;
  }
}

__device__ std::uint16_t coord_to_key(double coordinate,
                                      double resolution_factor) {
  return static_cast<std::uint16_t>(
      static_cast<int>(floor(resolution_factor * coordinate)) + kTreeMaxVal);
}

__device__ double key_to_coord(std::uint16_t key, double resolution) {
  return (static_cast<double>(static_cast<int>(key) - kTreeMaxVal) + 0.5) *
         resolution;
}

__global__ void cast_rays(const DeviceKey *table, DeviceKey capacity_mask,
                          double resolution, const DevicePoint *origins,
                          const DevicePoint *directions, std::size_t count,
                          double max_range, std::uint8_t *found,
                          DeviceKey *hit_keys) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;

  const DevicePoint origin = origins[index];
  DevicePoint direction = directions[index];
  const double norm = sqrt(static_cast<double>(direction.x) * direction.x +
                           static_cast<double>(direction.y) * direction.y +
                           static_cast<double>(direction.z) * direction.z);
  if (!(norm > 0.0) || !isfinite(norm)) return;
  direction.x = static_cast<float>(direction.x / norm);
  direction.y = static_cast<float>(direction.y / norm);
  direction.z = static_cast<float>(direction.z / norm);

  const double resolution_factor = 1.0 / resolution;
  std::uint16_t current[3] = {
      coord_to_key(origin.x, resolution_factor),
      coord_to_key(origin.y, resolution_factor),
      coord_to_key(origin.z, resolution_factor)};
  DeviceKey current_index = encode_key(current[0], current[1], current[2]);
  if (query(table, capacity_mask, current_index) == current_index) {
    found[index] = 1;
    hit_keys[index] = current_index;
    return;
  }

  const double origin_values[3] = {origin.x, origin.y, origin.z};
  const double direction_values[3] = {
      direction.x, direction.y, direction.z};
  int step[3];
  double t_max[3];
  double t_delta[3];
  for (int axis = 0; axis < 3; ++axis) {
    step[axis] = direction_values[axis] > 0.0
                     ? 1
                     : (direction_values[axis] < 0.0 ? -1 : 0);
    if (step[axis] == 0) {
      t_max[axis] = DBL_MAX;
      t_delta[axis] = DBL_MAX;
      continue;
    }
    double border = key_to_coord(current[axis], resolution);
    border += static_cast<double>(step[axis]) * resolution * 0.5;
    t_max[axis] =
        (border - origin_values[axis]) / direction_values[axis];
    t_delta[axis] = resolution / fabs(direction_values[axis]);
  }

  const float max_range_sq = static_cast<float>(max_range * max_range);
  while (true) {
    int axis;
    if (t_max[0] < t_max[1]) {
      axis = t_max[0] < t_max[2] ? 0 : 2;
    } else {
      axis = t_max[1] < t_max[2] ? 1 : 2;
    }
    if ((step[axis] < 0 && current[axis] == 0) ||
        (step[axis] > 0 && current[axis] == 65535)) {
      return;
    }
    current[axis] = static_cast<std::uint16_t>(
        static_cast<int>(current[axis]) + step[axis]);
    t_max[axis] += t_delta[axis];

    float distance_sq = 0.0F;
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      const float center = static_cast<float>(
          key_to_coord(current[coordinate], resolution));
      const float delta = center - static_cast<float>(origin_values[coordinate]);
      distance_sq += delta * delta;
    }
    if (distance_sq > max_range_sq) return;

    current_index = encode_key(current[0], current[1], current[2]);
    if (query(table, capacity_mask, current_index) == current_index) {
      found[index] = 1;
      hit_keys[index] = current_index;
      return;
    }
  }
}

std::size_t next_power_of_two(std::size_t value) {
  std::size_t result = 1;
  while (result < value) {
    if (result > std::numeric_limits<std::size_t>::max() / 2) {
      throw std::overflow_error("CUDA hash-table capacity overflow");
    }
    result *= 2;
  }
  return result;
}

std::vector<DeviceKey> encode_keys(
    const std::vector<octomap::OcTreeKey> &keys) {
  std::vector<DeviceKey> encoded;
  encoded.reserve(keys.size());
  for (const auto &key : keys) {
    encoded.push_back(encode_key(key[0], key[1], key[2]));
  }
  std::sort(encoded.begin(), encoded.end());
  encoded.erase(std::unique(encoded.begin(), encoded.end()), encoded.end());
  return encoded;
}

octomap::OcTreeKey decode_key(DeviceKey key) {
  return octomap::OcTreeKey(
      static_cast<std::uint16_t>((key >> 32) & 0xffffULL),
      static_cast<std::uint16_t>((key >> 16) & 0xffffULL),
      static_cast<std::uint16_t>(key & 0xffffULL));
}

}  // namespace

struct CudaRaycaster::Impl {
  DeviceKey *table{nullptr};
  std::size_t capacity{0};
  double resolution{0.0};
  std::string device;

  ~Impl() {
    if (table != nullptr) cudaFree(table);
  }
};

CudaRaycaster::CudaRaycaster(
    const std::vector<octomap::OcTreeKey> &occupied_leaf_keys,
    double resolution_m)
    : impl_(std::make_unique<Impl>()) {
  if (!(resolution_m > 0.0) || !std::isfinite(resolution_m)) {
    throw std::invalid_argument("CUDA raycaster resolution must be positive");
  }
  impl_->resolution = resolution_m;
  const auto encoded = encode_keys(occupied_leaf_keys);
  if (encoded.empty()) {
    throw std::invalid_argument("CUDA raycaster requires occupied leaf keys");
  }
  impl_->capacity = next_power_of_two(std::max<std::size_t>(1024, encoded.size() * 2));
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&impl_->table),
                        impl_->capacity * sizeof(DeviceKey)),
             "cudaMalloc hash table");
  check_cuda(cudaMemset(impl_->table, 0xff,
                        impl_->capacity * sizeof(DeviceKey)),
             "cudaMemset hash table");

  DeviceBuffer<DeviceKey> device_keys(encoded.size());
  check_cuda(cudaMemcpy(device_keys.get(), encoded.data(),
                        device_keys.size_bytes(), cudaMemcpyHostToDevice),
             "copy occupied keys to CUDA");
  constexpr int threads = 256;
  const int blocks = static_cast<int>((encoded.size() + threads - 1) / threads);
  insert_keys<<<blocks, threads>>>(impl_->table, impl_->capacity - 1,
                                  device_keys.get(), encoded.size());
  check_cuda(cudaGetLastError(), "launch CUDA hash insertion");
  check_cuda(cudaDeviceSynchronize(), "synchronize CUDA hash insertion");

  int device_index = 0;
  check_cuda(cudaGetDevice(&device_index), "cudaGetDevice");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device_index),
             "cudaGetDeviceProperties");
  impl_->device = properties.name;
}

CudaRaycaster::~CudaRaycaster() = default;
CudaRaycaster::CudaRaycaster(CudaRaycaster &&) noexcept = default;
CudaRaycaster &CudaRaycaster::operator=(CudaRaycaster &&) noexcept = default;

CudaRaycastResult CudaRaycaster::cast(
    const std::vector<octomap::point3d> &origins,
    const std::vector<octomap::point3d> &directions,
    double max_range_m) const {
  if (!impl_ || origins.size() != directions.size()) {
    throw std::invalid_argument("CUDA ray arrays have different sizes");
  }
  if (!(max_range_m > 0.0) || !std::isfinite(max_range_m)) {
    throw std::invalid_argument("CUDA ray max range must be positive");
  }
  CudaRaycastResult result;
  result.found.assign(origins.size(), 0);
  result.hit_keys.resize(origins.size());
  if (origins.empty()) return result;

  std::vector<DevicePoint> host_origins;
  std::vector<DevicePoint> host_directions;
  host_origins.reserve(origins.size());
  host_directions.reserve(directions.size());
  for (std::size_t index = 0; index < origins.size(); ++index) {
    host_origins.push_back(
        {origins[index].x(), origins[index].y(), origins[index].z()});
    host_directions.push_back({directions[index].x(), directions[index].y(),
                               directions[index].z()});
  }

  DeviceBuffer<DevicePoint> device_origins(origins.size());
  DeviceBuffer<DevicePoint> device_directions(directions.size());
  DeviceBuffer<std::uint8_t> device_found(origins.size());
  DeviceBuffer<DeviceKey> device_hits(origins.size());
  check_cuda(cudaMemcpy(device_origins.get(), host_origins.data(),
                        device_origins.size_bytes(), cudaMemcpyHostToDevice),
             "copy CUDA ray origins");
  check_cuda(cudaMemcpy(device_directions.get(), host_directions.data(),
                        device_directions.size_bytes(), cudaMemcpyHostToDevice),
             "copy CUDA ray directions");
  check_cuda(cudaMemset(device_found.get(), 0, device_found.size_bytes()),
             "clear CUDA ray results");

  constexpr int threads = 256;
  const int blocks = static_cast<int>((origins.size() + threads - 1) / threads);
  cast_rays<<<blocks, threads>>>(
      impl_->table, impl_->capacity - 1,
      impl_->resolution, device_origins.get(), device_directions.get(),
      origins.size(), max_range_m, device_found.get(), device_hits.get());
  check_cuda(cudaGetLastError(), "launch CUDA ray casting");
  check_cuda(cudaDeviceSynchronize(), "synchronize CUDA ray casting");

  std::vector<DeviceKey> host_hits(origins.size(), 0);
  check_cuda(cudaMemcpy(result.found.data(), device_found.get(),
                        device_found.size_bytes(), cudaMemcpyDeviceToHost),
             "copy CUDA hit flags");
  check_cuda(cudaMemcpy(host_hits.data(), device_hits.get(),
                        device_hits.size_bytes(), cudaMemcpyDeviceToHost),
             "copy CUDA hit keys");
  for (std::size_t index = 0; index < host_hits.size(); ++index) {
    if (result.found[index]) result.hit_keys[index] = decode_key(host_hits[index]);
  }
  return result;
}

std::string CudaRaycaster::device_name() const {
  return impl_ ? impl_->device : "unavailable";
}

std::size_t CudaRaycaster::hash_capacity() const {
  return impl_ ? impl_->capacity : 0;
}

bool cuda_backend_compiled() { return true; }

std::string cuda_runtime_device_name() {
  int device_index = 0;
  check_cuda(cudaGetDevice(&device_index), "cudaGetDevice");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device_index),
             "cudaGetDeviceProperties");
  return properties.name;
}

}  // namespace cuda_visibility
