#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <octomap/OcTreeKey.h>
#include <octomap/octomap.h>

namespace cuda_visibility {

struct CudaRaycastResult {
  std::vector<std::uint8_t> found;
  std::vector<octomap::OcTreeKey> hit_keys;
};

class CudaRaycaster {
 public:
  explicit CudaRaycaster(
      const std::vector<octomap::OcTreeKey> &occupied_leaf_keys,
      double resolution_m);
  ~CudaRaycaster();

  CudaRaycaster(const CudaRaycaster &) = delete;
  CudaRaycaster &operator=(const CudaRaycaster &) = delete;
  CudaRaycaster(CudaRaycaster &&) noexcept;
  CudaRaycaster &operator=(CudaRaycaster &&) noexcept;

  CudaRaycastResult cast(
      const std::vector<octomap::point3d> &origins,
      const std::vector<octomap::point3d> &directions,
      double max_range_m) const;

  std::string device_name() const;
  std::size_t hash_capacity() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

bool cuda_backend_compiled();
std::string cuda_runtime_device_name();

}  // namespace cuda_visibility
