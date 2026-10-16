#include "cuda_visibility/cuda_raycaster.hpp"

#include <stdexcept>
#include <utility>

namespace cuda_visibility {

struct CudaRaycaster::Impl {};

CudaRaycaster::CudaRaycaster(
    const std::vector<octomap::OcTreeKey> &, double) {
  throw std::runtime_error(
      "CUDA visibility was requested, but cuda_visibility was built without "
      "a CUDA compiler; reconfigure with -DCUDA_VISIBILITY_ENABLE_CUDA=ON");
}

CudaRaycaster::~CudaRaycaster() = default;
CudaRaycaster::CudaRaycaster(CudaRaycaster &&) noexcept = default;
CudaRaycaster &CudaRaycaster::operator=(CudaRaycaster &&) noexcept = default;

CudaRaycastResult CudaRaycaster::cast(
    const std::vector<octomap::point3d> &,
    const std::vector<octomap::point3d> &, double) const {
  throw std::runtime_error("CUDA visibility backend is unavailable");
}

std::string CudaRaycaster::device_name() const { return "unavailable"; }
std::size_t CudaRaycaster::hash_capacity() const { return 0; }
bool cuda_backend_compiled() { return false; }
std::string cuda_runtime_device_name() { return "unavailable"; }

}  // namespace cuda_visibility
