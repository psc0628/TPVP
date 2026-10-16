#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <octomap/OcTreeKey.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace cuda_visibility {

using KeySet = std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash>;

struct CameraModel {
  int width{0};
  int height{0};
  double fx{0.0}, fy{0.0}, cx{0.0}, cy{0.0};
  // Optional plumb_bob distortion: k1, k2, p1, p2, k3.
  double k1{0.0}, k2{0.0}, p1{0.0}, p2{0.0}, k3{0.0};
};

struct CandidateView {
  std::int64_t view_id{-1};
  Eigen::Vector3d camera_position{Eigen::Vector3d::Zero()};
  // Rotation from optical camera frame (+X right, +Y down, +Z forward)
  // to the common world frame. Must be a unit quaternion.
  Eigen::Quaterniond camera_orientation{Eigen::Quaterniond::Identity()};
};

struct VisibilityOptions {
  double resolution_m{0.04};
  double ray_max_distance_m{2.0};
  int confidence_count{0};
  // CPU reference reproduces the original per-voxel thread batches.
  int max_threads_per_batch{1000};
  std::string backend{"cpu"};  // cpu, cuda, or verify
  bool verbose{false};
};

struct ViewVisibility {
  std::int64_t view_id{-1};
  KeySet hits;
  std::size_t covering_voxels_considered{0};
  std::size_t in_front{0};
  std::size_t inside_image{0};
  std::size_t successful_rays{0};
};

}  // namespace cuda_visibility
