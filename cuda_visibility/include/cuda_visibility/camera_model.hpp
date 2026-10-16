#pragma once

#include "cuda_visibility/types.hpp"

namespace cuda_visibility {

bool project_camera_point(const CameraModel &camera,
                          const Eigen::Vector3d &point_camera,
                          Eigen::Vector2d &pixel);

Eigen::Vector3d pixel_to_camera_ray(const CameraModel &camera,
                                   const Eigen::Vector2d &pixel);

}  // namespace cuda_visibility
