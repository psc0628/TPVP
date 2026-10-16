#include "cuda_visibility/camera_model.hpp"

#include <cmath>

namespace cuda_visibility {

bool project_camera_point(const CameraModel &camera,
                          const Eigen::Vector3d &point_camera,
                          Eigen::Vector2d &pixel) {
  if (!point_camera.allFinite() || point_camera.z() <= 0.0) return false;
  const double x = point_camera.x() / point_camera.z();
  const double y = point_camera.y() / point_camera.z();
  const double r2 = x * x + y * y;
  const double radial =
      1.0 + camera.k1 * r2 + camera.k2 * r2 * r2 +
      camera.k3 * r2 * r2 * r2;
  const double xd = x * radial + 2.0 * camera.p1 * x * y +
                    camera.p2 * (r2 + 2.0 * x * x);
  const double yd = y * radial + camera.p1 * (r2 + 2.0 * y * y) +
                    2.0 * camera.p2 * x * y;
  pixel = Eigen::Vector2d(camera.fx * xd + camera.cx,
                          camera.fy * yd + camera.cy);
  return pixel.allFinite();
}

Eigen::Vector3d pixel_to_camera_ray(const CameraModel &camera,
                                    const Eigen::Vector2d &pixel) {
  const double xd = (pixel.x() - camera.cx) / camera.fx;
  const double yd = (pixel.y() - camera.cy) / camera.fy;
  double x = xd;
  double y = yd;
  // Fixed-point inverse of ROS plumb_bob distortion. This is deterministic
  // and converges rapidly for the D435 coefficients used in this project.
  for (int iteration = 0; iteration < 10; ++iteration) {
    const double r2 = x * x + y * y;
    const double radial =
        1.0 + camera.k1 * r2 + camera.k2 * r2 * r2 +
        camera.k3 * r2 * r2 * r2;
    if (std::abs(radial) < 1e-12) break;
    const double dx = 2.0 * camera.p1 * x * y +
                      camera.p2 * (r2 + 2.0 * x * x);
    const double dy = camera.p1 * (r2 + 2.0 * y * y) +
                      2.0 * camera.p2 * x * y;
    x = (xd - dx) / radial;
    y = (yd - dy) / radial;
  }
  return Eigen::Vector3d(x, y, 1.0).normalized();
}

}  // namespace cuda_visibility

