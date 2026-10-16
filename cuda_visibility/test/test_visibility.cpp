#include "cuda_visibility/camera_model.hpp"
#include "cuda_visibility/cuda_raycaster.hpp"
#include "cuda_visibility/visibility.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace cv = cuda_visibility;

namespace {

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

cv::CameraModel camera() {
  cv::CameraModel value;
  value.width = 640; value.height = 480;
  value.fx = 500; value.fy = 500; value.cx = 320; value.cy = 240;
  return value;
}

void occlusion_and_precoverage(const std::string &backend) {
  const std::vector<Eigen::Vector3d> geometry = {
      {0.0, 0.0, 1.0}, {0.0, 0.0, 2.0}, {0.2, 0.0, 1.0},
      {-0.2, 0.0, 1.0}, {0.0, 0.2, 1.0}, {0.0, -0.2, 1.0}};
  const auto map = cv::build_covering_map(geometry, 0.05);
  cv::CandidateView view;
  view.view_id = 17;
  cv::VisibilityOptions options;
  options.backend = backend;
  options.ray_max_distance_m = 3.0;
  const auto result = cv::compute_all_visibilities(map, {view}, camera(), options);
  const auto near_key = map.tree->coordToKey(0.0, 0.0, 1.0);
  const auto far_key = map.tree->coordToKey(0.0, 0.0, 2.0);
  require(result[0].hits.count(near_key) == 1, "Foreground surface not observed");
  require(result[0].hits.count(far_key) == 0, "Occluded surface marked visible");
  cv::KeySet demand;
  const auto active = cv::make_active_hit_sets(result, {near_key}, 0, &demand);
  require(active[0].count(near_key) == 0 && demand.count(near_key) == 0,
          "Passive surface remained in active demand");
  require(map.tree->isNodeOccupied(map.tree->search(near_key)),
          "Passive surface removed from occlusion map");
  const auto repeated = cv::union_hits({result[0], result[0]}, 1);
  require(repeated == result[0].hits, "Shared voxels were not deduplicated");

  // A shorter range must remove the foreground hits too.
  options.ray_max_distance_m = 0.5;
  const auto short_range = cv::compute_all_visibilities(map, {view}, camera(), options);
  require(short_range[0].hits.empty(), "Maximum range was not respected");
}

void rotated_views_and_distortion(const std::string &backend) {
  std::vector<Eigen::Vector3d> points;
  for (int i = -5; i <= 5; ++i) {
    for (int j = -4; j <= 4; ++j) {
      points.emplace_back(i * 0.08 + 0.013, j * 0.08 + 0.017, 1.813);
      if (i % 2 == 0) points.emplace_back(i * 0.06 + 0.013, j * 0.06 + 0.017, 1.013);
    }
  }
  const auto map = cv::build_covering_map(points, 0.04);
  std::vector<cv::CandidateView> views;
  for (int i = 0; i < 4; ++i) {
    cv::CandidateView view;
    view.view_id = i;
    view.camera_position = Eigen::Vector3d(-0.21 + i * 0.17, 0.03, 0.01);
    view.camera_orientation = Eigen::Quaterniond(
        Eigen::AngleAxisd((i - 1.5) * 0.12, Eigen::Vector3d::UnitY()));
    views.push_back(view);
  }
  auto model = camera();
  model.k1 = -0.015; model.k2 = 0.003; model.p1 = 0.0002; model.p2 = -0.0001;
  cv::VisibilityOptions options;
  options.backend = backend;
  options.ray_max_distance_m = 3.0;
  const auto result = cv::compute_all_visibilities(map, views, model, options);
  for (const auto &view : result) require(!view.hits.empty(), "Rotated view is unexpectedly empty");

  Eigen::Vector2d pixel;
  const Eigen::Vector3d point(0.1, -0.15, 1.0);
  require(cv::project_camera_point(model, point, pixel), "Distorted projection failed");
  require((cv::pixel_to_camera_ray(model, pixel) - point.normalized()).norm() < 1e-10,
          "Distortion round-trip failed");

  views[0].camera_orientation.coeffs() *= 2.0;
  bool rejected = false;
  try { cv::compute_all_visibilities(map, views, model, options); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "Non-unit camera quaternion was silently accepted");
}

// Exercise axis-aligned rays, negative grid coordinates, DDA ties, range
// boundaries, and an occupied ray origin independently of camera projection.
void raycast_edges() {
  const std::vector<Eigen::Vector3d> points = {
      {0.025, 0.025, 1.025}, {-0.975, 0.025, 0.025},
      {0.025, -0.975, 0.025}, {0.525, 0.525, 0.525},
      {-0.525, -0.525, -0.525}, {0.025, 0.025, -0.975}};
  const auto map = cv::build_covering_map(points, 0.05);
  cv::CudaRaycaster gpu(map.occupied_keys, 0.05);
  const std::vector<octomap::point3d> directions = {
      {0, 0, 1}, {-1, 0, 0}, {0, -1, 0}, {1, 1, 1},
      {-1, -1, -1}, {0, 0, -1}, {1, 0, 0}};
  for (const double range : {0.50, 1.0, 2.0}) {
    std::vector<octomap::point3d> origins(directions.size(), {0.025F, 0.025F, 0.025F});
    origins.back() = map.tree->keyToCoord(map.occupied_keys.front());
    const auto result = gpu.cast(origins, directions, range);
    for (std::size_t i = 0; i < directions.size(); ++i) {
      octomap::point3d hit;
      const bool found = map.tree->castRay(origins[i], directions[i], hit, true, range);
      require(found == (result.found[i] != 0), "CPU/CUDA edge-ray hit mismatch");
      if (found) require(map.tree->coordToKey(hit) == result.hit_keys[i],
                         "CPU/CUDA edge-ray key mismatch");
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const std::string backend = argc > 1 ? argv[1] : "cpu";
    occlusion_and_precoverage(backend);
    rotated_views_and_distortion(backend);
    if (backend == "verify") raycast_edges();
    std::cout << "Visibility checks passed (" << backend << ")\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Test failed: " << error.what() << '\n';
    return 1;
  }
}
