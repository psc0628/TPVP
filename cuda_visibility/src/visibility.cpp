#include "cuda_visibility/visibility.hpp"

#include "cuda_visibility/camera_model.hpp"
#include "cuda_visibility/cuda_raycaster.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace cuda_visibility {
namespace {

struct PreparedRay {
  bool in_front{false};
  bool inside_image{false};
  bool valid{false};
  octomap::point3d origin;
  octomap::point3d direction;
};

PreparedRay prepare_ray(
    const octomap::ColorOcTree &tree,
    const octomap::OcTreeKey &projected_key,
    const CandidateView &view,
    const CameraModel &camera,
    const VisibilityOptions &options,
    const Eigen::Quaterniond &world_from_camera,
    const Eigen::Quaterniond &camera_from_world,
    const octomap::point3d &origin) {
  PreparedRay result;
  result.origin = origin;
  const octomap::point3d center = tree.keyToCoord(projected_key);
  const Eigen::Vector3d point_world(center.x(), center.y(), center.z());
  const Eigen::Vector3d point_camera =
      camera_from_world * (point_world - view.camera_position);
  if (point_camera.z() <= 0.0) return result;
  result.in_front = true;

  Eigen::Vector2d pixel;
  if (!project_camera_point(camera, point_camera, pixel)) return result;
  if (pixel.x() < 0.0 || pixel.x() > static_cast<double>(camera.width) ||
      pixel.y() < 0.0 || pixel.y() > static_cast<double>(camera.height)) {
    return result;
  }
  result.inside_image = true;

  const Eigen::Vector2d legacy_pixel(
      static_cast<int>(pixel.x()), static_cast<int>(pixel.y()));
  const Eigen::Vector3d ray_camera =
      pixel_to_camera_ray(camera, legacy_pixel);
  if (!ray_camera.allFinite() || std::abs(ray_camera.z()) < 1e-18) {
    return result;
  }
  const Eigen::Vector3d endpoint_camera =
      ray_camera * (options.ray_max_distance_m / ray_camera.z());
  const Eigen::Vector3d endpoint_world =
      view.camera_position + world_from_camera * endpoint_camera;
  // Compatibility invariant: projection and the endpoint use the continuous
  // camera pose, but OctoMap casts from the snapped voxel-center origin.
  const Eigen::Vector3d snapped_origin(origin.x(), origin.y(), origin.z());
  const Eigen::Vector3d direction_world = endpoint_world - snapped_origin;
  if (!direction_world.allFinite() || direction_world.squaredNorm() < 1e-18) {
    return result;
  }
  result.direction = octomap::point3d(
      static_cast<float>(direction_world.x()),
      static_cast<float>(direction_world.y()),
      static_cast<float>(direction_world.z()));
  result.valid = true;
  return result;
}

ViewVisibility compute_view_visibility_cuda(
    const octomap::ColorOcTree &tree,
    const std::vector<octomap::OcTreeKey> &covering_keys,
    const CandidateView &view,
    const CameraModel &camera,
    const VisibilityOptions &options,
    const CudaRaycaster &raycaster) {
  ViewVisibility result;
  result.view_id = view.view_id;
  result.covering_voxels_considered = covering_keys.size();
  octomap::OcTreeKey origin_key;
  if (!tree.coordToKeyChecked(view.camera_position.x(),
                              view.camera_position.y(),
                              view.camera_position.z(), origin_key)) {
    return result;
  }
  const octomap::point3d origin = tree.keyToCoord(origin_key);
  const Eigen::Quaterniond world_from_camera = view.camera_orientation;
  const Eigen::Quaterniond camera_from_world = world_from_camera.conjugate();
  std::vector<octomap::point3d> origins;
  std::vector<octomap::point3d> directions;
  origins.reserve(covering_keys.size());
  directions.reserve(covering_keys.size());
  for (const auto &key : covering_keys) {
    const auto ray = prepare_ray(
        tree, key, view, camera, options, world_from_camera,
        camera_from_world, origin);
    result.in_front += ray.in_front ? 1 : 0;
    result.inside_image += ray.inside_image ? 1 : 0;
    if (!ray.valid) continue;
    origins.push_back(ray.origin);
    directions.push_back(ray.direction);
  }
  const auto rays = raycaster.cast(origins, directions,
                                   options.ray_max_distance_m);
  for (std::size_t index = 0; index < rays.found.size(); ++index) {
    if (!rays.found[index] || rays.hit_keys[index] == origin_key) continue;
    const auto *node = tree.search(rays.hit_keys[index]);
    if (node == nullptr || !tree.isNodeOccupied(node)) continue;
    ++result.successful_rays;
    result.hits.insert(rays.hit_keys[index]);
  }
  return result;
}

void require_visibility_parity(const ViewVisibility &cpu,
                               const ViewVisibility &cuda) {
  if (cpu.view_id == cuda.view_id &&
      cpu.covering_voxels_considered == cuda.covering_voxels_considered &&
      cpu.in_front == cuda.in_front &&
      cpu.inside_image == cuda.inside_image &&
      cpu.successful_rays == cuda.successful_rays &&
      cpu.hits == cuda.hits) {
    return;
  }
  std::size_t cpu_only = 0;
  std::size_t cuda_only = 0;
  for (const auto &key : cpu.hits) cpu_only += cuda.hits.count(key) == 0;
  for (const auto &key : cuda.hits) cuda_only += cpu.hits.count(key) == 0;
  throw std::runtime_error(
      "CPU/CUDA visibility mismatch for view_id=" +
      std::to_string(cpu.view_id) + ": cpu_hits=" +
      std::to_string(cpu.hits.size()) + " cuda_hits=" +
      std::to_string(cuda.hits.size()) + " cpu_only=" +
      std::to_string(cpu_only) + " cuda_only=" +
      std::to_string(cuda_only) + " cpu_rays=" +
      std::to_string(cpu.successful_rays) + " cuda_rays=" +
      std::to_string(cuda.successful_rays));
}

}  // namespace

CoveringMap build_covering_map(const std::vector<Eigen::Vector3d> &points,
                               double resolution_m) {
  if (!(resolution_m > 0.0) || !std::isfinite(resolution_m)) {
    throw std::invalid_argument("OctoMap resolution must be positive");
  }
  CoveringMap result;
  result.tree = std::make_shared<octomap::ColorOcTree>(resolution_m);
  for (const auto &point : points) {
    if (!point.allFinite()) continue;
    result.tree->updateNode(octomap::point3d(
                                static_cast<float>(point.x()),
                                static_cast<float>(point.y()),
                                static_cast<float>(point.z())),
                            true, true);
  }
  result.tree->updateInnerOccupancy();
  result.occupied_keys.reserve(result.tree->size());
  for (auto iterator = result.tree->begin_leafs(), end = result.tree->end_leafs();
       iterator != end; ++iterator) {
    if (result.tree->isNodeOccupied(*iterator)) {
      result.occupied_keys.push_back(iterator.getKey());
    }
  }
  if (result.occupied_keys.empty()) {
    throw std::runtime_error("Covering geometry produced an empty OctoMap");
  }
  std::sort(result.occupied_keys.begin(), result.occupied_keys.end(),
            [](const octomap::OcTreeKey &left,
               const octomap::OcTreeKey &right) {
              if (left[0] != right[0]) return left[0] < right[0];
              if (left[1] != right[1]) return left[1] < right[1];
              return left[2] < right[2];
            });
  return result;
}

KeySet map_points_to_occupied_keys(const octomap::ColorOcTree &tree,
                                   const std::vector<Eigen::Vector3d> &points,
                                   std::size_t *outside_or_unoccupied) {
  KeySet result;
  std::size_t rejected = 0;
  for (const auto &point : points) {
    octomap::OcTreeKey key;
    if (!point.allFinite() ||
        !tree.coordToKeyChecked(point.x(), point.y(), point.z(), key)) {
      ++rejected;
      continue;
    }
    const auto *node = tree.search(key);
    if (node == nullptr || !tree.isNodeOccupied(node)) {
      ++rejected;
      continue;
    }
    result.insert(key);
  }
  if (outside_or_unoccupied) *outside_or_unoccupied = rejected;
  return result;
}

ViewVisibility compute_view_visibility(
    const octomap::ColorOcTree &tree,
    const std::vector<octomap::OcTreeKey> &covering_keys,
    const CandidateView &view,
    const CameraModel &camera,
    const VisibilityOptions &options) {
  ViewVisibility result;
  result.view_id = view.view_id;
  result.covering_voxels_considered = covering_keys.size();
  octomap::OcTreeKey origin_key;
  if (!tree.coordToKeyChecked(view.camera_position.x(),
                              view.camera_position.y(),
                              view.camera_position.z(), origin_key)) {
    return result;
  }
  // The simulator snaps the camera origin to the covering OctoMap cell center
  // before casting any ray. Preserve that discretization exactly.
  const octomap::point3d origin = tree.keyToCoord(origin_key);
  const Eigen::Quaterniond world_from_camera = view.camera_orientation;
  const Eigen::Quaterniond camera_from_world = world_from_camera.conjugate();

  struct PerVoxelResult {
    bool in_front{false};
    bool inside_image{false};
    bool successful_ray{false};
    bool has_hit{false};
    octomap::OcTreeKey hit_key;
  };
  std::vector<PerVoxelResult> per_voxel(covering_keys.size());
  std::vector<std::thread> threads;
  threads.reserve(covering_keys.size());
  const std::size_t batch_size = static_cast<std::size_t>(
      std::max(1, options.max_threads_per_batch));

  auto process_voxel = [&](std::size_t index) {
    auto &voxel_result = per_voxel[index];
    const auto ray = prepare_ray(
        tree, covering_keys[index], view, camera, options,
        world_from_camera, camera_from_world, origin);
    voxel_result.in_front = ray.in_front;
    voxel_result.inside_image = ray.inside_image;
    if (!ray.valid) return;
    octomap::point3d hit;
    const bool found = tree.castRay(ray.origin, ray.direction, hit, true,
                                    options.ray_max_distance_m);
    if (!found || hit == origin) return;
    octomap::OcTreeKey hit_key;
    if (!tree.coordToKeyChecked(hit, hit_key)) return;
    const auto *node = tree.search(hit_key);
    if (node == nullptr || !tree.isNodeOccupied(node)) return;
    voxel_result.successful_ray = true;
    voxel_result.has_hit = true;
    voxel_result.hit_key = hit_key;
    // Deliberately do not require hit_key == projected_key.
  };

  // Strict simulator reproduction: create one std::thread per covering voxel,
  // in batches, rather than using a persistent worker pool. Joined thread
  // objects remain in the vector until the view has completed, matching the
  // legacy Perception_3D::precept implementation.
  for (std::size_t begin = 0; begin < covering_keys.size();
       begin += batch_size) {
    const std::size_t end =
        std::min(covering_keys.size(), begin + batch_size);
    for (std::size_t index = begin; index < end; ++index) {
      threads.emplace_back(process_voxel, index);
    }
    for (std::size_t index = begin; index < end; ++index) {
      threads[index].join();
    }
  }
  for (const auto &voxel_result : per_voxel) {
    result.in_front += voxel_result.in_front ? 1 : 0;
    result.inside_image += voxel_result.inside_image ? 1 : 0;
    result.successful_rays += voxel_result.successful_ray ? 1 : 0;
    if (voxel_result.has_hit) result.hits.insert(voxel_result.hit_key);
  }
  return result;
}

std::vector<ViewVisibility> compute_all_visibilities(
    const CoveringMap &covering,
    const std::vector<CandidateView> &views,
    const CameraModel &camera,
    const VisibilityOptions &options,
    VisibilityRuntime *runtime) {
  if (!covering.tree) throw std::invalid_argument("Missing covering OctoMap");
  if (!(options.ray_max_distance_m > 0.0) ||
      !std::isfinite(options.ray_max_distance_m)) {
    throw std::invalid_argument("Ray maximum distance must be finite and positive");
  }
  if (camera.width <= 0 || camera.height <= 0 ||
      !(camera.fx > 0.0) || !(camera.fy > 0.0) ||
      !std::isfinite(camera.fx) || !std::isfinite(camera.fy) ||
      !std::isfinite(camera.cx) || !std::isfinite(camera.cy) ||
      !std::isfinite(camera.k1) || !std::isfinite(camera.k2) ||
      !std::isfinite(camera.p1) || !std::isfinite(camera.p2) ||
      !std::isfinite(camera.k3)) {
    throw std::invalid_argument("Invalid camera calibration");
  }
  for (const auto &view : views) {
    if (!view.camera_position.allFinite() ||
        !view.camera_orientation.coeffs().allFinite() ||
        std::abs(view.camera_orientation.norm() - 1.0) > 1e-6) {
      throw std::invalid_argument("Camera poses require finite positions and unit quaternions");
    }
  }
  if (options.backend != "cpu" && options.backend != "cuda" &&
      options.backend != "verify") {
    throw std::invalid_argument(
        "visibility backend must be cpu, cuda, or verify");
  }
  std::unique_ptr<CudaRaycaster> raycaster;
  if (options.backend != "cpu") {
    raycaster = std::make_unique<CudaRaycaster>(
        covering.occupied_keys, covering.tree->getResolution());
    const auto progress_start = std::chrono::steady_clock::now();
    if (options.verbose) std::cout << "[visibility] CUDA device=" << raycaster->device_name()
              << " occupied_leaf_keys=" << covering.occupied_keys.size()
              << " hash_capacity=" << raycaster->hash_capacity() << std::endl;
    if (runtime != nullptr) {
      runtime->progress_reporting_s +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        progress_start)
              .count();
    }
  }
  std::vector<ViewVisibility> output;
  output.reserve(views.size());
  // The simulator evaluates views sequentially. Parallelism exists only in
  // the per-voxel thread batches inside compute_view_visibility().
  const auto all_start = std::chrono::steady_clock::now();
  for (std::size_t index = 0; index < views.size(); ++index) {
    const auto &view = views[index];
    const auto progress_start = std::chrono::steady_clock::now();
    if (options.verbose) std::cout << "[visibility] processing " << (index + 1) << '/'
              << views.size() << " view_id=" << view.view_id << std::endl;
    if (runtime != nullptr) {
      runtime->progress_reporting_s +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        progress_start)
              .count();
    }
    const auto view_start = std::chrono::steady_clock::now();
    if (options.backend == "cpu") {
      output.push_back(compute_view_visibility(
          *covering.tree, covering.occupied_keys, view, camera, options));
    } else if (options.backend == "cuda") {
      output.push_back(compute_view_visibility_cuda(
          *covering.tree, covering.occupied_keys, view, camera, options,
          *raycaster));
    } else {
      const auto cpu = compute_view_visibility(
          *covering.tree, covering.occupied_keys, view, camera, options);
      const auto cuda = compute_view_visibility_cuda(
          *covering.tree, covering.occupied_keys, view, camera, options,
          *raycaster);
      require_visibility_parity(cpu, cuda);
      output.push_back(cuda);
    }
    const auto view_end = std::chrono::steady_clock::now();
    const double view_seconds =
        std::chrono::duration<double>(view_end - view_start).count();
    const double elapsed_seconds =
        std::chrono::duration<double>(view_end - all_start).count();
    const double average_seconds = elapsed_seconds / (index + 1);
    const double eta_seconds = average_seconds * (views.size() - index - 1);
    const auto completed_progress_start = std::chrono::steady_clock::now();
    if (options.verbose) std::cout << std::fixed << std::setprecision(1)
              << "[visibility] completed  " << (index + 1) << '/'
              << views.size() << " view_id=" << view.view_id
              << " hits=" << output.back().hits.size()
              << " rays=" << output.back().successful_rays
              << " view=" << view_seconds << "s"
              << " elapsed=" << elapsed_seconds << "s"
              << " eta=" << eta_seconds << "s" << std::endl;
    if (runtime != nullptr) {
      runtime->progress_reporting_s +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        completed_progress_start)
              .count();
    }
  }
  return output;
}

KeySet union_hits(const std::vector<ViewVisibility> &visibilities,
                  int confidence_count) {
  std::unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>
      counts;
  for (const auto &visibility : visibilities) {
    for (const auto &key : visibility.hits) ++counts[key];
  }
  KeySet result;
  for (const auto &[key, count] : counts) {
    if (count > confidence_count) result.insert(key);
  }
  return result;
}

std::vector<KeySet> make_active_hit_sets(
    const std::vector<ViewVisibility> &visibilities,
    const KeySet &precovered,
    int confidence_count,
    KeySet *active_demand) {
  const KeySet confident = union_hits(visibilities, confidence_count);
  std::vector<KeySet> result(visibilities.size());
  KeySet demand;
  for (std::size_t view_index = 0; view_index < visibilities.size();
       ++view_index) {
    for (const auto &key : visibilities[view_index].hits) {
      if (confident.count(key) == 0 || precovered.count(key) != 0) continue;
      result[view_index].insert(key);
      demand.insert(key);
    }
  }
  if (active_demand) *active_demand = std::move(demand);
  return result;
}

std::vector<Eigen::Vector3d> key_centers(const octomap::ColorOcTree &tree,
                                         const KeySet &keys) {
  std::vector<Eigen::Vector3d> points;
  points.reserve(keys.size());
  for (const auto &key : keys) {
    const auto center = tree.keyToCoord(key);
    points.emplace_back(center.x(), center.y(), center.z());
  }
  std::sort(points.begin(), points.end(),
            [](const Eigen::Vector3d &left, const Eigen::Vector3d &right) {
              if (left.x() != right.x()) return left.x() < right.x();
              if (left.y() != right.y()) return left.y() < right.y();
              return left.z() < right.z();
            });
  return points;
}

}  // namespace cuda_visibility
