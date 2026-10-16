#pragma once

#include "cuda_visibility/types.hpp"

#include <Eigen/Core>
#include <octomap/ColorOcTree.h>

#include <memory>
#include <vector>

namespace cuda_visibility {

struct CoveringMap {
  std::shared_ptr<octomap::ColorOcTree> tree;
  std::vector<octomap::OcTreeKey> occupied_keys;
};

struct VisibilityRuntime {
  // Console progress is useful operationally but is not part of the
  // visibility operator. The caller subtracts it from formal algorithm time.
  double progress_reporting_s{0.0};
};

CoveringMap build_covering_map(const std::vector<Eigen::Vector3d> &points,
                               double resolution_m);

KeySet map_points_to_occupied_keys(const octomap::ColorOcTree &tree,
                                   const std::vector<Eigen::Vector3d> &points,
                                   std::size_t *outside_or_unoccupied = nullptr);

ViewVisibility compute_view_visibility(
    const octomap::ColorOcTree &tree,
    const std::vector<octomap::OcTreeKey> &covering_keys,
    const CandidateView &view,
    const CameraModel &camera,
    const VisibilityOptions &options);

std::vector<ViewVisibility> compute_all_visibilities(
    const CoveringMap &covering,
    const std::vector<CandidateView> &views,
    const CameraModel &camera,
    const VisibilityOptions &options,
    VisibilityRuntime *runtime = nullptr);

KeySet union_hits(const std::vector<ViewVisibility> &visibilities,
                  int confidence_count);

std::vector<KeySet> make_active_hit_sets(
    const std::vector<ViewVisibility> &visibilities,
    const KeySet &precovered,
    int confidence_count,
    KeySet *active_demand = nullptr);

std::vector<Eigen::Vector3d> key_centers(const octomap::ColorOcTree &tree,
                                         const KeySet &keys);

}  // namespace cuda_visibility
