# CUDA visibility evaluation

Standalone CPU/CUDA visibility evaluation extracted from the real-world TPVP
view-selection implementation. Copy this entire directory into another
repository; it has no references to its parent project. Exclude generated
`build*/` and `results/` directories when copying source files.

The module computes a set of visible occupied voxel keys for each candidate
camera pose. It also provides helpers to subtract current-session passive
precoverage and build the demand sets for a downstream set-cover solver.
It does not perform candidate sampling, registration, robot control, or set-cover
optimization.

## Dependencies

- CMake >= 3.18 and a C++17 compiler.
- Eigen >= 3.3.
- OctoMap >= 1.10 (the reference implementation was evaluated with 1.10.0).
- NVIDIA CUDA toolkit and a compatible GPU for the CUDA/verification modes.

No ROS, MoveIt, Gurobi, Open3D, Python, or yaml-cpp is required.
CPU-only builds do not require CUDA.
The supplied commands target Linux; WSL2 with GPU support also works.

## Build and run

Run from this directory. Set `CMAKE_CUDA_ARCHITECTURES` for your GPU; 89 is used
for the RTX 4060 Laptop GPU. If CUDA is outside PATH, add
`-DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc` to the configure command.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j 4
ctest --test-dir build --output-on-failure

# No external data required: generate two partially occluding surfaces and six views.
./build/visibility_example --backend verify

# Warm CUDA, time both backends, and verify their outputs.
./build/visibility_example --backend benchmark --repeats 3
```

For a custom OctoMap install, supply its package directory with
`-Doctomap_DIR=/path/to/octomap/share/octomap`.
For a CPU-only build:

```bash
cmake -S . -B build_cpu -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_VISIBILITY_ENABLE_CUDA=OFF
cmake --build build_cpu -j 4
ctest --test-dir build_cpu --output-on-failure
./build_cpu/visibility_example --backend cpu
```

`verify` requires CUDA; it never silently falls back to CPU.

## File-based example

```bash
./build/visibility_example --backend verify \
  --geometry examples/geometry.xyz \
  --views examples/views.csv \
  --camera examples/camera.txt \
  --precovered examples/passive.xyz \
  --resolution 0.04 --range 3.0 \
  --output results/hits.csv
```

All coordinates are in metres in one common world frame.
Plain text XYZ inputs contain exactly `x y z` per line, without a PLY/PCD header.
Blank lines and `#` comments are allowed.
The full covering geometry must include both passive and transferred/inflated
surfaces so passive surfaces continue to occlude rays.
`--precovered` is optional and changes demand sets, not the occlusion geometry.

The pose CSV header is exactly:

```text
view_id,x,y,z,qx,qy,qz,qw
```

IDs must be unique. Quaternions must be unit length and rotate from the optical
camera frame to the world frame. Optical axes are +X right, +Y down, +Z forward.
These are camera poses, not end-effector poses or look-at target coordinates.

The camera file has one data line:

```text
width height fx fy cx cy [k1 k2 p1 p2 k3]
```

The optional five coefficients use the `plumb_bob` distortion convention.
Omitting them gives a pinhole camera.

The output CSV contains one row per unique visible voxel per view:

```text
view_id,key_x,key_y,key_z,world_x,world_y,world_z,active
```

Keys are OctoMap's three unsigned 16-bit indices, not signed voxel coordinates.
World positions are voxel centres. `active=0` marks visible passive precoverage;
`active=1` marks a member of that view's remaining coverage set. The active
demand is the union of the latter sets, rather than their summed sizes.
Key numbering depends on resolution and the shared world origin.

## C++ integration

Add this directory using `add_subdirectory(cuda_visibility)` and link your target
to `cuda_visibility`. The public entry point is
`include/cuda_visibility/visibility.hpp`.

```cpp
#include <cuda_visibility/visibility.hpp>

namespace cv = cuda_visibility;
// Supply geometry_points, passive_points, candidate_views and camera intrinsics.
auto map = cv::build_covering_map(geometry_points, 0.04);
auto precovered = cv::map_points_to_occupied_keys(*map.tree, passive_points);
cv::VisibilityOptions options;
options.backend = "cuda";  // "cpu" or "verify" also supported
options.ray_max_distance_m = 2.0;
auto visible = cv::compute_all_visibilities(map, candidate_views, camera, options);
cv::KeySet demand;
auto active_sets = cv::make_active_hit_sets(visible, precovered, 0, &demand);
```

`active_sets[i]` is the coverage set for `candidate_views[i]`;
`demand` is their union. These containers can be passed to a set-cover formulation
without a dependency on a particular optimization backend.
The `confidence_count=0` argument retains every observed voxel; higher values
require observation by strictly more than that many candidate views.
The map resolution is authoritative; set it when calling `build_covering_map`.

## Visibility semantics and compatibility

The CPU and CUDA paths share camera projection and ray preparation. For every
occupied covering voxel and candidate view:

1. Transform the voxel centre into the continuous camera frame; reject points
   behind the camera or outside the image.
2. Project with the supplied intrinsics/distortion and truncate the resulting
   pixel coordinates to integers.
3. Deproject that pixel at camera-frame Z depth `ray_max_distance_m`, then
   transform the endpoint using the continuous camera pose.
4. Snap the camera origin to its OctoMap voxel centre. Cast toward the endpoint
   from that snapped origin, ignoring unknown space.
5. Collect and deduplicate the first occupied hit, excluding the camera-origin
   voxel. The hit need not equal the voxel that originally induced the ray.

This is covering-geometry-driven ray casting, not one ray per image pixel.
The projection/endpoint uses the continuous pose while the ray starts at the
snapped origin; replacing the direction by a vector directly toward the target
voxel changes the operator. Image bounds intentionally preserve the original
inclusive upper-bound check before integer truncation.
The scalar maximum range is measured along the ray by OctoMap, independently
of the Z depth used to construct the endpoint.

The CUDA backend stores occupied full-depth keys in a GPU hash table and traverses
the grid with DDA. Use maps returned by `build_covering_map`; arbitrary pruned or
mixed-depth octrees are not supported by the GPU representation.
`verify` checks exact hit-key sets and per-view counters, and returns an error
on a mismatch. Such checks validate the supplied inputs and tested scenes, not
a mathematical guarantee over every possible floating-point configuration.

## Timing interpretation

The CPU reference preserves the original per-covering-voxel `std::thread`
creation in batches (`--threads`, default 1000), including thread-launch overhead.
It is not an optimized persistent-thread-pool CPU baseline.
Both backends visit views sequentially. CUDA accelerates the ray traversal;
projection, preparation and hit-set collection remain on the CPU.

`benchmark` warms the CUDA path once, then reports arithmetic mean wall times
over `--repeats` paired runs. Timings include preparation, traversal, hit-set
collection and, for CUDA, hash construction and transfers. They exclude map
construction, file I/O, precoverage subtraction and parity comparison.
The small synthetic example checks usability and correctness; its speedup is
not a reproduction of the paper's three-case timing results.

See [VALIDATION.md](VALIDATION.md) for the completed local build and run checks.
