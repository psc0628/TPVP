# Local validation

Validated on 2026-10-16 using Ubuntu 24.04 under WSL2, GCC 13.3, CUDA 12.8,
OctoMap 1.10.0, and an NVIDIA GeForce RTX 4060 Laptop GPU.

## Checks completed

- Release build with CUDA enabled: successful.
- CTest: 5/5 checks passed, covering CPU operation, file inputs, foreground
  occlusion, maximum range, passive-demand subtraction, shared-voxel
  deduplication, rotated poses, distortion, and exact CPU/CUDA parity.
- Direct CPU/CUDA ray tests passed for axis-aligned rays, negative coordinates,
  DDA ties, range boundaries, and an occupied origin.
- Source copied to an independent temporary directory: CPU-only build and
  all 3 CPU checks passed without parent-project files or CUDA compilation.
- CPU-only CUDA requests and non-finite CLI numbers correctly return errors.
- File-based GPU verification and CSV export passed: 13 per-view rows,
  8 unique visible voxels, and 7 unique active-demand voxels after removing
  one passive voxel.

## Synthetic demo

The generated scene contains 1,590 occupied voxels and six candidate camera
poses. Exact per-view CPU/CUDA hit sets and counters matched.

| View ID | Visible occupied voxels |
| --- | ---: |
| 0 | 760 |
| 1 | 658 |
| 2 | 630 |
| 3 | 629 |
| 4 | 632 |
| 5 | 737 |
| Union | 1,158 |

Three paired runs after CUDA warm-up gave a mean CPU time of 0.323175 s and
a mean CUDA time of 0.003930 s (approximately 82x). The CPU reference retains
the original per-voxel thread creation. These illustrative timings depend on
the machine and scene and do not reproduce or replace the paper's benchmarks.
Timing boundaries are described in README.md.
