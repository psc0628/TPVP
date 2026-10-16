#include "cuda_visibility/cuda_raycaster.hpp"
#include "cuda_visibility/visibility.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace cv = cuda_visibility;
namespace fs = std::filesystem;

namespace {

struct Arguments {
  std::string backend{"verify"};
  fs::path geometry, views, camera, precovered, output;
  double resolution{0.04}, range{3.0};
  int threads{1000}, repeats{1};
};

double number(const std::string &text) {
  std::size_t end = 0;
  const double value = std::stod(text, &end);
  if (end != text.size() || !std::isfinite(value)) {
    throw std::invalid_argument("Invalid number: " + text);
  }
  return value;
}

int positive_integer(const std::string &text) {
  const double value = number(text);
  if (value < 1 || value > 100000 || std::floor(value) != value) {
    throw std::invalid_argument("Expected integer in [1, 100000]: " + text);
  }
  return static_cast<int>(value);
}

void usage() {
  std::cout
      << "visibility_example [--backend cpu|cuda|verify|benchmark]\n"
      << "  [--geometry cloud.xyz --views views.csv --camera camera.txt]\n"
      << "  [--precovered passive.xyz] [--output hits.csv]\n"
      << "  [--resolution 0.04] [--range 3.0] [--threads 1000] [--repeats 1]\n"
      << "Without input files, generates two partially occluding surfaces and six views.\n"
      << "verify fails on any CPU/CUDA per-view hit/counter mismatch.\n"
      << "benchmark warms CUDA, times both backends, and checks the same parity.\n";
}

Arguments arguments(int argc, char **argv) {
  Arguments args;
  for (int i = 1; i < argc; ++i) {
    const std::string flag(argv[i]);
    if (i + 1 == argc) throw std::invalid_argument("Missing value for " + flag);
    const std::string value(argv[++i]);
    if (flag == "--backend") args.backend = value;
    else if (flag == "--geometry") args.geometry = value;
    else if (flag == "--views") args.views = value;
    else if (flag == "--camera") args.camera = value;
    else if (flag == "--precovered") args.precovered = value;
    else if (flag == "--output") args.output = value;
    else if (flag == "--resolution") args.resolution = number(value);
    else if (flag == "--range") args.range = number(value);
    else if (flag == "--threads") args.threads = positive_integer(value);
    else if (flag == "--repeats") args.repeats = positive_integer(value);
    else throw std::invalid_argument("Unknown option: " + flag);
  }
  if (args.backend != "cpu" && args.backend != "cuda" &&
      args.backend != "verify" && args.backend != "benchmark") {
    throw std::invalid_argument("Unknown backend: " + args.backend);
  }
  if (!(args.resolution > 0.0) || !(args.range > 0.0)) {
    throw std::invalid_argument("Resolution and range must be positive");
  }
  const int inputs = !args.geometry.empty() + !args.views.empty() + !args.camera.empty();
  if (inputs != 0 && inputs != 3) {
    throw std::invalid_argument("Supply geometry, views, and camera together");
  }
  return args;
}

std::vector<std::string> lines(const fs::path &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot open " + path.string());
  std::vector<std::string> result;
  std::string line;
  while (std::getline(input, line)) {
    line = line.substr(0, line.find('#'));
    const auto begin = line.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) continue;
    const auto end = line.find_last_not_of(" \t\r\n");
    result.push_back(line.substr(begin, end - begin + 1));
  }
  if (!input.eof()) throw std::runtime_error("Read failed: " + path.string());
  return result;
}

std::vector<double> values(const std::string &line) {
  std::istringstream input(line);
  std::vector<double> result;
  std::string token;
  while (input >> token) result.push_back(number(token));
  return result;
}

std::vector<Eigen::Vector3d> load_xyz(const fs::path &path) {
  std::vector<Eigen::Vector3d> points;
  for (const auto &line : lines(path)) {
    const auto row = values(line);
    if (row.size() != 3) throw std::runtime_error("XYZ requires exactly x y z: " + path.string());
    points.emplace_back(row[0], row[1], row[2]);
  }
  if (points.empty()) throw std::runtime_error("Empty point cloud: " + path.string());
  return points;
}

cv::CameraModel load_camera(const fs::path &path) {
  const auto rows = lines(path);
  if (rows.size() != 1) throw std::runtime_error("Camera file requires one data line");
  const auto row = values(rows[0]);
  if (row.size() != 6 && row.size() != 11) {
    throw std::runtime_error("Camera requires width height fx fy cx cy [k1 k2 p1 p2 k3]");
  }
  if (row[0] < 1 || row[0] > 100000 || row[1] < 1 || row[1] > 100000 ||
      std::floor(row[0]) != row[0] || std::floor(row[1]) != row[1]) {
    throw std::runtime_error("Invalid camera image dimensions");
  }
  cv::CameraModel camera;
  camera.width = static_cast<int>(row[0]);
  camera.height = static_cast<int>(row[1]);
  camera.fx = row[2]; camera.fy = row[3]; camera.cx = row[4]; camera.cy = row[5];
  if (row.size() == 11) {
    camera.k1 = row[6]; camera.k2 = row[7]; camera.p1 = row[8];
    camera.p2 = row[9]; camera.k3 = row[10];
  }
  return camera;
}

std::vector<cv::CandidateView> load_views(const fs::path &path) {
  const auto rows = lines(path);
  if (rows.empty() || rows[0] != "view_id,x,y,z,qx,qy,qz,qw") {
    throw std::runtime_error("Expected CSV header: view_id,x,y,z,qx,qy,qz,qw");
  }
  std::vector<cv::CandidateView> views;
  std::unordered_set<std::int64_t> ids;
  for (std::size_t i = 1; i < rows.size(); ++i) {
    const auto comma = rows[i].find(',');
    if (comma == std::string::npos || std::count(rows[i].begin(), rows[i].end(), ',') != 7) {
      throw std::runtime_error("Expected eight CSV columns");
    }
    cv::CandidateView view;
    std::size_t end = 0;
    view.view_id = std::stoll(rows[i].substr(0, comma), &end);
    if (end != comma || !ids.insert(view.view_id).second) {
      throw std::runtime_error("Invalid or duplicate view ID");
    }
    std::string pose = rows[i].substr(comma + 1);
    std::replace(pose.begin(), pose.end(), ',', ' ');
    const auto row = values(pose);
    if (row.size() != 7) throw std::runtime_error("Expected seven pose values");
    view.camera_position = Eigen::Vector3d(row[0], row[1], row[2]);
    view.camera_orientation = Eigen::Quaterniond(row[6], row[3], row[4], row[5]);
    views.push_back(view);
  }
  if (views.empty()) throw std::runtime_error("No views in " + path.string());
  return views;
}

void make_demo(std::vector<Eigen::Vector3d> &points,
               std::vector<cv::CandidateView> &views, cv::CameraModel &camera) {
  camera.width = 640; camera.height = 480;
  camera.fx = 500; camera.fy = 500; camera.cx = 320; camera.cy = 240;
  // Dense rear wall, smaller foreground patch. Off-centre samples avoid
  // ambiguous world-coordinate grid boundaries in this illustrative scene.
  for (int x = -22; x <= 22; ++x) {
    for (int y = -14; y <= 14; ++y) {
      points.emplace_back(x * 0.04 + 0.013, y * 0.04 + 0.017, 1.813);
    }
  }
  for (int x = -9; x <= 9; ++x) {
    for (int y = -7; y <= 7; ++y) {
      points.emplace_back(x * 0.04 + 0.013, y * 0.04 + 0.017, 1.013);
    }
  }
  for (int i = 0; i < 6; ++i) {
    cv::CandidateView view;
    view.view_id = i;
    view.camera_position = Eigen::Vector3d(-0.45 + 0.18 * i, 0.037, 0.019);
    const Eigen::Vector3d forward =
        (Eigen::Vector3d(0.0, 0.0, 1.4) - view.camera_position).normalized();
    const Eigen::Vector3d right = Eigen::Vector3d::UnitY().cross(forward).normalized();
    Eigen::Matrix3d rotation;
    rotation.col(0) = right;
    rotation.col(1) = forward.cross(right);
    rotation.col(2) = forward;
    view.camera_orientation = Eigen::Quaterniond(rotation);
    views.push_back(view);
  }
}

void check_parity(const std::vector<cv::ViewVisibility> &cpu,
                  const std::vector<cv::ViewVisibility> &gpu) {
  if (cpu.size() != gpu.size()) throw std::runtime_error("View-count mismatch");
  for (std::size_t i = 0; i < cpu.size(); ++i) {
    if (cpu[i].view_id != gpu[i].view_id || cpu[i].hits != gpu[i].hits ||
        cpu[i].covering_voxels_considered != gpu[i].covering_voxels_considered ||
        cpu[i].in_front != gpu[i].in_front || cpu[i].inside_image != gpu[i].inside_image ||
        cpu[i].successful_rays != gpu[i].successful_rays) {
      throw std::runtime_error("CPU/CUDA mismatch for view " + std::to_string(cpu[i].view_id));
    }
  }
}

void write_hits(const fs::path &path, const cv::CoveringMap &map,
                 const std::vector<cv::ViewVisibility> &results,
                 const std::vector<cv::KeySet> &active) {
  if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
  std::ofstream output(path);
  if (!output) throw std::runtime_error("Cannot write " + path.string());
  output << "view_id,key_x,key_y,key_z,world_x,world_y,world_z,active\n"
         << std::setprecision(10);
  for (std::size_t i = 0; i < results.size(); ++i) {
    std::vector<octomap::OcTreeKey> keys(results[i].hits.begin(), results[i].hits.end());
    std::sort(keys.begin(), keys.end(), [](const auto &a, const auto &b) {
      if (a[0] != b[0]) return a[0] < b[0];
      if (a[1] != b[1]) return a[1] < b[1];
      return a[2] < b[2];
    });
    for (const auto &key : keys) {
      const auto point = map.tree->keyToCoord(key);
      output << results[i].view_id << ',' << key[0] << ',' << key[1] << ',' << key[2]
             << ',' << point.x() << ',' << point.y() << ',' << point.z()
             << ',' << active[i].count(key) << '\n';
    }
  }
  output.close();
  if (!output) throw std::runtime_error("Failed writing " + path.string());
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") { usage(); return 0; }
    const auto args = arguments(argc, argv);
    std::vector<Eigen::Vector3d> points;
    std::vector<cv::CandidateView> views;
    cv::CameraModel camera;
    if (args.geometry.empty()) make_demo(points, views, camera);
    else {
      points = load_xyz(args.geometry);
      views = load_views(args.views);
      camera = load_camera(args.camera);
    }
    const auto map = cv::build_covering_map(points, args.resolution);
    const auto passive = args.precovered.empty() ? cv::KeySet{} :
        cv::map_points_to_occupied_keys(*map.tree, load_xyz(args.precovered));
    cv::VisibilityOptions options;
    options.resolution_m = args.resolution;
    options.ray_max_distance_m = args.range;
    options.max_threads_per_batch = args.threads;
    options.backend = args.backend;
    std::cout << "Occupied voxels: " << map.occupied_keys.size()
              << "; views: " << views.size() << "; passive voxels: " << passive.size() << '\n';
    if (args.backend != "cpu") std::cout << "CUDA device: " << cv::cuda_runtime_device_name() << '\n';
    std::vector<cv::ViewVisibility> results;
    if (args.backend == "benchmark") {
      // Warm the context and kernel before timing. Map construction and file
      // I/O are excluded; each timed GPU call still builds/uploads its hash.
      options.backend = "cuda";
      cv::compute_all_visibilities(map, views, camera, options);
      double cpu_s = 0, cuda_s = 0;
      for (int repeat = 0; repeat < args.repeats; ++repeat) {
        options.backend = "cpu";
        auto start = std::chrono::steady_clock::now();
        const auto cpu = cv::compute_all_visibilities(map, views, camera, options);
        cpu_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        options.backend = "cuda";
        start = std::chrono::steady_clock::now();
        results = cv::compute_all_visibilities(map, views, camera, options);
        cuda_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        check_parity(cpu, results);
      }
      std::cout << std::fixed << std::setprecision(6)
                << "Mean CPU visibility: " << cpu_s / args.repeats << " s\n"
                << "Mean CUDA visibility: " << cuda_s / args.repeats << " s\n"
                << "Speedup: " << cpu_s / cuda_s << "x\n"
                << "Exact per-view CPU/CUDA parity: PASS\n";
    } else {
      const auto start = std::chrono::steady_clock::now();
      results = cv::compute_all_visibilities(map, views, camera, options);
      std::cout << std::fixed << std::setprecision(6) << args.backend
                << " elapsed: "
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                << " s\n";
      if (args.backend == "verify") std::cout << "Exact per-view CPU/CUDA parity: PASS\n";
    }
    cv::KeySet demand;
    const auto active = cv::make_active_hit_sets(results, passive, 0, &demand);
    for (std::size_t i = 0; i < results.size(); ++i) {
      std::cout << "view " << results[i].view_id << ": visible=" << results[i].hits.size()
                << ", active=" << active[i].size() << '\n';
    }
    std::cout << "Visible union: " << cv::union_hits(results, 0).size()
              << "; active demand union: " << demand.size() << '\n';
    if (!args.output.empty()) {
      write_hits(args.output, map, results, active);
      std::cout << "Coverage CSV: " << args.output.string() << '\n';
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
