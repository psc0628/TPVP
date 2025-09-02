# run with wsl
# export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# python main.py config.yaml

from model.geometry import *
import os
import torch
import argparse
import cv2
from model.registration import Registration
import  yaml
from easydict import EasyDict as edict
import open3d as o3d
from scipy.spatial import cKDTree
import time

def _make_pcd(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(xyz.astype(np.float64, copy=False))
    return p

def _estimate_normals(pcd: o3d.geometry.PointCloud, vsize: float, max_nn: int = 40):
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=vsize*2.5, max_nn=max_nn))
    return pcd

def _compute_fpfh(pcd: o3d.geometry.PointCloud, vsize: float, max_nn: int = 100):
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=vsize*5.0, max_nn=max_nn)
    )

def _ransa_criteria_safe():
    # 兼容不同 Open3D 版本的 RANSAC 收敛参数签名
    try:
        # 某些版本是 (max_iteration, max_validation)
        return o3d.pipelines.registration.RANSACConvergenceCriteria(1_000_000, 1_000)
    except TypeError:
        # 新一些的版本是 (max_iteration, confidence[0~1])
        return o3d.pipelines.registration.RANSACConvergenceCriteria(1_000_000, 0.999)

def _dedup_by_residual(corr_idx, src_down, tgt_down, T, mode='both', dist_th=np.inf):
    """
    在 T 下将 src_down 变换到 target 坐标系，按残差从小到大排序，
    再在 src/tgt 维度做唯一化（优先保留残差更小的对应）。
    """
    if len(corr_idx) == 0:
        return corr_idx

    src_sel = np.asarray(src_down.points)[corr_idx[:, 0]]
    tgt_sel = np.asarray(tgt_down.points)[corr_idx[:, 1]]

    src_sel_h = np.c_[src_sel, np.ones(len(src_sel))]
    src_tf = (src_sel_h @ T.T)[:, :3]
    resid = np.linalg.norm(src_tf - tgt_sel, axis=1)

    keep = resid < dist_th
    corr_idx = corr_idx[keep]
    resid = resid[keep]

    if len(corr_idx) == 0:
        return corr_idx

    order = np.argsort(resid)  # 小残差优先
    corr_idx = corr_idx[order]

    # 唯一化：保留第一次出现（即残差最小的那个）
    if mode in ('src', 'both'):
        _, uniq = np.unique(corr_idx[:, 0], return_index=True)
        corr_idx = corr_idx[np.sort(uniq)]
    if mode in ('tgt', 'both'):
        _, uniq = np.unique(corr_idx[:, 1], return_index=True)
        corr_idx = corr_idx[np.sort(uniq)]
    return corr_idx

def build_landmarks_via_ransac(
    src_xyz_np: np.ndarray,
    tgt_xyz_np: np.ndarray,
    voxel_size: float = 0.02,
    max_pairs: int = 4000,
    max_nn_normal: int = 40,
    max_nn_fpfh: int = 100,
    ransac_n: int = 4,
    mutual_filter: bool = True,
    dedup_mode: str = 'both',
    # 下面三个都与 voxel_size 挂钩，通常不需要手动改动
    max_corr_dist_scale: float = 2.0,     # max_correspondence_distance = scale * voxel_size
    dedup_dist_scale: float = 1.5,        # dedup 残差阈值 = scale * max_correspondence_distance
    add_normal_checker: bool = False,     # 是否加法线一致性检查
    visualize: bool = True,
):
    """
    返回:
        lmk_src_np: (K, 3) 源点原始分辨率的地标
        lmk_tgt_np: (K, 3) 目标点原始分辨率的地标
        T: (4, 4) 估计的刚体变换（把 src→tgt）
    """
    assert src_xyz_np.ndim == 2 and src_xyz_np.shape[1] == 3
    assert tgt_xyz_np.ndim == 2 and tgt_xyz_np.shape[1] == 3

    src_pcd = _make_pcd(src_xyz_np)
    tgt_pcd = _make_pcd(tgt_xyz_np)

    # 下采样 + 法线
    src_down = src_pcd.voxel_down_sample(voxel_size)
    tgt_down = tgt_pcd.voxel_down_sample(voxel_size)
    _estimate_normals(src_down, voxel_size, max_nn=max_nn_normal)
    _estimate_normals(tgt_down, voxel_size, max_nn=max_nn_normal)

    # FPFH
    src_fpfh = _compute_fpfh(src_down, voxel_size, max_nn=max_nn_fpfh)
    tgt_fpfh = _compute_fpfh(tgt_down, voxel_size, max_nn=max_nn_fpfh)

    # RANSAC 一次粗配准
    max_corr_dist = max_corr_dist_scale * voxel_size
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_corr_dist),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
    ]
    if add_normal_checker:
        # 30° 内视为一致：cos(30°)
        checkers.append(
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(np.cos(np.deg2rad(30)))
        )

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=mutual_filter,
        max_correspondence_distance=max_corr_dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        checkers=checkers,
        criteria=_ransa_criteria_safe(),
    )

    T = result.transformation
    corr = np.asarray(result.correspondence_set)  # (K, 2)

    if corr.size == 0:
        raise RuntimeError(
            "[FPFH-RANSAC] 没有找到对应点：请尝试增大 voxel_size 或 max_corr_dist_scale，"
            "或关闭 mutual_filter 再试。"
        )

    # 基于“变换后残差”的去重（关键）
    dedup_dist_th = dedup_dist_scale * max_corr_dist
    corr = _dedup_by_residual(
        corr_idx=corr,
        src_down=src_down,
        tgt_down=tgt_down,
        T=T,
        mode=dedup_mode,
        dist_th=dedup_dist_th
    )
    if corr.size == 0:
        raise RuntimeError("[FPFH-RANSAC] 去重后没有可用匹配：请放宽 dedup_dist_scale 或改用 mode='src'/'tgt'。")

    # 限制数量
    if len(corr) > max_pairs:
        corr = corr[np.random.choice(len(corr), max_pairs, replace=False)]

    # 映射回“原始分辨率”的对应（对 FULL 建树，用 DOWN 点去 query）
    src_full_kd = cKDTree(src_xyz_np)
    tgt_full_kd = cKDTree(tgt_xyz_np)

    src_down_sel = np.asarray(src_down.points)[corr[:, 0]]
    tgt_down_sel = np.asarray(tgt_down.points)[corr[:, 1]]

    src_full_idx = src_full_kd.query(src_down_sel, k=1)[1]
    tgt_full_idx = tgt_full_kd.query(tgt_down_sel, k=1)[1]

    lmk_src_np = src_xyz_np[src_full_idx]
    lmk_tgt_np = tgt_xyz_np[tgt_full_idx]

    # 可视化（把源点云先变换到目标坐标系，连线长度会很短）
    if visualize:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="FPFH-RANSAC Matches")
        psrc_tf = o3d.geometry.PointCloud(src_pcd)  # 拷贝
        psrc_tf.transform(T)
        psrc_tf.paint_uniform_color([1, 0, 0])     # 变换后的源：红
        ptgt = _make_pcd(tgt_xyz_np)
        ptgt.paint_uniform_color([0, 0, 1])        # 目标：蓝
        vis.add_geometry(psrc_tf)
        vis.add_geometry(ptgt)

        # 画配准后的连线（短、直观）
        src_tf_sel = (np.c_[src_down_sel, np.ones(len(corr))] @ T.T)[:, :3]
        match_xyz = np.vstack([src_tf_sel, tgt_down_sel])
        lines = np.c_[np.arange(len(corr)), np.arange(len(corr)) + len(corr)]
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(match_xyz),
            lines=o3d.utility.Vector2iVector(lines.astype(np.int32))
        )
        ls.colors = o3d.utility.Vector3dVector(np.tile([1, 1, 0], (len(lines), 1)))  # 黄线
        vis.add_geometry(ls)
        opt = vis.get_render_option()
        opt.point_size = 5.5
        opt.line_width = 3.5
        vis.run()
        vis.destroy_window()

    print(f"[FPFH-RANSAC] inliers(after dedup) = {len(lmk_src_np)}")
    return lmk_src_np, lmk_tgt_np, T


def join(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join([str(i) for i in seq])
yaml.add_constructor('!join', join)


# register from pointclouds
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    cfg = edict(yaml.safe_load(open(parser.parse_args().config)))

    cfg.device = torch.device("cuda:0") if cfg.gpu_mode else torch.device("cpu")

    if not hasattr(cfg, "intrinsics"):
        K = None
    else:
        K = np.loadtxt(cfg.intrinsics)
        print (K)

    src_pcd_np = np.asarray(o3d.io.read_point_cloud(cfg.src_pcd).points)
    tgt_pcd_np = np.asarray(o3d.io.read_point_cloud(cfg.tgt_pcd).points)
    model      = Registration(src_pcd_np, K=K, config=cfg)

    tgt_pcd_path = cfg.tgt_pcd

    if cfg.use_fpfh:
        fpfp_start_time = time.time()
        # 使用 FPFH 特征和 RANSAC 估计地标
        lmk_src_np, lmk_tgt_np, T_src_to_tgt = build_landmarks_via_ransac(
                                                src_xyz_np=src_pcd_np,          # (N,3) 源点云
                                                tgt_xyz_np=tgt_pcd_np,          # (M,3) 目标点云
                                                voxel_size=cfg.voxel_size,      # 体素下采样大小（核心尺度参数）
                                                max_pairs=1000,                 # 最终最多保留的对应点数
                                                max_nn_normal=10,               # 法线估计邻居数
                                                max_nn_fpfh=10,                 # FPFH 特征邻居数
                                                ransac_n=4,                     # RANSAC每次采样点数（3或4常用）越大越慢
                                                mutual_filter=True,             # 是否使用互为最近邻的特征过滤
                                                dedup_mode='both',              # 匹配后去重的策略：'src' / 'tgt' / 'both'
                                                max_corr_dist_scale=1.5,        # RANSAC几何容差 = 2.0 * voxel_size 越大去掉越少
                                                dedup_dist_scale=1.2,           # 去重残差阈值 = 1.5 * 上面的容差 越大去掉越少
                                                add_normal_checker=False,       # 有稳定法线时可开，提高鲁棒性
                                                visualize=cfg.visualize_fpfh    # 服务器/无显示环境设为 False
                                            )
        fpfp_cost_time = time.time() - fpfp_start_time
        print(f"FPFH-RANSAC cost time: {fpfp_cost_time:.3f}s")
        landmarks = (torch.from_numpy(lmk_src_np).float().to(cfg.device),
                     torch.from_numpy(lmk_tgt_np).float().to(cfg.device),)
    else:
        fpfp_cost_time = 0.0
        landmarks = None

    nricp_start_time = time.time()
    # NRICP registration
    model.register_a_pointcloud(tgt_pcd_path, landmarks=landmarks)
    nricp_cost_time = time.time() - nricp_start_time
    print(f"NRICP cost time: {nricp_cost_time:.3f}s")

    if cfg.save_nricp_result:
        with open(cfg.cost_time_path, 'w+') as f:
            f.write(f"fpfh_cost_time: {fpfp_cost_time:.3f}\n")
            f.write(f"nricp_cost_time: {nricp_cost_time:.3f}\n")

# # register from depth images
# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument('config', type=str, help= 'Path to the config file.')
#     args = parser.parse_args()
#     with open(args.config,'r') as f:
#         config = yaml.load(f, Loader=yaml.Loader)
#     config = edict(config)

#     if config.gpu_mode:
#         config.device = torch.device("cuda:0")
#     else:
#         config.device = torch.device('cpu')

#     """demo data"""
#     intrinsics = np.loadtxt(config.intrinsics)
#     print (intrinsics)

#     """load lepard predicted matches as landmarks"""
#     data = np.load(config.correspondence)
#     ldmk_src = data['src_pcd'][0][data['match'][:,1] ]
#     ldmk_tgt = data['tgt_pcd'][0][data['match'][:,2] ]
#     uv_src = xyz_2_uv(ldmk_src, intrinsics)
#     uv_tgt = xyz_2_uv(ldmk_tgt, intrinsics)
#     landmarks = ( torch.from_numpy(uv_src).to(config.device),
#                   torch.from_numpy(uv_tgt).to(config.device))
#     """init model with source frame"""
#     model = Registration(config.src_depth, K=intrinsics, config=config)

#     model.register_a_depth_frame( config.tgt_depth,  landmarks=landmarks)


