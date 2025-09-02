import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from skimage import io
from PIL import Image
from timeit import default_timer as timer
import datetime
import argparse
from .geometry import *

import yaml
import matplotlib.pyplot as plt
import torch
from lietorch import SO3, SE3, LieGroupParameter
import torch.optim as optim
from .loss import *
from .point_render import PCDRender

@torch.no_grad()
def laplacian_pc_loss_sampled(
    x_def,                # (N,3) warped_pcd (torch.Tensor)
    samples=1000,
    K=8,
    sigma=None,          # 若为 None，则用采样内的 KNN 距离中位数自适应
    mode='membrane',     # 'membrane' 一阶；'thinplate' 二阶(可选)
):
    """
    仅用采样子集计算点云拉普拉斯正则，简单&快。
    注意：KNN 在采样子集内计算（无外部依赖），每次随机邻域会略有抖动。
    """
    device = x_def.device
    N = x_def.shape[0]
    m = min(samples, N)
    idx = torch.randperm(N, device=device)[:m]
    xs  = x_def[idx]                               # (m,3)

    # 采样内 KNN（排除自环）
    d = torch.cdist(xs, xs)                        # (m,m)
    knn_d, knn_idx = torch.topk(d, k=K+1, largest=False)   # 包含自身
    knn_d, knn_idx = knn_d[:, 1:], knn_idx[:, 1:]          # 去掉自身

    # 高斯权重（行归一化）
    if sigma is None:
        # 用 KNN 距离中位数自适应一个尺度
        sigma = knn_d.median()
    W = torch.exp(-(knn_d**2) / (2 * sigma**2 + 1e-12))
    W = W / (W.sum(dim=1, keepdim=True) + 1e-8)            # (m,K)

    xi = xs[:, None, :]                      # (m,1,3)
    xj = xs[knn_idx]                         # (m,K,3)

    if mode == 'membrane':                   # 一阶：∑ w_ij ||xi - xj||^2
        loss = (W * ((xi - xj)**2).sum(-1)).mean()
    elif mode == 'thinplate':                # 二阶：||L x||^2
        Lx = (W[..., None] * (xj - xi)).sum(dim=1)         # (m,3)
        loss = (Lx**2).sum(dim=-1).mean()
    else:
        raise ValueError("mode must be 'membrane' or 'thinplate'")
    return loss

def tensor_to_o3d_pcd(tensor):
    # tensor: (N, 3), on cpu
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tensor.cpu().numpy())
    return pcd

def build_graph_from_pointcloud(points: np.ndarray,
                                node_coverage: float = 0.09,
                                num_neighbors: int = 8,
                                K_anchor: int = 4,
                                debug: bool = False):
    import open3d as o3d
    from sklearn.neighbors import NearestNeighbors
    
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd_down = pcd.voxel_down_sample(voxel_size=node_coverage)
    nodes_np = np.asarray(pcd_down.points)          # (M,3)
    M = nodes_np.shape[0]
    if debug:
        print(f"[graph] sample {M} nodes (voxel={node_coverage})")

    knn_node = NearestNeighbors(n_neighbors=num_neighbors + 1).fit(nodes_np)
    nbr_idx  = knn_node.kneighbors(return_distance=False)[:, 1:]   # (M,k)
    nbr_dist = np.linalg.norm(
        nodes_np[:, None, :] - nodes_np[nbr_idx], axis=-1) + 1e-8
    w_edge   = 1.0 / nbr_dist
    w_edge  /= w_edge.sum(axis=1, keepdims=True)

    knn_pt = NearestNeighbors(n_neighbors=K_anchor).fit(nodes_np)
    dist_anch, idx_anch = knn_pt.kneighbors(points)                # (N,K)
    sigma2 = (node_coverage * 1.5) ** 2
    w_anch = np.exp(-dist_anch ** 2 / sigma2)
    w_anch /= w_anch.sum(axis=1, keepdims=True)

    graph = dict(
        graph_nodes         = torch.from_numpy(nodes_np).float(),  # (M,3)
        graph_edges         = torch.from_numpy(nbr_idx).long(),    # (M,k)
        graph_edges_weights = torch.from_numpy(w_edge).float(),    # (M,k)
        graph_clusters      = torch.full((M, 1), -1, dtype=torch.int32),
        point_anchors       = torch.from_numpy(idx_anch).long(),   # (N,K)
        point_weights       = torch.from_numpy(w_anch).float(),    # (N,K)
        pixel_anchors       = torch.zeros((0)),
        pixel_weights       = torch.zeros((0)),
        point_image         = torch.from_numpy(points).unsqueeze(0)
    )
    return graph

class Registration():

    def __init__(self, source, K, config):
        """
        source :  str（depth）| np.ndarray(N,3)（pointcloud）
        """
        self.device, self.config, self.intrinsics = config.device, config, K
        self.deformation_model = config.deformation_model

        if isinstance(source, str): # depth
            """initialize deformation graph"""
            depth_image = io.imread(source)
            image_size = (depth_image.shape[0], depth_image.shape[1])
            data = get_deformation_graph_from_depthmap( depth_image, K)
            self.graph_nodes = data['graph_nodes'].to(self.device)
            self.graph_edges = data['graph_edges'].to(self.device)
            self.graph_edges_weights = data['graph_edges_weights'].to(self.device)
            self.graph_clusters = data['graph_clusters'] #.to(self.device)
            """initialize point clouds"""
            valid_pixels = torch.sum(data['pixel_anchors'], dim=-1) > -4
            self.source_pcd = data["point_image"][valid_pixels].to(self.device)
            self.point_anchors = data["pixel_anchors"][valid_pixels].long().to(self.device)
            self.anchor_weight = data["pixel_weights"][valid_pixels].to(self.device)
            self.anchor_loc = data["graph_nodes"].to(self.device)[self.point_anchors.to(self.device)]
            self.frame_point_len = [ len(self.source_pcd)]
            """pixel to pcd map"""
            self.pix_2_pcd_map = [ self.map_pixel_to_pcd(valid_pixels).to(config.device) ]
            """define differentiable pcd renderer"""
            self.renderer = PCDRender(K, img_size=image_size)

        else: # pointcloud
            src_np = source                                 # (N,3)
            self.source_pcd = torch.from_numpy(src_np).float().to(self.device)

            graph = build_graph_from_pointcloud(
                src_np,
                node_coverage=config.node_coverage,
                num_neighbors=8,
                debug=True)

            for k in ["graph_nodes", "graph_edges",
                      "graph_edges_weights", "graph_clusters"]:
                setattr(self, k, graph[k].to(self.device))

            self.point_anchors = graph["point_anchors"].long().to(self.device)   # (N,K)
            self.anchor_weight = graph["point_weights"].to(self.device)   # (N,K)
            self.anchor_loc    = self.graph_nodes[self.point_anchors]     # (N,K,3)
            self.frame_point_len = [len(self.source_pcd)]

            self.pix_2_pcd_map  = []    
            self.renderer       = None
            # image_size          = (self.config.image_height, self.config.image_width)
            # self.renderer       = PCDRender(K, img_size=image_size)


    def register_a_depth_frame(self, tgt_depth_path, landmarks=None):
        """
        :param tgt_depth_path:
        :return:
        """

        """load target frame"""
        tgt_depth = io.imread(tgt_depth_path) / 1000.
        depth_mask = torch.from_numpy(tgt_depth > 0)
        tgt_pcd = depth_2_pc(tgt_depth, self.intrinsics).transpose(1, 2, 0)
        self.tgt_pcd = torch.from_numpy(tgt_pcd[tgt_depth > 0]).float().to(self.device)
        pix_2_pcd = self.map_pixel_to_pcd(depth_mask).to(self.device)

        # o3d.io.write_point_cloud("cam1_pcd.pcd", tensor_to_o3d_pcd(self.source_pcd))
        # o3d.io.write_point_cloud("cam2_pcd.pcd", tensor_to_o3d_pcd(self.tgt_pcd))

        if landmarks is not None:
            s_uv, t_uv = landmarks
            s_id = self.pix_2_pcd_map[-1][s_uv[:, 1], s_uv[:, 0]]
            t_id = pix_2_pcd[t_uv[:, 1], t_uv[:, 0]]
            valid_id = (s_id > -1) * (t_id > -1)
            s_ldmk = s_id[valid_id]
            t_ldmk = t_id[valid_id]

            landmarks = (s_ldmk, t_ldmk)

        self.visualize_results(self.tgt_pcd)
        warped_pcd = self.solve(landmarks=landmarks)
        self.visualize_results(self.tgt_pcd, warped_pcd)

    def register_a_pointcloud(self, tgt_pcd_path: str, landmarks=None):

        tgt_np = np.asarray(o3d.io.read_point_cloud(tgt_pcd_path).points)
        self.tgt_pcd = torch.from_numpy(tgt_np).float().to(self.device)

        # o3d.io.write_point_cloud("src_dump.pcd", tensor_to_o3d_pcd(self.source_pcd))
        # o3d.io.write_point_cloud("tgt_dump.pcd", tensor_to_o3d_pcd(self.tgt_pcd))

        if landmarks is not None:
            s_xyz, t_xyz = landmarks  # (M,3)
            s_id = torch.cdist(self.source_pcd, s_xyz).argmin(dim=0)
            t_id = torch.cdist(self.tgt_pcd, t_xyz).argmin(dim=0)
            landmarks = (s_id.to(torch.long), t_id.to(torch.long))

        if self.config.visualize_nricp:
            self.visualize_results(self.tgt_pcd)  # before

        warped = self.solve(landmarks=landmarks)

        if self.config.visualize_nricp: 
            self.visualize_results(self.tgt_pcd, warped)  # after

        if self.config.save_nricp_result:
            warped_np = warped.detach().cpu()
            o3d.io.write_point_cloud(self.config.nricp_result_path, tensor_to_o3d_pcd(warped_np))

    def solve(self, **kwargs):

        if self.deformation_model == "ED":
            # Embeded_deformation, c.f. https://people.inf.ethz.ch/~sumnerb/research/embdef/Sumner2007EDF.pdf
            return self.optimize_ED(**kwargs)



    def optimize_ED(self, landmarks=None):
        '''
        :param landmarks:
        :return:
        '''

        """translations"""
        node_translations = torch.zeros_like(self.graph_nodes)
        self.t = torch.nn.Parameter(node_translations)
        self.t.requires_grad = True

        """rotations"""
        phi = torch.zeros_like(self.graph_nodes)
        node_rotations = SO3.exp(phi)
        self.R = LieGroupParameter(node_rotations)

        """optimizer setup"""
        optimizer = optim.Adam([self.R, self.t], lr=self.config.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

        """render reference pcd"""
        if self.renderer != None:
            sil_tgt, d_tgt, _ = self.render_pcd(self.tgt_pcd)

        # Transform points
        for i in range(self.config.iters):

            anchor_trn = self.t[self.point_anchors]
            anchor_rot = self.R[self.point_anchors]
            warped_pcd = ED_warp(self.source_pcd, self.anchor_loc, anchor_rot, anchor_trn, self.anchor_weight)

            err_arap = arap_cost(self.R, self.t, self.graph_nodes, self.graph_edges, self.graph_edges_weights)
            err_ldmk = landmark_cost(warped_pcd, self.tgt_pcd, landmarks) if landmarks is not None else 0

            if self.renderer != None:
                sil_src, d_src, _ = self.render_pcd(warped_pcd)
                err_silh = silhouette_cost(sil_src, sil_tgt) if self.config.w_silh > 0 else 0
                err_depth = projective_depth_cost(d_src, d_tgt) if self.config.w_depth > 0 else 0
            else:
                err_silh = 0
                err_depth = 0

            cd = chamfer_dist(warped_pcd, self.tgt_pcd) if self.config.w_chamfer > 0 else 0
            err_lap  = laplacian_pc_loss_sampled(warped_pcd) if self.config.w_lap > 0 else 0

            loss = \
                err_arap * self.config.w_arap + \
                err_ldmk * self.config.w_ldmk + \
                err_silh * self.config.w_silh + \
                err_depth * self.config.w_depth + \
                cd * self.config.w_chamfer + \
                err_lap * self.config.w_lap

            print(i, loss)
            if loss.item() < 1e-7:
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        return warped_pcd

    def render_pcd(self, x):
        if self.renderer is None:
            return 0, 0, None

        INF = 1e+6
        px, dx = self.renderer(x)
        px, dx = map(lambda feat: feat.squeeze(), [px, dx])
        dx[dx < 0] = INF
        mask = px[..., 0] > 0
        return px, dx, mask

    def map_pixel_to_pcd(self, valid_pix_mask):
        ''' establish pixel to point cloud mapping, with -1 filling for invalid pixels
        :param valid_pix_mask:
        :return:
        '''
        image_size = valid_pix_mask.shape
        pix_2_pcd_map = torch.cumsum(valid_pix_mask.view(-1), dim=0).view(image_size).long() - 1
        pix_2_pcd_map[~valid_pix_mask] = -1
        return pix_2_pcd_map

    def visualize_results(self, tgt_pcd, warped_pcd=None):
        import open3d as o3d
        import numpy as np

        c_red = np.array([224 / 255, 0 / 255, 125 / 255])
        c_pink = np.array([224 / 255, 75 / 255, 232 / 255])
        c_blue = np.array([0 / 255, 0 / 255, 255 / 255])

        src_np = self.source_pcd.detach().cpu().numpy()
        tgt_np = tgt_pcd.detach().cpu().numpy()

        def make_pcd(points: np.ndarray, color: np.ndarray) -> o3d.geometry.PointCloud:
            p = o3d.geometry.PointCloud()
            p.points = o3d.utility.Vector3dVector(points)
            p.paint_uniform_color(color)
            return p

        geoms = []
        if warped_pcd is None:
            geoms.append(make_pcd(src_np, c_red))
        else:
            warped_np = warped_pcd.detach().cpu().numpy()
            geoms.append(make_pcd(warped_np, c_pink))
        geoms.append(make_pcd(tgt_np, c_blue))

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Open3D ICP Result")
        for g in geoms:
            vis.add_geometry(g)

        render_option = vis.get_render_option()
        if render_option is not None:
            render_option.point_size = 3.0
            render_option.background_color = np.array([0, 0, 0])
        else:
            print(
                '[WARNING] Open3D render_option is None. Possibly running in headless/WSL/Wayland environment. Visualization skipped.')
            vis.destroy_window()
            return

        vis.run()
        vis.destroy_window()
