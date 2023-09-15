import sys
import torch
import numpy as np
import os
import os.path as osp

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from moyo.core.support import StabilityLossCoS
from moyo.utils.constants import CONTACT_THRESH
from moyo.utils.mesh_utils import HDfier
from moyo.utils.pressure_mat_utils import vis_heatmap

class MetricsCollector:
    def __init__(self):
        self.ious = []
        self.cop_errors = []
        self.frame_diffs = []
        self.binary_ths = []
        self.best_cop_w = []
        self.best_cop_k = []
        self.pred_heatmaps = []
        self.pred_cops_relative = []

    def assign(self, *, iou, cop_error, frame_diff, binary_th, best_cop_w, best_cop_k, pred_heatmap, pred_cop_relative):
        self.ious.append(iou)
        self.cop_errors.append(cop_error)
        self.frame_diffs.append(frame_diff)
        self.binary_ths.append(binary_th)
        self.best_cop_w.append(best_cop_w)
        self.best_cop_k.append(best_cop_k)
        self.pred_heatmaps.append(pred_heatmap)
        self.pred_cops_relative.append(pred_cop_relative)


class BiomechanicalEvaluator(StabilityLossCoS):
    def __init__(self,
                 faces,
                 cop_w=10,
                 cop_k=100,
                 contact_thresh=CONTACT_THRESH,
                 model_type='smplx',
                 ):
        super().__init__(faces, cop_w, cop_k, contact_thresh, model_type)

        self.iou_first = MetricsCollector()
        self.cop_error_first = MetricsCollector()
        self.frame_diff_first = MetricsCollector()

        self.cop_k_range = np.linspace(10, 200, 10).astype(np.float64)
        self.cop_w_range = np.linspace(10, 5000, 15).astype(np.float64)

    def generate_aligned_heatmap(self, vertices, mat_size, marker_positions,
                                 ground_plane_height=0.0, heatmap_res=512, cop_w=10, cop_k=100, vp=0.1,
                                 com_type='volume', cop_type='pressure', debug=False):
        """
        get vertices height along *z-axis* from ground-plane and pass it through function e(-wx) to get values for the heatmap.
        align and crop the heatmap so that it is it matches the Gt pressure map
        Args:
            mat_size: size of the pressure mat
            marker_positions: bottom left, top left, top right, bottom right
            cop_w: paramaeter for contact function
            vp: view padding empty space around contact region (in m)
        """
        # map 4 x 4 m area to a 512x512 image
        pressure_map = np.zeros((mat_size[1], mat_size[0]), dtype=np.float32)
        heatmap_point = np.zeros((heatmap_res, heatmap_res), dtype=np.float32)

        # uniformly sample vertices from the SMPL mesh so that the contact vertices are not biased to hands and face
        vertices = torch.tensor(vertices)[None, :, :]
        vertices_hd = HDfier().hdfy_mesh(vertices).cpu().numpy()[0]
        vertices = vertices.cpu().numpy()[0]

        # calculate values for heatmap
        vertex_height = vertices_hd[:, 2] - ground_plane_height
        vertex_height = vertex_height.astype(np.float64)

        # Get metric range. x-z plane is ground
        mat_bl_corner = marker_positions[0, :2] / 1000
        mat_tl_corner = marker_positions[1, :2] / 1000
        mat_tr_corner = marker_positions[2, :2] / 1000
        mat_br_corner = marker_positions[3, :2] / 1000
        m_x = mat_br_corner[0], mat_bl_corner[0]
        m_y = mat_tr_corner[1], mat_br_corner[1]
        m_range_x = m_x[1] - m_x[0]
        m_range_y = m_y[1] - m_y[0]

        # filter out vertices outside of the mat
        mask = (vertices_hd[:, 0] >= m_x[0]) & (vertices_hd[:, 0] <= m_x[1]) & (vertices_hd[:, 1] >= m_y[0]) & (
                vertices_hd[:, 1] <= m_y[1])
        vertex_height = vertex_height[mask]
        vertices_hd = vertices_hd[mask]

        # Normalize metric values to image range.
        max_mat_idx_x = mat_size[0] - 1
        max_mat_idx_y = mat_size[1] - 1
        v_x = np.rint((m_x[1] - vertices_hd[:, 0]) / m_range_x * (max_mat_idx_x)).astype(int) # flipped because global x axis is opposite of pressure mat x-axis
        v_y = np.rint((m_y[1] - vertices_hd[:, 1]) / m_range_y * (max_mat_idx_y)).astype(
            int)  # flipped because mat goes from bottom to up while image goes from top to bottom

        # assymetric function for inside and outside vertices
        inside_mask = (vertex_height < 0)
        outside_mask = (vertex_height >= 0)
        # v_z = inside_mask * np.exp(-4 * cop_w * vertex_height) + outside_mask * np.exp(-cop_w * vertex_height)
        v_z = inside_mask * (1 - cop_k * vertex_height) + outside_mask * np.exp(-cop_w * vertex_height)
        # v_z = np.exp(-cop_w*vertex_height)

        # normalize v_z to [0, 1]
        v_z = (v_z - np.min(v_z)) / (np.max(v_z) - np.min(v_z))

        # foe each overlapping vertex, add the value to the heatmap
        for i in range(len(v_x)):
            pressure_map[v_y[i], v_x[i]] += v_z[i]
        # normalize heatmap to [0, 1]
        pressure_map = (pressure_map - np.min(pressure_map)) / (np.max(pressure_map) - np.min(pressure_map))

        # flip the pressure_map because the GT is inverted
        # pressure_map = np.flipud(pressure_map)
        heatmap = vis_heatmap(pressure_map)
        return pressure_map, heatmap

    def iou(self, gt_pressure, pred_pressure):
        """
        Intersection over union
        :return:
        """
        binary_threholds = np.linspace(0, 1, 10)
        ious = []
        for th in binary_threholds:
            pred_binary = pred_pressure > th
            gt_binary = gt_pressure > th
            intersection = np.logical_and(pred_binary, gt_binary)
            union = np.logical_or(pred_binary, gt_binary)
            eps = 1e-13
            iou_score = np.sum(intersection) / (np.sum(union) + eps)
            ious.append(iou_score)
        max_iou_score = np.max(ious)
        max_th_idx = np.argmax(ious)
        max_th = binary_threholds[max_th_idx]
        return max_iou_score, max_th

    def draw_ious_graph(self):
        from numpy import exp, arange
        from pylab import meshgrid, cm, imshow, contour, clabel, colorbar, axis, title, show

        # the function that I'm going to plot
        x = arange(-3.0, 3.0, 0.1)
        y = arange(-3.0, 3.0, 0.1)
        X, Y = meshgrid(x, y)  # grid of point
        Z = z_func(X, Y)  # evaluation of the function on the grid

        im = imshow(Z, cmap=cm.RdBu)  # drawing the function
        # adding the Contour lines with labels
        cset = contour(Z, arange(-1, 1.5, 0.2), linewidths=2, cmap=cm.Set2)
        clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
        colorbar(im)  # adding the colobar on the right
        # latex fashion title
        title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')
        show()

    def evaluate_com(self, gt_com, vertices):
        # Note: the vertices should be aligned along y-axis and in world coordinates
        batch_size = vertices.shape[0]
        vertices = vertices.float()
        # calculate per part volume
        per_part_volume = self.compute_per_part_volume(vertices)
        # sample 20k vertices uniformly on the smpl mesh
        vertices_hd = self.hdfy_op.hdfy_mesh(vertices)
        # get volume per vertex id in the hd mesh
        volume_per_vert_hd = self.vertex_id_to_part_volume_mapping(per_part_volume, vertices.device)
        # calculate com using volume weighted mean
        com = torch.sum(vertices_hd * volume_per_vert_hd, dim=1) / torch.sum(volume_per_vert_hd, dim=1)
        com_error = torch.norm(com - gt_com, dim=1)
        return com, com_error

    def evaluate_pressure(self, gt_pressure, gt_cop_relative, vertices, mat_size, mat_size_global, marker_positions):
        """
        Evaluate the predicted pressure map against the ground truth pressure map.
        Args:
            gt_pressure:
            gt_cop_relative: cop relative to the pressure mat in mms
            vertices:
            mat_size: the resolution of the mat image
            mat_bbox: [tl_x, tl_y, br_x, br_y] of the mat in the image

        Returns:

        """
        # Note: the vertices should be aligned along y-axis and in world coordinates
        batch_size = vertices.shape[0]
        vertices = vertices.float()
        # calculate per part volume
        per_part_volume = self.compute_per_part_volume(vertices)
        # sample 20k vertices uniformly on the smpl mesh
        vertices_hd = self.hdfy_op.hdfy_mesh(vertices)
        # # get volume per vertex id in the hd mesh
        # volume_per_vert_hd = self.vertex_id_to_part_volume_mapping(per_part_volume, vertices.device)
        # # calculate com using volume weighted mean
        # com = torch.sum(vertices_hd * volume_per_vert_hd, dim=1) / torch.sum(volume_per_vert_hd, dim=1)

        # pressure based center of support
        ground_plane_height = 0.0
        eps = 1e-6
        vertex_height = (vertices_hd[:, :, 2] - ground_plane_height).double()
        inside_mask = (vertex_height < 0.0).float()
        outside_mask = (vertex_height >= 0.0).float()

        iou_first_best_iou = 0.0 # best iou score
        cop_error_first_best_cop_error = np.inf # best cop error
        frame_diff_first_best_frame_diff = np.inf
        for cop_k in self.cop_k_range:
            for cop_w in self.cop_w_range:
                pressure_weights = inside_mask * (1 - cop_k * vertex_height) + outside_mask * torch.exp(
                    - cop_w * vertex_height)
                pred_cos_global = torch.sum(vertices_hd * pressure_weights.unsqueeze(-1), dim=1) / (
                        torch.sum(pressure_weights, dim=1, keepdim=True) + eps)

                ## convert pred_cos relative to the pressure mat
                # Get metric range. x-z plane is ground
                mat_bl_corner = marker_positions[0, :2] / 1000
                mat_tl_corner = marker_positions[1, :2] / 1000
                mat_tr_corner = marker_positions[2, :2] / 1000
                mat_br_corner = marker_positions[3, :2] / 1000
                m_x = mat_br_corner[0], mat_bl_corner[0]
                m_y = mat_tr_corner[1], mat_br_corner[1]
                m_range_x = m_x[1] - m_x[0]
                m_range_y = m_y[1] - m_y[0]

                # convert all vertices relative to the pressure mat
                mat_size_global_x = mat_size_global[0]
                mat_size_global_y = mat_size_global[1]
                v_x_relative = (m_x[1] - vertices_hd[:, :, 0]) / m_range_x * (mat_size_global_x) # flipped because global x axis is opposite of pressure mat x-axis
                v_y_relative = (m_y[1] - vertices_hd[:, :, 1]) / m_range_y * (mat_size_global_y) # flipped because mat goes from bottom to up while image goes from top to bottom
                vertices_hd_relative = torch.stack([v_x_relative, v_y_relative, vertices_hd[:, :, 2]], dim=2)
                pred_cos_relative = torch.sum(vertices_hd_relative * pressure_weights.unsqueeze(-1), dim=1) / (
                        torch.sum(pressure_weights, dim=1, keepdim=True) + eps)
                pred_cos_relative = pred_cos_relative[0, :2].cpu().numpy()

                # compute l2 distance between gt and pred cop
                cop_error = np.linalg.norm(pred_cos_relative - gt_cop_relative, axis=0) # in metres

                # get pressure heatmap
                pred_pressure, pred_heatmap = self.generate_aligned_heatmap(vertices, mat_size, marker_positions,
                                                                            cop_w=cop_w, cop_k=cop_k)

                # calculate intersection-over-union between gt_pressure and predicted pressure map
                iou, binary_th = self.iou(gt_pressure, pred_pressure)
                # calculate mean frame diff
                frame_diff = np.mean(np.abs(gt_pressure - pred_pressure))
                if frame_diff <= frame_diff_first_best_frame_diff:
                    frame_diff_first_best_iou = iou
                    frame_diff_first_best_th = binary_th
                    frame_diff_first_best_cop_error = cop_error
                    frame_diff_first_best_frame_diff = frame_diff
                    frame_diff_first_best_pred_heatmap = pred_heatmap
                    frame_diff_first_best_pred_cop_relative = pred_cos_relative
                    frame_diff_first_best_cop_w = cop_w
                    frame_diff_first_best_cop_k = cop_k
                if iou >= iou_first_best_iou:
                    iou_first_best_iou = iou
                    iou_first_best_cop_error = cop_error
                    iou_first_best_frame_diff = frame_diff
                    iou_first_best_th = binary_th
                    iou_first_best_pred_heatmap = pred_heatmap
                    iou_first_best_pred_cop_relative = pred_cos_relative
                    iou_first_best_cop_w = cop_w
                    iou_first_best_cop_k = cop_k
                if cop_error <= cop_error_first_best_cop_error:
                    cop_error_first_best_iou = iou
                    cop_error_first_best_cop_error = cop_error
                    cop_error_first_best_frame_diff = frame_diff
                    cop_error_first_best_th = binary_th
                    cop_error_first_best_pred_heatmap = pred_heatmap
                    cop_error_first_best_pred_cop_relative = pred_cos_relative
                    cop_error_first_best_cop_w = cop_w
                    cop_error_first_best_cop_k = cop_k

        self.frame_diff_first.assign(iou=frame_diff_first_best_iou,
                                     cop_error=frame_diff_first_best_cop_error,
                                     frame_diff=frame_diff_first_best_frame_diff,
                                     binary_th=frame_diff_first_best_th,
                                     best_cop_w=frame_diff_first_best_cop_w,
                                     best_cop_k=frame_diff_first_best_cop_k,
                                     pred_heatmap=frame_diff_first_best_pred_heatmap,
                                     pred_cop_relative=frame_diff_first_best_pred_cop_relative)

        self.iou_first.assign(iou=iou_first_best_iou,
                                      cop_error=iou_first_best_cop_error,
                                      frame_diff = iou_first_best_frame_diff,
                                      binary_th=iou_first_best_th,
                                        best_cop_w=iou_first_best_cop_w,
                                        best_cop_k=iou_first_best_cop_k,
                                      pred_heatmap=iou_first_best_pred_heatmap,
                                      pred_cop_relative=iou_first_best_pred_cop_relative)
        self.cop_error_first.assign(iou=cop_error_first_best_iou,
                                            cop_error=cop_error_first_best_cop_error,
                                            frame_diff=cop_error_first_best_frame_diff,
                                            binary_th=cop_error_first_best_th,
                                            best_cop_w=cop_error_first_best_cop_w,
                                            best_cop_k=cop_error_first_best_cop_k,
                                            pred_heatmap=cop_error_first_best_pred_heatmap,
                                            pred_cop_relative=cop_error_first_best_pred_cop_relative)
