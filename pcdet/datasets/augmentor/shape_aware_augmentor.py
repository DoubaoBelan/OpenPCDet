import pickle

import numpy as np
# import mayavi.mlab as mlab

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils
from ...utils import common_utils
# from ....tools.visual_utils import visualize_utils as V
# from visual_utils import visualize_utils as V


def global_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    return gt_boxes, points


class ShapeAwareAugmentor(object):
    def __init__(self, sampler_config={"DROPOUT_PROBABILITY": 0.5,
                                       "SPARSE_PROBABILITY": 0.5,
                                       "SPARSE_RATIO": 0.5}):
        """
        Args:
            sampler_config:
                p0: (N, 3 + C_in)
                p1: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                p2: optional, (N), string
                ...

        Returns:
        """
        self.config = sampler_config

    def extract_pyramid_points_idxs(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns: (out_box_points, [{'pyramid_0,...5': (M', )}] -> len=gt_boxes_num)
        """
        gt_boxes = data_dict['gt_boxes']
        N = gt_boxes.shape[0]  # FIXME when gt_boxes is empty
        # Step1. get gt_box transformation matrix (N, 3, 4)
        gt_heading = gt_boxes[:, 6][:, np.newaxis, np.newaxis]  # (N, 1)
        # rot matrix (N, 3, 3)
        rot_row1 = np.concatenate((np.cos(gt_heading), -np.sin(gt_heading), np.zeros([N, 1, 1])), axis=2)
        rot_row2 = np.concatenate((np.sin(gt_heading), np.cos(gt_heading), np.zeros([N, 1, 1])), axis=2)
        rot_row3 = np.concatenate((np.zeros([N, 1, 1]), np.zeros([N, 1, 1]), np.ones([N, 1, 1])), axis=2)
        rot = np.concatenate((rot_row1, rot_row2, rot_row3), axis=1)  # (N, 3, 3)
        # translation
        gt_centers = gt_boxes[:, (0, 1, 2)]  # (N, 3)
        gt_centers = gt_centers[:, :, np.newaxis]
        # assemble transformation matrix, [R.T, -R.T*t]
        transformation = np.concatenate((rot, gt_centers), axis=2)
        transformation[:, 0: 3, 0: 3] = np.transpose(transformation[:, 0: 3, 0: 3], (0, 2, 1))
        transformation[:, 0: 3, 3, None] = - transformation[:, 0: 3, 0: 3] @ transformation[:, 0:3, 3, None]
        # Step2. extract point cloud in bbox (N, M', 3+C) [(M', 3+C)] -> len=N
        lwh = data_dict['gt_boxes'][:, (3, 4, 5)]  # (N, 3)
        xyz_max = (lwh / 2)[:, :, np.newaxis]
        xyz_min = (- lwh / 2)[:, :, np.newaxis]
        xyz_range = np.concatenate((xyz_max, xyz_min), axis=2)
        points = data_dict['points'][:, 0:3]
        points = points[np.newaxis, :, :]
        points = np.repeat(points, N, axis=0)
        points = np.transpose(points, (0, 2, 1))
        # corners3d = box_utils.boxes_to_corners_3d(gt_boxes)
        points = transformation[:, 0:3, 0:3] @ points + transformation[:, 0:3, 3, None]
        in_box = (points < xyz_range[:, :, 0, None]) & (points > xyz_range[:, :, 1, None])
        in_box = in_box.all(axis=1)
        out_box = ~(in_box.any(axis=0))
        out_box_points = data_dict['points'][out_box, :]
        box_pts_idxs = []
        assert (N == in_box.shape[0])
        for i in range(N):
            box_pts_idxs.append(np.where(in_box[i, :])[0])
            # V.draw_scenes(
            #     points=data_dict['points'][box_pts_idxs[i], :], ref_boxes=gt_boxes
            # )
            # mlab.show(stop=True)
        # Step3. extract point idxs in pyramid [{'pyramid_0,...5': (M', 3+C)}] -> len=N
        pyramid_pts_idxs = []
        lwhs = data_dict['gt_boxes'][:, (3, 4, 5)]  # (N, 3)
        points = data_dict['points']
        for obj in range(N):
            valid = box_pts_idxs[obj]
            ori_pts = points[valid, :]
            pyramid_points = dict()
            cam_pts = ori_pts[:, 0:3] @ transformation[obj, 0:3, 0:3].transpose() + \
                      transformation[obj, 0:3, 3, None].transpose()
            y_div_x = cam_pts[:, 1] / cam_pts[:, 0]
            z_div_x = cam_pts[:, 2] / cam_pts[:, 0]
            z_div_y = cam_pts[:, 2] / cam_pts[:, 1]
            w_div_l = lwhs[obj, 1] / lwhs[obj, 0]
            h_div_l = lwhs[obj, 2] / lwhs[obj, 0]
            h_div_w = lwhs[obj, 2] / lwhs[obj, 1]
            label02 = (z_div_x <= h_div_l) & (y_div_x <= w_div_l)
            label13 = (z_div_y <= h_div_w) & (y_div_x >= w_div_l)
            label45 = (z_div_y >= h_div_w) & (z_div_x >= h_div_l)
            label0 = label02 & (cam_pts[:, 0] >= 0)
            label2 = label02 & (cam_pts[:, 0] <= 0)
            label1 = label13 & (cam_pts[:, 1] <= 0)
            label3 = label13 & (cam_pts[:, 1] >= 0)
            label4 = label45 & (cam_pts[:, 2] >= 0)
            label5 = label45 & (cam_pts[:, 2] <= 0)
            pyramid_points['pyramid_0'] = valid[np.where(label0)[0]]
            pyramid_points['pyramid_1'] = valid[np.where(label1)[0]]
            pyramid_points['pyramid_2'] = valid[np.where(label2)[0]]
            pyramid_points['pyramid_3'] = valid[np.where(label3)[0]]
            pyramid_points['pyramid_4'] = valid[np.where(label4)[0]]
            pyramid_points['pyramid_5'] = valid[np.where(label5)[0]]
            pyramid_pts_idxs.append(pyramid_points)
        return out_box_points, pyramid_pts_idxs

    @staticmethod
    def dropout(pyramid_idxs, probability=0.2):
        """
        Args:
            pyramid_idxs: [{'pyramid_0,...5': (M', )}] -> len=gt_boxes_num
            probability: ,
        Returns: pyramid_idxs
        """
        valid_pyramid_num = 0
        for idxs in pyramid_idxs:
            valid_pyramid_num += len(idxs.keys())
        drop_out_num = np.int(valid_pyramid_num * probability)

        N = len(pyramid_idxs)
        pyramid_num = 6 * N
        assert(valid_pyramid_num <= pyramid_num)
        idxs = list(range(pyramid_num))
        import random
        random.shuffle(idxs)

        drop = 0
        for del_idx in idxs:
            row = del_idx // 6
            col = del_idx % 6
            key = 'pyramid_%s' % col
            if key in pyramid_idxs[row]:
                drop += 1
                pyramid_idxs[row].pop(key)
            if drop >= drop_out_num:
                break

        return pyramid_idxs

    @staticmethod
    def sparsify(pyramid_idxs, probability=0.2, sparse_ratio=0.5):
        """
        Args:
            pyramid_idxs: [{'pyramid_0,...5': (M', )}] -> len=gt_boxes_num
            probability: ,
        Returns: pyramid_idxs
        """
        # TODO call furthest_point_sampling_wrapper pcdet/ops/pointnet2/pointnet2_batch/pointnet2_utils.py line28
        valid_pyramid_num = 0
        for idxs in pyramid_idxs:
            valid_pyramid_num += len(idxs.keys())
        sparse_num = np.int(valid_pyramid_num * probability)

        N = len(pyramid_idxs)
        pyramid_num = 6 * N
        assert(valid_pyramid_num <= pyramid_num)
        idxs = list(range(pyramid_num))
        import random
        random.shuffle(idxs)

        num = 0
        for sparse_idx in idxs:
            row = np.int(sparse_idx) // 6
            col = np.int(sparse_idx) % 6
            key = 'pyramid_%s' % col
            if key in pyramid_idxs[row].keys():
                num += 1
                cur_pyramid_idxs = pyramid_idxs[row][key]
                point_num = cur_pyramid_idxs.shape[0]
                np.random.shuffle(cur_pyramid_idxs)
                pyramid_idxs[row][key] = cur_pyramid_idxs[0: np.int((1-sparse_ratio)*point_num)]
            if num >= sparse_num:
                break

        return pyramid_idxs

    @staticmethod
    def swap(gt_boxes, points, probability):
        # TODO(LIJINGWEI)
        pass

    def cluster_points(self, full_points, obj_pyramid_idxs, out_points):
        for obj in obj_pyramid_idxs:
            for key in obj.keys():
                out_points = np.concatenate((out_points, full_points[obj[key]]), axis=0)
        return out_points

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        out_box_pts, obj_pyramid_pts_idxs = self.extract_pyramid_points_idxs(data_dict)
        num = 0
        for obj in obj_pyramid_pts_idxs:
            for key in obj.keys():
                num += obj[key].shape[0]
        obj_pyramid_pts_idxs = ShapeAwareAugmentor.dropout(obj_pyramid_pts_idxs,
                                                           probability=self.config['DROPOUT_PROBABILITY'])
        obj_pyramid_pts_idxs = ShapeAwareAugmentor.sparsify(obj_pyramid_pts_idxs,
                                                            probability=self.config['SPARSE_PROBABILITY'],
                                                            sparse_ratio=self.config['SPARSE_RATIO'])
        data_dict['points'] = self.cluster_points(data_dict['points'], obj_pyramid_pts_idxs, out_box_pts)
        return data_dict
