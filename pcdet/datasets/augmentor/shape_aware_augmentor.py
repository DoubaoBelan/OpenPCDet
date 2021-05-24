import pickle

import numpy as np
import mayavi.mlab as mlab

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils
from ...utils import common_utils
# from ....tools.visual_utils import visualize_utils as V
from visual_utils import visualize_utils as V


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
    def __init__(self, sampler_config={"p0": 0.5, "p1": 0.5, "p2": 0.5}):
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

    @staticmethod
    def dropout(gt_boxes, points, pyramid_pts, probability):
        """
        Args:
            gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
            points: (M, 3 + C),
            rot_range: [min, max]
        Returns:
        """
        return gt_boxes, points
        pass

    @staticmethod
    def sparsify(gt_boxes, points, probability):
        """
        Args:
            gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
            points: (M, 3 + C),
            rot_range: [min, max]
        Returns:
        """
        pass

    @staticmethod
    def swap(gt_boxes, points, probability):
        """
        Args:
            gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
            points: (M, 3 + C),
            rot_range: [min, max]
        Returns:
        """
        pass

    def extract_pyramid_points_idxs(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns: [{'pyramid_0,...5': (M', )}] -> len=gt_boxes_num
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
            label2 = label02 & (cam_pts[:, 0] < 0)
            label1 = label13 & (cam_pts[:, 1] < 0)
            label3 = label13 & (cam_pts[:, 1] >= 0)
            label4 = label45 & (cam_pts[:, 2] >= 0)
            label5 = label45 & (cam_pts[:, 2] < 0)
            pyramid_points['pyramid_0'] = valid[np.where(label0)[0]]
            pyramid_points['pyramid_1'] = valid[np.where(label1)[0]]
            pyramid_points['pyramid_2'] = valid[np.where(label2)[0]]
            pyramid_points['pyramid_3'] = valid[np.where(label3)[0]]
            pyramid_points['pyramid_4'] = valid[np.where(label4)[0]]
            pyramid_points['pyramid_5'] = valid[np.where(label5)[0]]
            pyramid_pts_idxs.append(pyramid_points)
        return pyramid_pts_idxs

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
        obj_pyramid_pts = self.extract_pyramid_points_idxs(data_dict)
        points = data_dict['points']
        V.draw_scenes(
            points=points[obj_pyramid_pts[0]['pyramid_0']], ref_boxes=data_dict['gt_boxes']
        )
        mlab.show(stop=True)
        exit(1)
        # points = points[in_box, :]
        ## project
        # V.draw_scenes(
        #     points=data_dict['points'][:, :], ref_boxes=data_dict['gt_boxes']
        # )
        # mlab.show(stop=True)
        # V.draw_scenes(
        #     points=points[:, :], ref_boxes=data_dict['gt_boxes']
        # )
        # mlab.show(stop=True)
        import pdb
        pdb.set_trace()
        # points = points[np.newaxis, np.newaxis, :, :]
        # (N, M', 3+C)
        # (N, 6, M'', 3+C), sum(M'') = M'
        ## judge
        frustum_headings = gt_heading, gt_heading - np.pi / 2, gt_heading - np.pi, gt_heading - 1.5 * np.pi, (
        0, 0, -1), (0, 0, 1)  # (N, 3, 3)
        pass

        # gt_boxes = data_dict['gt_boxes']
        # gt_names = data_dict['gt_names'].astype(str)
        # existed_boxes = gt_boxes
        # total_valid_sampled_dict = []
        # for class_name, sample_group in self.sample_groups.items():
        #     if self.limit_whole_scene:
        #         num_gt = np.sum(class_name == gt_names)
        #         sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)
        #     if int(sample_group['sample_num']) > 0:
        #         sampled_dict = self.sample_with_fixed_number(class_name, sample_group)

        #         sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

        #         if self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False):
        #             sampled_boxes = box_utils.boxes3d_kitti_fakelidar_to_lidar(sampled_boxes)

        #         iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
        #         iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
        #         iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
        #         iou1 = iou1 if iou1.shape[1] > 0 else iou2
        #         valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
        #         valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
        #         valid_sampled_boxes = sampled_boxes[valid_mask]

        #         existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes), axis=0)
        #         total_valid_sampled_dict.extend(valid_sampled_dict)

        # sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]
        # if total_valid_sampled_dict.__len__() > 0:
        #     data_dict = self.add_sampled_boxes_to_scene(data_dict, sampled_gt_boxes, total_valid_sampled_dict)

        # data_dict.pop('gt_boxes_mask')
        return data_dict
