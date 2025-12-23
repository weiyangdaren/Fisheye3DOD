from typing import List, Union, Optional, Dict
from pathlib import Path

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.datasets import Det3DDataset
from mmdet3d.structures import LiDARInstance3DBoxes


@DATASETS.register_module()
class Fisheye3DODDataset(Det3DDataset):
    METAINFO = {
        'classes':
        ('Car', 'Van', 'Truck', 'Bus', 'Pedestrian', 'Cyclist'),
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Dict],
                 test_mode: bool = False,
                 metainfo: Optional[Dict[str, str]] = None,
                 **kwargs):

        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            metainfo=metainfo,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)

    def parse_anno_info(self, info: dict) -> dict:
        ann_info = dict()
        if len(info['annos']['gt_names']) > 0:
            gt_bboxes_3d = LiDARInstance3DBoxes(info['annos']['gt_boxes'], origin=(0.5, 0.5, 0.5))
            ann_info['gt_bboxes_3d'] = gt_bboxes_3d
            ann_info['gt_labels_3d'] = np.vectorize(
                self.metainfo['categories'].get)(info['annos']['gt_names']).astype(np.int64)
        else:
            ann_info['gt_bboxes_3d'] = LiDARInstance3DBoxes(np.zeros((0, 7), dtype=np.float32))
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)
        info.pop('annos')

        for label in ann_info['gt_labels_3d']:
            if label != -1:
                self.num_ins_per_cat[label] += 1

        return ann_info

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:

        info['pc_info']['lidar_path'] = Path(
            self.data_root) / info['pc_info'].pop('pts_path')
        info['pc_info']['semantic_lidar_path'] = Path(
            self.data_root) / info['pc_info'].pop('semantic_pts_path')
        info['pc_info']['num_pts_feats'] = info['pc_info'].pop('num_features')

        for cam_type, cam_list in info['cam_info'].items():
            for cam_name, cam_info in cam_list.items():
                cam_info['cam_path'] = Path(self.data_root) / cam_info['cam_path']
                if cam_type == 'cam_dvs':
                    cam_info['cam_npz_path'] = Path(
                        self.data_root) / cam_info['cam_npz_path']
        # info['ann_info'] = self.parse_anno_info(info)

        if not self.test_mode:
            # used in training
            info['ann_info'] = self.parse_anno_info(info)
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = self.parse_anno_info(info)

        return info
