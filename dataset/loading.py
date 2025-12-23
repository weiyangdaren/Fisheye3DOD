import copy
from typing import Optional, Union, List

import mmcv
import numpy as np
from mmengine.fileio import get

from mmdet3d.datasets.transforms import LoadPointsFromFile, LoadMultiViewImageFromFiles
from mmdet3d.structures.points import get_points_type
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadFisheye3DODPointsFromFile(LoadPointsFromFile):
    def __init__(self,
                 coord_type: str,
                 load_dim: int = 6,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 shift_height: bool = False,
                 use_color: bool = False,
                 norm_intensity: bool = False,
                 norm_elongation: bool = False,
                 backend_args: Optional[dict] = None,
                 load_point_type: str = 'lidar') -> None:
        assert load_point_type in ['lidar', 'semantic_lidar'], \
            f'Unsupported load_point_type {load_point_type}, available options are: lidar, semantic_lidar'  # noqa: E501
        self.load_point_type = load_point_type  # 'lidar' or 'semantic_lidar'
        super().__init__(coord_type,
                         load_dim,
                         use_dim,
                         shift_height,
                         use_color,
                         norm_intensity,
                         norm_elongation,
                         backend_args)

    def transform(self, results: dict) -> dict:
        pts_file_path = results['pc_info'][f'{self.load_point_type}_path']
        points = self._load_points(pts_file_path)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        if self.norm_intensity:
            assert len(self.use_dim) >= 4, \
                f'When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}'  # noqa: E501
            points[:, 3] = np.tanh(points[:, 3])
        if self.norm_elongation:
            assert len(self.use_dim) >= 5, \
                f'When using elongation norm, expect used dimensions >= 5, got {len(self.use_dim)}'  # noqa: E501
            points[:, 4] = np.tanh(points[:, 4])
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)

        if self.load_point_type == 'lidar':
            results['points'] = points
        else:
            results[f'{self.load_point_type}_points'] = points
        return results


@TRANSFORMS.register_module()
class LoadFisheye3DODMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'unchanged',
                 backend_args: Optional[dict] = None,
                 num_views: int = 5,
                 num_ref_frames: int = -1,
                 test_mode: bool = False,
                 set_default_scale: bool = True,
                 load_cam_type: str = 'cam_nusc',
                 load_cam_names: Optional[List[str]] = None) -> None:
        if load_cam_names is None:
            load_cam_names = ['nu_rgb_camera_front']
        assert load_cam_type in ['cam_nusc',
                                 'cam_fisheye', 'cam_rgb', 'cam_dvs']
        self.load_cam_type = load_cam_type
        self.load_cam_names = load_cam_names
        self.num_views = len(load_cam_names)

        super().__init__(to_float32,
                         color_type,
                         backend_args,
                         num_views,
                         num_ref_frames,
                         test_mode,
                         set_default_scale)

    def transform(self, results: dict) -> Optional[dict]:
        filename, lidar2cam, cam2lidar, lidar2img, cam2img = [], [], [], [], []
        ret_dict = dict()

        for cam_name in self.load_cam_names:
            filename.append(results['cam_info']
                            [self.load_cam_type][cam_name]['cam_path'])
            lidar2cam.append(
                results['cam_info'][self.load_cam_type][cam_name]['lidar2cam'])
            cam2lidar.append(
                results['cam_info'][self.load_cam_type][cam_name]['cam2lidar'])
            lidar2img.append(
                results['cam_info'][self.load_cam_type][cam_name]['lidar2img'])
            cam2img.append(
                results['cam_info'][self.load_cam_type][cam_name]['cam2img'])
                # cam_intrinsic = np.eye(4)
                # cam_intrinsic[:3, :3] = results['cam_info'][self.load_cam_type][cam_name]['cam_intrinsic']
                # cam2img.append(cam_intrinsic)

        ret_dict['img_path'] = filename
        ret_dict['lidar2cam'] = np.stack(lidar2cam, axis=0)
        ret_dict['cam2lidar'] = np.stack(cam2lidar, axis=0)
        ret_dict['lidar2img'] = np.stack(lidar2img, axis=0)
        if self.load_cam_type != 'cam_fisheye':
            ret_dict['cam2img'] = np.stack(cam2img, axis=0)
            ret_dict['ori_cam2img'] = copy.deepcopy(ret_dict['cam2img'])


        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]

        imgs = [
            mmcv.imfrombytes(img_byte, flag=self.color_type)
            for img_byte in img_bytes
        ]
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        ret_dict['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        ret_dict['img'] = [img[..., i] for i in range(img.shape[-1])]
        ret_dict['img_shape'] = img.shape[:2]
        ret_dict['ori_shape'] = img.shape[:2]
        # Set initial values for default meta_keys
        ret_dict['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            ret_dict['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        ret_dict['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        ret_dict['num_views'] = self.num_views
        ret_dict['num_ref_frames'] = self.num_ref_frames
        # results.pop('cam_info')
        results[self.load_cam_type] = ret_dict
        return results
