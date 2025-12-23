# modify from https://github.com/mit-han-lab/bevfusion
from typing import Any, Dict, Tuple, List

import numpy as np
import torch
import copy
import cv2
import math
from mmcv.transforms import BaseTransform
from PIL import Image

from mmdet3d.datasets import GlobalRotScaleTrans
from mmdet3d.registry import TRANSFORMS
from scipy.spatial.transform import Rotation as R
from ocamcamera import OcamCamera


@TRANSFORMS.register_module()
class ImageAug3D(BaseTransform):

    def __init__(self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip,
                 is_train, img_key=None):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train
        self.img_key = img_key

    def sample_augmentation(self, results):
        H, W = results['ori_shape'] if self.img_key is None else results[
            self.img_key]['ori_shape']
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(self, img, rotation, translation, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = Image.fromarray(img.astype('uint8'), mode='RGB')
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data[self.img_key]['img'] if self.img_key else data['img']
        new_imgs = []
        transforms = []
        for img in imgs:
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                data)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(np.array(new_img).astype(np.float32))
            transforms.append(transform.numpy())

        if self.img_key is None:
            data['img'] = new_imgs
            # update the calibration matrices
            data['img_aug_matrix'] = transforms
        else:
            data[self.img_key]['img'] = new_imgs
            data[self.img_key]['img_aug_matrix'] = transforms
        return data


@TRANSFORMS.register_module()
class BEVFusionRandomFlip3D:
    """Compared with `RandomFlip3D`, this class directly records the lidar
    augmentation matrix in the `data`."""

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flip_horizontal = np.random.choice([0, 1])
        flip_vertical = np.random.choice([0, 1])

        rotation = np.eye(3)
        if flip_horizontal:
            rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
            if 'points' in data:
                data['points'].flip('horizontal')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('horizontal')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, :, ::-1].copy()

        if flip_vertical:
            rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ rotation
            if 'points' in data:
                data['points'].flip('vertical')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('vertical')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, ::-1, :].copy()

        if 'lidar_aug_matrix' not in data:
            data['lidar_aug_matrix'] = np.eye(4)
        data['lidar_aug_matrix'][:3, :] = rotation @ data[
            'lidar_aug_matrix'][:3, :]
        return data


@TRANSFORMS.register_module()
class BEVFusionGlobalRotScaleTrans(GlobalRotScaleTrans):
    """Compared with `GlobalRotScaleTrans`, the augmentation order in this
    class is rotation, translation and scaling (RTS)."""

    def transform(self, input_dict: dict) -> dict:
        """Private function to rotate, scale and translate bounding boxes and
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans' and `gt_bboxes_3d` are updated
            in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._trans_bbox_points(input_dict)
        self._scale_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'T', 'S'])

        lidar_augs = np.eye(4)
        lidar_augs[:3, :3] = input_dict['pcd_rotation'].T * input_dict[
            'pcd_scale_factor']
        lidar_augs[:3, 3] = input_dict['pcd_trans'] * \
            input_dict['pcd_scale_factor']

        if 'lidar_aug_matrix' not in input_dict:
            input_dict['lidar_aug_matrix'] = np.eye(4)
        input_dict[
            'lidar_aug_matrix'] = lidar_augs @ input_dict['lidar_aug_matrix']

        return input_dict
    

@TRANSFORMS.register_module()
class ResizeCropFlipImage(BaseTransform):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, data_aug_conf=None, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.img_key= data_aug_conf.get('img_key', None)

    def transform(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """

        imgs = results[self.img_key]['img'] if self.img_key else results['img']
        cam2img = results[self.img_key]['cam2img'] if self.img_key else results['cam2img']
        N = len(imgs)
        new_imgs = []
        new_cam2imgs = []
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
        
        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            # augmentation (resize, crop, horizontal flip, rotate)
            # different view use different aug (BEV Det)
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            new_cam2img = np.eye(4)
            new_cam2img[:3, :3] = ida_mat @ cam2img[i][:3, :3]
            new_cam2imgs.append(new_cam2img)
        new_cam2imgs = np.array(new_cam2imgs)
        
        if self.img_key is None:
            results['img'] = new_imgs
            results['cam2img'] = new_cam2imgs
        else:
            results[self.img_key]['img'] = new_imgs
            results[self.img_key]['cam2img'] = new_cam2imgs

        return results

    def _get_rot(self, h):

        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim'])) *
                newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.mean(self.data_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate


@TRANSFORMS.register_module()
class GridMask(BaseTransform):

    def __init__(
        self,
        use_h,
        use_w,
        max_epoch,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=0,
        prob=1.0,
        fixed_prob=False,
        img_key=None,
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.epoch = None
        self.max_epoch = max_epoch
        self.fixed_prob = fixed_prob
        self.img_key = img_key

    def set_epoch(self, epoch):
        self.epoch = epoch
        if not self.fixed_prob:
            self.set_prob(self.epoch, self.max_epoch)

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * self.epoch / self.max_epoch

    def transform(self, results):
        if np.random.rand() > self.prob:
            return results
        imgs = results['img'] if self.img_key is None else results[self.img_key]['img']
        h = imgs[0].shape[0]
        w = imgs[0].shape[1]
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.length = np.random.randint(1, d)
        else:
            self.length = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.length, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.length, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h,
                    (ww - w) // 2:(ww - w) // 2 + w]

        mask = mask.astype(np.float32)
        mask = mask[:, :, None]
        if self.mode == 1:
            mask = 1 - mask

        # mask = mask.expand_as(imgs[0])
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            imgs = [x * mask + offset for x in imgs]
        else:
            imgs = [x * mask for x in imgs]

        if self.img_key is None:
            results.update(img=imgs)
        else:
            results[self.img_key].update(img=imgs)

        return results
    

@TRANSFORMS.register_module()
class MultiViewFisheyePerspectiveProjection(BaseTransform):
    """Multi-view fisheye perspective projection.

    Args:
        image_size (Tuple[int, int]): The size of the output image.
            Default: (400, 400).
        perspective_fov (float): The field of view of the perspective camera.
            Default: 110.0.
        num_views (int): The number of views. Default: 2.
        camera_orientation (List[float]): The orientation of the camera.
            Default: [-55, 55].
        ocam_path (str): The path to the ocam camera calibration file.
            Default: 'data/CarlaCollection/calib_results.txt'.
        ocam_fov (float): The field of view of the ocam camera.
            Default: 220.0.
    """

    def __init__(self,
                 image_size: Tuple[int, int] = (400, 400),
                 perspective_fov: float = 110.0,
                 num_imgs_of_per_view: int = 2,
                 camera_orientation: List[float] = [-55, 55],
                 ocam_path: str = 'data/CarlaCollection/calib_results.txt',
                 ocam_fov: float = 220.0) -> None:
        self.image_size = image_size
        self.perspective_fov = perspective_fov
        self.num_imgs = num_imgs_of_per_view
        self.camera_orientation = camera_orientation
        assert len(camera_orientation) == self.num_imgs, \
            f'camera_orientation should have {self.num_imgs} elements, ' \
            f'but got {len(camera_orientation)}.'
        
        self.omni_ocam = OcamCamera(filename=ocam_path, fov=ocam_fov)

    @staticmethod
    def generate_perspective_map(ocam, img, W=500, H=500, fov_deg=110, yaw_deg=0):
        fov_rad = np.radians(fov_deg)
        yaw_rad = np.radians(yaw_deg)

        # Compute focal length from FoV and image width
        f = (W / 2) / np.tan(fov_rad / 2)

        # Create image plane in camera frame (centered at 0,0,z)
        x = np.linspace(-W / 2, W / 2 - 1, W)
        y = np.linspace(-H / 2, H / 2 - 1, H)
        x_grid, y_grid = np.meshgrid(x, y)

        # These are 3D points on the image plane at z = f
        rays = np.stack([x_grid, y_grid, np.full_like(x_grid, f)], axis=-1)  # shape: (H, W, 3)

        # Rotate camera by yaw angle around y-axis
        R_yaw = R.from_euler('y', yaw_deg, degrees=True).as_matrix()
        rays_rotated = rays @ R_yaw.T  # shape: (H, W, 3)

        # Reshape to (3, N) for ocam.world2cam
        point3D = rays_rotated.reshape(-1, 3).T
        mapx, mapy = ocam.world2cam(point3D)
        mapx = mapx.reshape(H, W).astype(np.float32)
        mapy = mapy.reshape(H, W).astype(np.float32)

        # Sample fisheye image
        out = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return out, R_yaw

    @staticmethod
    def get_camera_intrinsics(img_height, img_width, view_fov):
        calibration = np.eye(4, dtype=np.float32)
        calibration[0, 2] = img_width / 2.0
        calibration[1, 2] = img_height / 2.0
        calibration[0, 0] = img_width / (2.0 * np.tan(view_fov * np.pi / 360.0))
        calibration[1, 1] = calibration[0, 0]
        return calibration
        
    
    def transform(self, results):
        ori_imgs = results['cam_fisheye']['img']
        new_imgs = list()
        lidar2cam, cam2lidar, lidar2img, cam2img = list(), list(), list(), list()
        # import matplotlib.pyplot as plt
        for i, img in enumerate(ori_imgs):
            # plt.figure('ori_img_{}'.format(i))
            # plt.imshow(img.astype(np.uint8))
            for j in range(self.num_imgs):
                w, h = self.image_size
                new_img, _R_yaw = self.generate_perspective_map(
                    self.omni_ocam, img, w, h, self.perspective_fov,  self.camera_orientation[j])
                new_imgs.append(new_img)
                R_yaw = np.eye(4)
                R_yaw[:3, :3] = _R_yaw
                
                _cam2lidar = copy.deepcopy(results['cam_fisheye']['cam2lidar'][i])
                _cam2lidar = _cam2lidar @ R_yaw
                cam2lidar.append(_cam2lidar)

                _lidar2cam = np.linalg.inv(_cam2lidar)
                lidar2cam.append(_lidar2cam)

                # _lidar2cam = copy.deepcopy(results['cam_fisheye']['lidar2cam'][i])
                # # _lidar2cam = np.linalg.inv(lidar2cam_M) @ R_yaw @ lidar2cam_M @ _lidar2cam
                # _lidar2cam = lidar2cam_M @ R_yaw @ cam2lidar_M @ _lidar2cam
                # lidar2cam.append(_lidar2cam)
               
                _cam2img = self.get_camera_intrinsics(h, w, self.perspective_fov)
                cam2img.append(_cam2img)
                _lidar2img = np.dot(_cam2img, _lidar2cam)
                lidar2img.append(_lidar2img)

        #         plt.figure('new_img_{}_{}'.format(i, j))
        #         plt.imshow(new_img.astype(np.uint8))
        # plt.show()

        lidar2cam = np.stack(lidar2cam, axis=0)
        cam2lidar = np.stack(cam2lidar, axis=0)
        lidar2img = np.stack(lidar2img, axis=0)
        cam2img = np.stack(cam2img, axis=0)
        results['cam_fisheye']['img'] = new_imgs
        results['cam_fisheye']['lidar2cam'] = lidar2cam
        results['cam_fisheye']['cam2lidar'] = cam2lidar
        results['cam_fisheye']['lidar2img'] = lidar2img
        results['cam_fisheye']['cam2img'] = cam2img
        results['cam_fisheye']['img_shape'] = self.image_size
        results['cam_fisheye']['ori_shape'] = self.image_size
        results['cam_fisheye']['pad_shape'] = self.image_size

        # print('new_imgs', len(new_imgs))

        return results
    

@TRANSFORMS.register_module()
class MultiViewFisheyeCylindricalProjection(BaseTransform):
    def __init__(self,
                 image_size: Tuple[int, int] = (400, 400),
                 horizontal_range: List[float] = [-math.radians(110), math.radians(110)],
                 vertical_range: List[float] = [-math.radians(45), math.radians(45)],
                 ocam_path: str = 'data/CarlaCollection/calib_results.txt',
                 ocam_fov: float = 220.0) -> None:
        self.image_size = image_size
        self.horizontal_range = horizontal_range
        self.vertical_range = vertical_range
        self.omni_ocam = OcamCamera(filename=ocam_path, fov=ocam_fov)

    @staticmethod
    def generate_cylindrical_map(ocam, img, W=500, H=500, horizontal_range=None, vertical_range=None):
        h_min, h_max = horizontal_range  # azimuth φ
        v_min, v_max = vertical_range    # elevation θ
        y_min, y_max = np.tan(v_min), np.tan(v_max)

        phi = np.linspace(h_min, h_max, W)           # shape: (W,)
        y = np.linspace(y_min, y_max, H)             # shape: (H,)
        phi_grid, y_grid = np.meshgrid(phi, y, indexing='xy')  # shape: (H, W)
        
        # Map to cylinder surface (radius = 1)
        x = np.sin(phi_grid)
        z = np.cos(phi_grid)
        point3D = np.stack([x, y_grid, z], axis=-1)  # shape: (H, W, 3)
        point3D = point3D.reshape(-1, 3).T  # shape: (3, H*W)

        mapx, mapy = ocam.world2cam(point3D)
        mapx = mapx.reshape(H, W).astype(np.float32)
        mapy = mapy.reshape(H, W).astype(np.float32)

        # Sample fisheye image
        out = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return out
    
    def transform(self, results):
        ori_imgs = results['cam_fisheye']['img']
        new_imgs = list()

        # import matplotlib.pyplot as plt
        for i, img in enumerate(ori_imgs):
            # plt.figure('ori_img_{}'.format(i))
            # plt.imshow(img.astype(np.uint8))
           
            w, h = self.image_size
            new_img = self.generate_cylindrical_map(
                self.omni_ocam, img, w, h, self.horizontal_range, self.vertical_range)
            new_imgs.append(new_img)

        #     plt.figure('new_img_{}'.format(i))
        #     plt.imshow(new_img.astype(np.uint8))
        # plt.show()

        results['cam_fisheye']['img'] = new_imgs
        results['cam_fisheye']['img_shape'] = self.image_size
        results['cam_fisheye']['ori_shape'] = self.image_size
        results['cam_fisheye']['pad_shape'] = self.image_size
        return results
