from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from mmengine.utils.dl_utils import TimeCounter


@MODELS.register_module()
class BEVDet(Base3DDetector):
    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        img_backbone: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        depth_head: Optional[dict] = None,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        **kwargs
    ) -> None:

        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(
            view_transform) if view_transform is not None else None
        self.pts_backbone = MODELS.build(
            pts_backbone) if pts_backbone is not None else None
        self.pts_neck = MODELS.build(
            pts_neck) if pts_neck is not None else None
        if bbox_head is not None:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            bbox_head.update(test_cfg=pts_test_cfg)
            self.bbox_head = MODELS.build(bbox_head)
        else:
            self.bbox_head = None

        self.depth_head = MODELS.build(
            depth_head) if depth_head is not None else None
        
        if self.training and train_cfg:
            self.img_key= train_cfg.input_key.img_key
            self.lidar_key = train_cfg.input_key.lidar_key
        
        if not self.training and test_cfg:
            self.img_key= test_cfg.input_key.img_key
            self.lidar_key = test_cfg.input_key.lidar_key

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None
    
    @property
    def with_depth_head(self):
        """bool: Whether the detector has a depth head."""
        return hasattr(self, 'depth_head') and self.depth_head is not None

    def extract_cam_feat(
        self,
        x,
        points,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:

        B, N, C, H, W = x.shape
        x = x.view(B*N, C, H, W)
        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        img_feat = x.view(B, int(BN / B), C, H, W)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(
                img_feat,
                points,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas
            )
        
        if isinstance(x, tuple):
            bev_feat = x[0]
            img_feat = x[1]
        else:
            bev_feat = x
            
        return bev_feat, img_feat

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get(self.img_key, None)
        points = batch_inputs_dict.get(self.lidar_key, None)

        assert imgs is not None, 'Image data is required.'
        if self.training:
            assert (self.with_depth_head and points) or (not self.with_depth_head and not points), \
                "When self.training, both self.with_depth_head and self.points must either both exist or both be None."

        imgs = imgs.contiguous()
        lidar2camera, lidar2image, camera_intrinsics, camera2lidar = [], [], [], []
        img_aug_matrix, lidar_aug_matrix = [], []
        for i, meta in enumerate(batch_input_metas):
            lidar2camera.append(meta[self.img_key]['lidar2cam'])
            lidar2image.append(meta[self.img_key]['lidar2img'])
            camera_intrinsics.append(meta[self.img_key].get('cam2img', np.eye(4)))  # fisheye camera intrinsic matrix is not available
            camera2lidar.append(meta[self.img_key]['cam2lidar'])
            img_aug_matrix.append(meta[self.img_key].get('img_aug_matrix', np.eye(4)))
            lidar_aug_matrix.append(
                meta[self.img_key].get('lidar_aug_matrix', np.eye(4)))

        lidar2camera = imgs.new_tensor(np.asarray(lidar2camera))
        lidar2image = imgs.new_tensor(np.asarray(lidar2image))
        camera_intrinsics = imgs.new_tensor(np.asarray(camera_intrinsics))
        camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
        img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
        lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))

        bev_feat, img_feat = self.extract_cam_feat(
            imgs, 
            points, 
            lidar2camera,
            lidar2image,
            camera_intrinsics, 
            camera2lidar, 
            img_aug_matrix, 
            lidar_aug_matrix, 
            batch_input_metas)
    
        bev_feat = self.pts_backbone(bev_feat)
        bev_feat = self.pts_neck(bev_feat)
        return bev_feat, img_feat

    def _forward(
        self,
        batch_inputs: Tensor,
        batch_data_samples: OptSampleList = None
    ):
        pass

    @TimeCounter(log_interval=100, warmup_interval=100, tag="FisheyeBEVDet Infer Time") 
    def predict(
        self,
        batch_inputs_dict: Dict[str, Optional[Tensor]],
        batch_data_samples: List[Det3DDataSample],
        **kwargs
    ) -> List[Det3DDataSample]:

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        bev_feat, img_feat = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(bev_feat, batch_data_samples)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)
        return res

    def loss(
        self,
        batch_inputs_dict: Dict[str, Optional[Tensor]],
        batch_data_samples: List[Det3DDataSample],
        **kwargs
    ) -> List[Det3DDataSample]:

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        bev_feat, img_feat = self.extract_feat(batch_inputs_dict, batch_input_metas)

        losses = dict()
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(bev_feat, batch_data_samples)
            losses.update(bbox_loss)

        if self.with_depth_head:
            points = batch_inputs_dict.get(self.lidar_key, None)
            depth_loss = self.depth_head.loss(img_feat, points, batch_data_samples)
            losses.update(depth_loss)

        return losses
