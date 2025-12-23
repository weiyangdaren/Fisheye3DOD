
from flask.cli import F
import torch
import torch.nn as nn
from mmengine.structures import InstanceData

from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures.ops import bbox3d2result
from mmengine.utils.dl_utils import TimeCounter


@MODELS.register_module()
class PETR(MVXTwoStageDetector):
    def __init__(self,
                 pts_voxel_layer=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None,
                 **kwargs):
    
        if train_cfg:
            self.img_key = train_cfg.input_key.img_key
            self.lidar_key = train_cfg.input_key.lidar_key
        else:
            self.img_key = None
            self.lidar_key = None
        
        self.use_lidar2img = False if self.img_key == 'cam_fisheye' else True

        pts_bbox_head.update(input_key=self.img_key)

        super(PETR,
              self).__init__(pts_voxel_layer, pts_middle_encoder,
                             pts_fusion_layer, img_backbone, pts_backbone,
                             img_neck, pts_neck, pts_bbox_head, img_roi_head,
                             img_rpn_head, train_cfg, test_cfg, init_cfg,
                             data_preprocessor)
        
    def extract_img_feat(self, img, img_metas):
        if isinstance(img, list):
            img = torch.stack(img, dim=0)
        
        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img.squeeze_()
                else:
                    B, N, C, H, W = img.size()
                    img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped
    
    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        return img_feats
    
    def _forward(self, mode='loss', **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        raise NotImplementedError('tensor mode is yet to add')
    
    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss_by_feat(*loss_inputs)

        return losses

    def loss(self,
             inputs=None,
             data_samples=None):

        '''
            Attention: lidar2img matrix is not supported in fisheye camera, use lidar2cam and ocam_calib instead.
        '''
        img = inputs[self.img_key]
        batch_img_metas = [ds.metainfo for ds in data_samples]
        batch_gt_instances_3d = [ds.gt_instances_3d for ds in data_samples]
        gt_bboxes_3d = [gt.bboxes_3d for gt in batch_gt_instances_3d]
        gt_labels_3d = [gt.labels_3d for gt in batch_gt_instances_3d]

        if self.use_lidar2img:
            batch_img_metas = self.add_lidar2img(img, batch_img_metas)
        img_feats = self.extract_feat(img=img, img_metas=batch_img_metas)

        losses = dict()
        gt_bboxes_ignore = None
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, batch_img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses
    
    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list
    
    @TimeCounter(log_interval=100, warmup_interval=100, tag="PETR Infer Time")
    def predict(self, inputs=None, data_samples=None, mode=None, **kwargs):
        img = inputs[self.img_key]
        batch_img_metas = [ds.metainfo for ds in data_samples]
        
        for var, name in [(batch_img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if self.use_lidar2img:
            batch_img_metas = self.add_lidar2img(img, batch_img_metas)

        results_list_3d = self.simple_test(batch_img_metas, img, **kwargs)

        for i, data_sample in enumerate(data_samples):
            results_list_3d_i = InstanceData(
                metainfo=results_list_3d[i]['pts_bbox'])
            data_sample.pred_instances_3d = results_list_3d_i
            data_sample.pred_instances = InstanceData()

        return data_samples
    
    # may need speed-up
    def add_lidar2img(self, img, batch_input_metas):
        """add 'lidar2img' transformation matrix into batch_input_metas.

        Args:
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.
        Returns:
            batch_input_metas (list[dict]): Meta info with lidar2img added
        """
        for meta in batch_input_metas:
            lidar2img_rts = []
            # obtain lidar to image transformation matrix
            cam2img_matrix = meta[self.img_key]['cam2img']
            lidar2cam_matrix = meta[self.img_key]['lidar2cam']
            for i in range(len(cam2img_matrix)):
                lidar2cam_rt = torch.tensor(lidar2cam_matrix[i]).double()
                intrinsic = torch.tensor(cam2img_matrix[i]).double()
                viewpad = torch.eye(4).double()
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt)
                # The extrinsics mean the transformation from lidar to camera.
                # If anyone want to use the extrinsics as sensor to lidar,
                # please use np.linalg.inv(lidar2cam_rt.T)
                # and modify the ResizeCropFlipImage
                # and LoadMultiViewImageFromMultiSweepsFiles.
                lidar2img_rts.append(lidar2img_rt)

            meta[self.img_key]['lidar2img'] = lidar2img_rts
            img_shape = meta[self.img_key]['img_shape'][:3]
            meta[self.img_key]['img_shape'] = [img_shape] * len(img[0])
        return batch_input_metas