import torch
import torch.nn as nn
import numpy as np
from mmdet.models.layers.transformer import inverse_sigmoid
from mmdet3d.registry import MODELS
from .fisheye_petr_head import FisheyePETRHead


@MODELS.register_module()
class FisheyeCylindricalPetrHead(FisheyePETRHead):
    """OmniPetr head for fisheye cylindrical camera."""

    def __init__(self, *args, **kwargs):
        super(FisheyeCylindricalPetrHead, self).__init__(*args, **kwargs)
        self.cam_type = kwargs.get('cam_type', 'cam_fisheye')

    def create_frustum(self):
        fH, fW = self.feature_size  # H: vertical, W: horizontal
        ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)  # D x H x W

        D, _, _ = ds.shape

        # Cylindrical angles and heights
        h_min, h_max = self.azimuth_range         # horizontal angle φ
        v_min, v_max = self.elevation_range       # vertical field-of-view, mapped to y via tan
        h_min = torch.tensor(h_min, dtype=torch.float)
        h_max = torch.tensor(h_max, dtype=torch.float)
        v_min = torch.tensor(v_min, dtype=torch.float)
        v_max = torch.tensor(v_max, dtype=torch.float)

        # 1. Azimuth angles φ from h_min to h_max
        phi = torch.linspace(h_min, h_max, fW).view(1, 1, fW).expand(D, fH, fW)  # D x H x W

        # 2. Vertical axis y from tan(v_min) to tan(v_max)
        y_min = torch.tan(v_min)
        y_max = torch.tan(v_max)
        y = torch.linspace(y_min, y_max, fH).view(1, fH, 1).expand(D, fH, fW)    # D x H x W

        # 3. Project to 3D cylindrical coordinates, radius = ds
        x = ds * torch.sin(phi)          # D x H x W
        y = ds * y                       # D x H x W (directly stretched y-axis)
        z = ds * torch.cos(phi)         # D x H x W

        frustum = torch.stack((x, y, z), dim=-1)  # D x H x W x 3

        return nn.Parameter(frustum, requires_grad=False)

    def position_embeding(self, img_feats, img_metas, masks=None):
        B, N, C, H, W = img_feats[self.position_level].shape
        if self.LID:
            index = torch.arange(
                start=0,
                end=self.depth_num,
                step=1,
                device=img_feats[0].device).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (
                self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index = torch.arange(
                start=0,
                end=self.depth_num,
                step=1,
                device=img_feats[0].device).float()
            bin_size = (self.position_range[3] -
                        self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords_d = coords_d.view(D, 1, 1).expand(-1, H, W)

        h_min, h_max = self.azimuth_range         # horizontal angle φ
        v_min, v_max = self.elevation_range       # vertical field-of-view, mapped to y via tan
        h_min = torch.tensor(h_min, dtype=torch.float)
        h_max = torch.tensor(h_max, dtype=torch.float)
        v_min = torch.tensor(v_min, dtype=torch.float)
        v_max = torch.tensor(v_max, dtype=torch.float)

        # 1. Azimuth angles φ from h_min to h_max
        phi = torch.linspace(h_min, h_max, W, dtype=torch.float, 
            device=img_feats[0].device).view(1, 1, W).expand(D, H, W)  # D x H x W
        # 2. Vertical axis y from tan(v_min) to tan(v_max)
        y_min = torch.tan(v_min)
        y_max = torch.tan(v_max)
        y = torch.linspace(y_min, y_max, H, dtype=torch.float, 
            device=img_feats[0].device).view(1, H, 1).expand(D, H, W)    # D x H x W

        # 3. Project to 3D cylindrical coordinates, radius = coords_d
        xs = coords_d * torch.sin(phi)          # D x H x W
        ys = coords_d * y                       # D x H x W (directly stretched y-axis)
        zs = coords_d * torch.cos(phi)          # D x H x W

        # Stack and permute to get [W, H, D, 4]
        coords = torch.stack([xs, ys, zs], dim=-1).permute(2, 1, 0, 3)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)

        cam2lidars = []
        for img_meta in img_metas:
            cam2lidars.append(img_meta[self.input_key]['cam2lidar'])
        cam2lidars = np.asarray(cam2lidars)
        cam2lidars = coords.new_tensor(cam2lidars)  # (B, N, 4, 4)

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        cam2lidars = cam2lidars.view(B, N, 1, 1, 1, 4,
                                     4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(cam2lidars, coords).squeeze(-1)[..., :3]

        # coords3d_numpy = coords3d.cpu().numpy()
        # np.save('debug/coords3d_sphere.npy', coords3d_numpy)

        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (
            self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (
            self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (
            self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)  
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
        coords3d = coords3d.permute(0, 1, 4, 5, 3,
                                    2).contiguous().view(B * N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)
        coords_position_embeding = coords_position_embeding.view(
            B, N, self.embed_dims, H, W)

        return coords_position_embeding, coords_mask  # coords_mask not used in PETR