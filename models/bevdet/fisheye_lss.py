from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mmdet3d.registry import MODELS
from .ops import bev_pool


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])  # 步长
    bx = torch.Tensor(
        [row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])  # 起始点
    nx = torch.LongTensor([int((row[1] - row[0]) / row[2])
                           for row in [xbound, ybound, zbound]])  # 网格个数
    return dx, bx, nx


class BaseViewTransform(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        # self.dx = nn.Parameter(dx, requires_grad=False)
        # self.bx = nn.Parameter(bx, requires_grad=False)
        # self.nx = nn.Parameter(nx, requires_grad=False)
        self.register_buffer('dx', dx)
        self.register_buffer('bx', bx)
        self.register_buffer('nx', nx)

        self.C = out_channels
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (
            torch.arange(*self.dbound,
                         dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW))
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW,
                           dtype=torch.float).view(1, 1, fW).expand(D, fH, fW))
        ys = (
            torch.linspace(0, iH - 1, fH,
                           dtype=torch.float).view(1, fH, 1).expand(D, fH, fW))

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        **kwargs,
    ):
        B, N, _ = camera2lidar_trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots).view(B, N, 1, 1, 1, 3,
                                          3).matmul(points.unsqueeze(-1)))
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        if 'extra_rots' in kwargs:
            extra_rots = kwargs['extra_rots']
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3,
                                3).repeat(1, N, 1, 1, 1, 1, 1).matmul(
                                    points.unsqueeze(-1)).squeeze(-1))
        if 'extra_trans' in kwargs:
            extra_trans = kwargs['extra_trans']
            points += extra_trans.view(B, 1, 1, 1, 1,
                                       3).repeat(1, N, 1, 1, 1, 1)

        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        # self.bx - self.dx / 2.0 is the minimum boundary value of the grid
        # (geom_feats - (self.bx - self.dx / 2.0)) shifts geom_feats to the grid coordinate system
        # (...) / self.dx normalizes to grid coordinates
        # That is, the original point (x, y, z) in space coordinates is quantized to integer voxel index coordinates
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) /
                      self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([
            torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
            for ix in range(B)
        ])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = ((geom_feats[:, 0] >= 0)
                & (geom_feats[:, 0] < self.nx[0])
                & (geom_feats[:, 1] >= 0)
                & (geom_feats[:, 1] < self.nx[1])
                & (geom_feats[:, 2] >= 0)
                & (geom_feats[:, 2] < self.nx[2]))
        x = x[kept]
        geom_feats = geom_feats[kept]

        x = bev_pool(x, geom_feats, B,
                     self.nx[2], self.nx[0], self.nx[1], mean_pool=True)

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)  # Unbind x along dimension 2, then concatenate along dimension 1

        return final

    def forward(
        self,
        img,
        points,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x


@MODELS.register_module()
class LSSTransform(BaseViewTransform):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.depthnet = nn.Conv2d(in_channels, self.D + self.C, 1)
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_cam_feats(self, x):
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depthnet(x)
        depth = x[:, :self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x


@MODELS.register_module()
class BaseFisheyeLSSTransform(BaseViewTransform):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        azimuth_range: Tuple[float, float],
        elevation_range: Tuple[float, float],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
        ocam_path: str = 'data/CarlaCollection/calib_results.txt',
        ocam_fov: float = 220,
    ) -> None:

        self.azimuth_range = azimuth_range
        self.elevation_range = elevation_range

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )

        from ocamcamera import OcamCamera
        self.omni_ocam = OcamCamera(filename=ocam_path, fov=ocam_fov)
        self.sphere_grid_3d = self.get_sphere_grid()  # [D, H, W, 3]
        self.register_buffer('valid_fov_mask', self.get_valid_mask())
        
        self.down_ratio = downsample
        self.init_layer()
    
    def init_layer(self):
        self.proj_x = nn.Conv2d(self.in_channels, self.C, 1)
        if self.down_ratio > 1:
            assert self.down_ratio == 2, "only support downsample=2"
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    self.C, self.C, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.C),
                nn.ReLU(True),
                nn.Conv2d(
                    self.C,
                    self.C,
                    3,
                    stride=self.down_ratio,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.C),
                nn.ReLU(True),
                nn.Conv2d(
                    self.C, self.C, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.C),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def create_frustum(self):
        fH, fW = self.feature_size
        ds = torch.arange(
            *self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)

        D, _, _ = ds.shape
  
        min_theta, max_theta = self.azimuth_range
        min_phi, max_phi = self.elevation_range
        azimuths = torch.linspace(min_theta, max_theta, fW,
                                  dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        elevations = torch.linspace(min_phi, max_phi, fH,
                                    dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        xs = ds * torch.cos(elevations) * torch.sin(azimuths)
        ys = ds * torch.sin(elevations)
        zs = ds * torch.cos(elevations) * torch.cos(azimuths)
        frustum = torch.stack((xs, ys, zs), -1)

        return nn.Parameter(frustum, requires_grad=False)

    def get_sphere_grid(self):
        proj_pts = self.frustum.cpu().numpy()
        sphere_grid = []
        for d in range(proj_pts.shape[0]):
            mapx, mapy = self.omni_ocam.world2cam(
                proj_pts[d, ...].reshape(-1, 3).T)
            mapx, mapy = mapx.reshape(
                self.feature_size), mapy.reshape(self.feature_size)
            mapx, mapy = mapx * 2 / self.omni_ocam.width - \
                1, mapy * 2 / self.omni_ocam.height - 1
            grid = torch.from_numpy(np.stack([mapx, mapy], axis=-1))
            sphere_grid.append(grid)
        sphere_grid = torch.stack(sphere_grid, dim=0)
        depth_coords = torch.zeros(
            size=(self.D, self.feature_size[0], self.feature_size[1], 1))
        sphere_grid = torch.cat(
            (sphere_grid, depth_coords), dim=-1)  # [D, H, W, 3]
        return nn.Parameter(sphere_grid, requires_grad=False)

    def get_valid_mask(self):
        valid_fov_mask = self.omni_ocam.valid_area()
        valid_fov_mask = torch.from_numpy(valid_fov_mask).unsqueeze(
            0).unsqueeze(0).float() / 255.0
        valid_fov_mask = F.grid_sample(
            valid_fov_mask, self.sphere_grid_3d[0, :, :, :2].unsqueeze(0), align_corners=True)
        return valid_fov_mask

    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        **kwargs,
    ):
        B, N, _ = camera2lidar_trans.shape
        points = self.frustum.unsqueeze(
            0).unsqueeze(0)  # 1 x 1 x D x H x W x 3
        points = torch.matmul(camera2lidar_rots.view(
            B, N, 1, 1, 1, 3, 3), points.unsqueeze(-1)).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)
        return points

@MODELS.register_module()
class FisheyeLSSTransform(BaseFisheyeLSSTransform):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        azimuth_range: Tuple[float, float],
        elevation_range: Tuple[float, float],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
        ocam_path: str = 'data/Fisheye3D/calib_results.txt',
        ocam_fov: float = 220,
    ) -> None:
        self.azimuth_range = azimuth_range
        self.elevation_range = elevation_range
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            azimuth_range=azimuth_range,
            elevation_range=elevation_range,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
            downsample=downsample,
            ocam_path=ocam_path,
            ocam_fov=ocam_fov,
        )

        self.register_buffer('sphere_grid_2d', self.sphere_grid_3d[0, :, :, :2])  # [H, W, 2]
        self.init_layer()
    
    def init_layer(self):
        self.depthnet = nn.Conv2d(self.in_channels, self.D + self.C, 1)
        if self.down_ratio > 1:
            assert self.down_ratio == 2, "self.down_ratio should be 2"
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    self.C, self.C, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.C),
                nn.ReLU(True),
                nn.Conv2d(
                    self.C,
                    self.C,
                    3,
                    stride=self.down_ratio,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.C),
                nn.ReLU(True),
                nn.Conv2d(
                    self.C, self.C, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.C),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()
    
    def get_cam_feats(self, x):
        B, N, C, fH, fW = x.shape
        x = x.view(B * N, C, fH, fW)
        warp_x = F.grid_sample(
            x, self.sphere_grid_2d.unsqueeze(0).expand(B * N, -1, -1, -1), align_corners=True)  # BN x C x H x W
        x = self.depthnet(warp_x)

        depth = x[:, :self.D].softmax(dim=1)

        x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        fH, fW = self.sphere_grid_2d.shape[:2]
        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x
    

    def forward(
        self,
        img_feat,
        points,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        **kwargs
    ):
        # import matplotlib.pyplot as plt

        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        geom = self.get_geometry(camera2lidar_rots, camera2lidar_trans)
        img_feat = self.get_cam_feats(img_feat)  # B x N x D x H x W x C
        bev_feat = self.bev_pool(geom, img_feat)
        bev_feat = self.downsample(bev_feat)
        return bev_feat, img_feat
    

@MODELS.register_module()
class FisheyeCylindricalLSSTransform(FisheyeLSSTransform):
    """Fisheye projection transform with cylindrical projection, similar to FisheyeLSSTransformV2."""
    
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

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
    
    
@MODELS.register_module()
class CylindricalLSSTransform(LSSTransform):
    """Cylindrical projection transform, similar to FisheyeLSSTransform."""
    
    def __init__(
        self,
        *args,
        horizontal_range: tuple = (-math.radians(110), math.radians(110)),
        vertical_range: tuple = (-math.radians(45), math.radians(45)),
        **kwargs
    ) -> None:
        
        
        self.horizontal_range = horizontal_range
        self.vertical_range = vertical_range
        super().__init__(*args, **kwargs)

    def create_frustum(self):
        fH, fW = self.feature_size
        D = int((self.dbound[1] - self.dbound[0]) / self.dbound[2])
        ds = torch.linspace(self.dbound[0], self.dbound[1], D).view(D, 1, 1)

        # Convert to radians
        h_min, h_max = self.horizontal_range  # φ
        v_min, v_max = self.vertical_range    # θ

        # Cylindrical projection: y = tan(θ)
        y_min, y_max = math.tan(v_min), math.tan(v_max)

        # Create angular grid φ and y grid
        phi = torch.linspace(h_min, h_max, fW)  # (W,)
        y = torch.linspace(y_min, y_max, fH)    # (H,)
        phi_grid, y_grid = torch.meshgrid(phi, y, indexing='xy')  # shape: (H, W)

        # Expand to 3D: (D, H, W)
        phi_grid = phi_grid[None, :, :].expand(D, fH, fW)
        y_grid = y_grid[None, :, :].expand(D, fH, fW)
        ds = ds.expand(D, fH, fW)

        # Cylindrical rays * depth
        x = ds * torch.sin(phi_grid)
        y = ds * y_grid
        z = ds * torch.cos(phi_grid)
        frustum = torch.stack([x, y, z], dim=-1)  # shape: (D, H, W, 3)

        return nn.Parameter(frustum, requires_grad=False)
    
    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        **kwargs,
    ):
        B, N, _ = camera2lidar_trans.shape
        points = self.frustum.unsqueeze(
            0).unsqueeze(0)  # 1 x 1 x D x H x W x 3
        points = torch.matmul(camera2lidar_rots.view(
            B, N, 1, 1, 1, 3, 3), points.unsqueeze(-1)).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)
        return points
    
    def get_cam_feats(self, x):
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depthnet(x)
        depth = x[:, :self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x
    
    def forward(
        self,
        img_feat,
        points,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        **kwargs
    ):
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        geom = self.get_geometry(camera2lidar_rots, camera2lidar_trans)
        img_feat = self.get_cam_feats(img_feat)  # B x N x D x H x W x C
        bev_feat = self.bev_pool(geom, img_feat)
        bev_feat = self.downsample(bev_feat)
        return bev_feat, img_feat
    

if __name__ == '__main__':
    pinhole_view_transform = LSSTransform(
        in_channels=256,
        out_channels=96,
        image_size=[400, 800],
        feature_size=[25, 50],
        xbound=[-48.0, 48.0, 0.3],
        ybound=[-48.0, 48.0, 0.3],
        zbound=[-5.0, 5.0, 10.0],
        dbound=[0.5, 48.5, 0.5],
        downsample=1)

    import math
    fisheye_view_transform = FisheyeLSSTransform(
        in_channels=256,
        out_channels=96,
        image_size=[800, 800],
        feature_size=[25, 100],
        elevation_range=[-math.pi/4, math.pi/4],
        xbound=[-48.0, 48.0, 0.3],
        ybound=[-48.0, 48.0, 0.3],
        zbound=[-5.0, 5.0, 10.0],
        dbound=[0.5, 48.5, 0.5],
        downsample=1)

