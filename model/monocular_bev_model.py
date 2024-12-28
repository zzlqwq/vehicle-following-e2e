import torch

from torch import nn
from model.monocular_cam_encoder import CamEncode
from tool.geometry import VoxelsSumming, calculate_birds_eye_view_parameters
from tool.image_utils import CamModel
from tool.image_utils import ImageTransform


class MonocularBevModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        dx, bx, nx = calculate_birds_eye_view_parameters(self.cfg['bev_x_bound'],
                                                         self.cfg['bev_y_bound'],
                                                         self.cfg['bev_z_bound'])
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.down_sample = self.cfg['bev_down_sample']

        self.frustum = self.create_frustum()
        self.depth_channel, _, _, _ = self.frustum.shape
        self.cam_encoder = CamEncode(self.depth_channel, cfg)

        self.cam_model = CamModel(self.cfg['calib_file'])
        self.image_transform = ImageTransform(self.cam_model)

    def create_frustum(self):
        h, w = self.cfg['final_dim']
        down_sample_h, down_sample_w = h // self.down_sample, w // self.down_sample

        depth_grid = torch.arange(*self.cfg['d_bound'], dtype=torch.float)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, down_sample_h, down_sample_w)
        depth_slice = depth_grid.shape[0]

        x_grid = torch.linspace(0, w - 1, down_sample_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, down_sample_w).expand(depth_slice, down_sample_h, down_sample_w)
        y_grid = torch.linspace(0, h - 1, down_sample_h, dtype=torch.float)
        y_grid = y_grid.view(1, down_sample_h, 1).expand(depth_slice, down_sample_h, down_sample_w)

        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)

        return frustum.cuda()

    def get_geometry(self, extrinsic):
        extrinsic = extrinsic.cuda()
        rotation, translation = extrinsic[..., :3, :3], extrinsic[..., :3, 3]
        batch_size = extrinsic.shape[0]

        # points_uv_depth[i, j, k, : ] means i depth, j width, k height pixel
        points_uv_depth = self.frustum.unsqueeze(0).clone()

        # change uv to original resized uv
        points_uv_depth[..., :2] = points_uv_depth[..., :2] * self.cfg['img_down_sample']

        # transform to camera coordinate
        points_camera = self.image_transform.pixel2cam(points_uv_depth)

        # transform to base_link coordinate
        points_base = torch.matmul(rotation[0], points_camera.t()).t() + translation[0]

        # reshape to points_uv_depth
        points_base = points_base.view(*points_uv_depth.shape)

        # add batch size
        points_base = points_base.unsqueeze(0).expand(batch_size, *points_base.shape)

        return points_base

    def encoder_forward(self, images):
        b, n, c, h, w = images.shape
        images = images.view(b * n, c, h, w)
        x, depth, semantic_mask = self.cam_encoder(images)

        x = x.view(b, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x, depth, semantic_mask

    def voxel_pooling(self, geom_feats, x, semantic_mask, pred_depth):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.reshape(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        # filter out points background by semantic mask
        new_semantic_mask = (semantic_mask[:, 1] > 0.5).unsqueeze(1).unsqueeze(1).expand(-1, -1, D, -1, -1).reshape(-1)
        # filter out points by pred_depth
        # pred_depth_mask = (pred_depth > 0.4).unsqueeze(1).reshape(-1)

        kept = kept & new_semantic_mask

        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        x, geom_feats = VoxelsSumming.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)

        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def calc_bev_feature(self, images, extrinsic):
        geom = self.get_geometry(extrinsic)
        x, pred_depth, semantic_mask = self.encoder_forward(images)
        bev_feature = self.voxel_pooling(geom, x, semantic_mask, pred_depth)

        return bev_feature, pred_depth, semantic_mask

    def forward(self, images, prev_images, trans, extrinsic):
        bev_feature, pred_depth, semantic_mask = self.calc_bev_feature(images, extrinsic)
        return bev_feature, pred_depth, semantic_mask
