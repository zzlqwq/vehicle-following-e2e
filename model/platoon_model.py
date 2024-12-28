from torch import nn

from model.monocular_bev_model import MonocularBevModel
from model.bev_encoder import MonoCularBevEncode
from model.gru_traj_predict import GRUTrajectoryPredict
from model.detection_head import DetectionHead


class PlatoonModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.bev_model = MonocularBevModel(self.cfg)

        self.bev_encoder = MonoCularBevEncode(self.cfg['bev_encoder_in_channel'] * (self.cfg['hist_frame_nums'] + 1))

        self.detection_head = DetectionHead(self.cfg)

        self.trajectory_head = GRUTrajectoryPredict(self.cfg).cuda()

    def encoder(self, data):
        image = data['image']
        prev_images = data['prev_images']
        trans = data['trans']
        extrinsic = data['extrinsic']

        bev_feature, pred_depth, semantic_mask = self.bev_model(image, prev_images, trans, extrinsic)
        encoded_bev_feature = self.bev_encoder(bev_feature)
        pred_trajectory = self.trajectory_head(encoded_bev_feature)
        detection_result = self.detection_head(encoded_bev_feature)

        return pred_trajectory, detection_result, pred_depth, semantic_mask

    def forward(self, data):
        pred_trajectory, detection_result, pred_depth, semantic_mask = self.encoder(data)
        return pred_trajectory, detection_result, pred_depth, semantic_mask
