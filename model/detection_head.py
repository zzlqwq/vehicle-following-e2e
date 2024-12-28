import torch
from torch import nn


class DetectionHead(nn.Module):
    def __init__(self, cfg):
        super(DetectionHead, self).__init__()

        self.cfg = cfg

        self.position_head = nn.Sequential(
            nn.Linear(361, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )
        self.yaw_head = nn.Sequential(
            nn.Linear(361, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Tanh()
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(361, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # first 12 elements are position, next 12 elements are yaw, last 12 elements are velocity
        position = self.position_head(x)
        yaw = self.yaw_head(x)
        velocity = self.velocity_head(x)
        detection_result = torch.cat((position, yaw, velocity), dim=1)

        return detection_result
