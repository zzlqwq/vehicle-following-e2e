import torch

from torch import nn


class GRUTrajectoryPredict(nn.Module):
    def __init__(self, cfg):
        super(GRUTrajectoryPredict, self).__init__()

        self.cfg = cfg
        self.pred_len = self.cfg['pred_len']

        # waypoints prediction
        self.join = nn.Sequential(
            nn.Linear(361, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.GRUCell(input_size=2, hidden_size=64)
        self.output = nn.Linear(64, 2)

    def forward(self, z):
        bs = z.shape[0]

        z = self.join(z)

        output_wp = list()

        # initial input variable to GRU
        x = torch.zeros(size=(bs, 2), dtype=z.dtype).cuda()

        # autoregressive generation of output waypoints
        for _ in range(self.pred_len):
            x_in = x

            z = self.decoder(x_in, z)
            dx = self.output(z)

            x = dx[:, :2] + x

            output_wp.append(x[:, :2])

        pred_wp = torch.stack(output_wp, dim=1)

        return pred_wp
