import torch.nn as nn
import torch


class Cnn_TF(nn.Module):
    def __init__(self):
        super(Cnn_TF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=100,
                kernel_size=10
            ),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=100),
            nn.AdaptiveAvgPool1d(output_size=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=8,
                out_channels=100,
                kernel_size=10
            ),
            nn.ReLU(True),
            nn.AdaptiveAvgPool1d(output_size=1)
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.fully_connection = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(True),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, s, h):
        s = self.conv1(s)
        h = self.conv2(h)

        x = torch.cat((self.flatten(s), self.flatten(h)), dim=1)

        y = self.fully_connection(x)

        return y


seq = torch.ones(size=(1, 4, 101))
h = torch.ones(size=(1, 8, 20))

Cnn_TF_class = Cnn_TF()
Cnn_TF_class(seq, h)