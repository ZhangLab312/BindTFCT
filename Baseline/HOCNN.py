import torch.nn as nn
import torch


class Hocnn(nn.Module):

    def __init__(self, order):
        super(Hocnn, self).__init__()

        self.conv_branch_1 = nn.Conv1d(in_channels=4**order, out_channels=40, kernel_size=24)
        self.bn_branch_1 = nn.BatchNorm1d(num_features=40)

        self.conv_branch_2 = nn.Conv1d(in_channels=4**order, out_channels=40, kernel_size=12)
        self.bn_branch_2 = nn.BatchNorm1d(num_features=40)

        self.conv_branch_3 = nn.Conv1d(in_channels=4**order, out_channels=48, kernel_size=8)
        self.bn_branch_3 = nn.BatchNorm1d(num_features=48)

        self.relu = nn.ReLU(inplace=True)

        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.flatten = nn.Flatten(start_dim=1)

        self.FullyConnection = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_branch_1 = self.bn_branch_1(self.relu(self.conv_branch_1(x)))
        x_branch_2 = self.bn_branch_2(self.relu(self.conv_branch_2(x)))
        x_branch_3 = self.bn_branch_3(self.relu(self.conv_branch_3(x)))

        p_branch_1 = self.pool(x_branch_1)
        p_branch_2 = self.pool(x_branch_2)
        p_branch_3 = self.pool(x_branch_3)

        x_conv = torch.cat((p_branch_1, p_branch_2, p_branch_3), dim=1)
        x_conv = self.flatten(x_conv)

        y = self.FullyConnection(x_conv)

        return y