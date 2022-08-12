import torch.nn as nn


class DeepBind(nn.Module):

    def __init__(self):
        super(DeepBind, self).__init__()
        self.Convolutions = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=24),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=16)
        )

        self.GlobalMaxPool = nn.AdaptiveAvgPool1d(output_size=1)

        self.flatten = nn.Flatten(start_dim=1)

        self.FullyConnection = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.Convolutions(x)
        x = self.GlobalMaxPool(x)
        x = self.flatten(x)

        y = self.FullyConnection(x)

        return y