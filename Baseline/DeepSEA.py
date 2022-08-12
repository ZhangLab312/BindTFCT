import torch.nn as nn


class DeepSEA(nn.Module):
    def __init__(self):
        super(DeepSEA, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=160, kernel_size=8),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=160)
        )

        self.Conv2 = nn.Sequential(
            nn.Conv1d(in_channels=160, out_channels=320, kernel_size=8),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=320)
        )

        self.Conv3 = nn.Sequential(
            nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=480)
        )

        self.MaxPool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.Drop1 = nn.Dropout(p=0.2)
        self.Drop2 = nn.Dropout(p=0.5)

        self.Linear1 = nn.Sequential(
            nn.Linear(13 * 480, 925),
            nn.ReLU(inplace=True)
        )
        self.Linear2 = nn.Sequential(
            nn.Linear(925, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
            shape:
            [batch_size, 4, 101] - > [batch_size, 160, 94]
        """
        x = self.Conv1(input)
        """
            shape:
            [batch_size, 160, 94] - > [batch_size, 160, 47]
        """
        x = self.MaxPool(x)
        x = self.Drop1(x)
        """
            shape:
            [batch_size, 160, 47] - > [batch_size, 320, 40]
        """
        x = self.Conv2(x)
        """
            shape:
            [batch_size, 320, 40] - > [batch_size, 320, 20]
        """
        x = self.MaxPool(x)
        x = self.Drop1(x)
        """
            shape:
            [batch_size, 320, 20] - > [batch_size, 480, 13]
        """
        x = self.Conv3(x)
        x = self.Drop2(x)
        """
            shape:
            [batch_size, 12480]
        """
        x = x.view(-1, 13 * 480)
        """
            shape:
            [batch_size, 925]
        """
        x = self.Linear1(x)
        """
            shape:
            [batch_size, TFs_cell_line_pair]
        """
        x = self.Linear2(x)
        return x
