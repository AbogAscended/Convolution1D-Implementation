import torch.nn as nn
import torch.nn.init as init


class Convo(nn.Module):
    """
    Basic convo model using a sequential container, 3 hidden layers with kernel size 3 stride 2 and 15 channels each.
    Each using ReLU activation After the last conv layer it's flattened for the linear layer which contains 10 units for
    classification of base 10 which is then fed into softmax to work as probability function.
    """
    def __init__(self):
        super(Convo, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=15, kernel_size=3, stride=2, padding="valid"),
            nn.ReLU(),
            nn.Conv1d(in_channels=15, out_channels=15, kernel_size=3, stride=2, padding="valid"),
            nn.ReLU(),
            nn.Conv1d(in_channels=15, out_channels=15, kernel_size=3, stride=2, padding="valid"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=60, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.sequential(x)

    def weights_init(self):
        for layer in self.sequential:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                layer.bias.data.fill_(0.0)
