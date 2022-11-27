import torch
import torch.nn as nn

class simpleCNN(nn.Module):
    def __init__(self) -> None:
        super(simpleCNN, self).__init__()
        self.cnn_layer1 = nn.Conv2d(1, 8, 3, padding=2)
        self.cnn_layer2 = nn.Conv2d(8, 16, 3, padding=2)
        self.max_pool1 = nn.MaxPool2d(2)
        self.max_pool2 = nn.MaxPool2d(2)
        self.max_pool3 = nn.MaxPool2d(2)
        self.max_pool4 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(1024, 256)
        self.linear2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.cnn_layer1(x)
        x = self.max_pool1(x)
        x = self.relu(x)
        x = self.cnn_layer2(x)
        x = self.max_pool2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = nn.Softmax(dim=-1)(x)
        return x

        