import config
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from dataset import YOLODataset
from torch.utils.data import DataLoader
from tqdm import tqdm

class YOLOv1ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = config.B * 5 + config.C

        # Load backbone ResNet
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        backbone.requires_grad_(False)            # Freeze backbone weights

        # Delete last two layers and attach detection layers
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()

        self.model = nn.Sequential(
            backbone,
            Reshape(512, 20, 20),
            DetectionNet(512)              # 4 conv, 2 linear
        )

    def forward(self, x):
        return self.model.forward(x)

class DetectionNet(nn.Module):
    """The layers added on for detection as described in the paper."""

    def __init__(self, in_channels):
        super().__init__()

        inner_channels = 1024
        self.depth = 5 * config.B + config.C
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inner_channels),                # Batch Normalization
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.1),                           # Dropout with 30% probability

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=2, padding=1),  # (Ch, 20, 20) -> (Ch, 10, 10)
            nn.BatchNorm2d(inner_channels),                # Batch Normalization
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.1),                        # Dropout with 30% probability

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inner_channels),                # Batch Normalization
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.1),                        # Dropout with 30% probability

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inner_channels),                # Batch Normalization
            nn.LeakyReLU(negative_slope=0.1),                           # Dropout with 30% probability
            nn.Dropout(0.1),                        # Dropout with 30% probability

            nn.Flatten(),

            nn.Linear(10 * 10 * inner_channels, 4096),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.1),                        # Dropout with 30% probability



            nn.Linear(4096, config.S * config.S * self.depth)
        )

    def forward(self, x):
        return torch.reshape(
            self.model.forward(x),
            (-1, config.S, config.S, self.depth)
        )

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = tuple(args)

    def forward(self, x):
        return torch.reshape(x, (-1, *self.shape))

class Probe(nn.Module):
    names = set()

    def __init__(self, name, forward=None):
        super().__init__()

        assert name not in self.names, f"Probe named '{name}' already exists"
        self.name = name
        self.names.add(name)
        self.forward = self.probe_func_factory(probe_size if forward is None else forward)

    def probe_func_factory(self, func):
        def f(x):
            print(f"\nProbe '{self.name}':")
            func(x)
            return x
        return f

def probe_size(x):
    print(x.size())

def probe_mean(x):
    print(torch.mean(x).item())

def probe_dist(x):
    print(torch.min(x).item(), '|', torch.median(x).item(), '|', torch.max(x).item())
