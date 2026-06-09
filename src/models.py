## Enter the Deep neural network architectures here
import torch.nn.functional as F
import torch
import torch.nn as nn
from pathlib import Path


# Network for Multimnist Daza
class MultiTaskNet56(nn.Module):
    """
    Multitask CNN for 56x56 grayscale images (merged 3-MNIST).
    - Shared layer -> feature vector h
    - Three task-specific linear heads -> 10-class logits each # 317,748 parameters
    """
    def __init__(self, num_tasks=3, num_classes=10, in_channels=1, feat_dim=128):
        super().__init__()
        # Shared CNN backbone for 56x56 -> 1x1 via AdaptiveAvgPool
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=False),
            nn.Flatten(),
            nn.Linear(2420, feat_dim),
            nn.ReLU(inplace=False),
        )
        # Projection layer (just identity mapping here, keeps naming consistent)
        self.proj = nn.Identity()

        # Task-specific heads -> `self.heads` like MultiTaskNet56old
        self.heads = nn.ModuleList([nn.Linear(feat_dim, num_classes) for _ in range(num_tasks)])

    def forward(self, x, return_features=False):
        """
        Forward pass.
        Returns a list of per-task outputs (log_softmax applied).
        """
        h = self.layer(x)          # shared block
        h = self.proj(h)           # proj layer (identity, keeps naming consistent)

        outs = [F.log_softmax(head(h), dim=1) for head in self.heads]

        if return_features:
            return outs, h
        return outs

    # Convenience: save/load
    def save_model(self, path):
        path = Path(path)
        if path.suffix != ".pth":
            path = path.with_suffix(".pth")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_model(self, path, map_location=None):
        path = Path(path)
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)
        return self.eval()  # set eval by default; call .train() if continuing training
    




class MLP(nn.Module):
    """
    # Network for UCI data set 3 tasks
    """
    def __init__(self, **kwargs):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(487, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_age = nn.Linear(128, 2)
        self.fc_education = nn.Linear(128, 2)
        self.fc_marriage = nn.Linear(128, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return [self.fc_age(x), self.fc_education(x), self.fc_marriage(x)]
    
    # Convenience: save/load
    def save_model(self, path):
        path = Path(path)
        if path.suffix != ".pth":
            path = path.with_suffix(".pth")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_model(self, path, map_location=None):
        path = Path(path)
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)
        return self.eval()  # set eval by default; call .train() if continuing tr
    


class MLP_uci_3plus(nn.Module):
    """
    # Network for UCI data set >3 tasks (age, education, marriage, race, sex)
    """
    def __init__(self, **kwargs):
        super(MLP_uci_3plus, self).__init__()
        self.fc1 = nn.Linear(480, 256)
        self.fc2 = nn.Linear(256, 128)

        self.fc_age = nn.Linear(128, 2)
        self.fc_education = nn.Linear(128, 2)
        self.fc_marriage = nn.Linear(128, 2)            
        self.fc_race = nn.Linear(128, 5)
        self.fc_sex = nn.Linear(128, 2)
        self.relu = nn.ReLU(inplace=True)   
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return [self.fc_age(x), self.fc_education(x), self.fc_marriage(x), self.fc_race(x), self.fc_sex(x)]
    
    # Convenience: save/load# Convenience: save/load
    def save_model(self, path):
        path = Path(path)
        if path.suffix != ".pth":
            path = path.with_suffix(".pth")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_model(self, path, map_location=None):
        path = Path(path)
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)
        return self.eval()  # set eval by default; call .train() if continuing tr


