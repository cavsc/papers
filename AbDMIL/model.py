import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self):
        super(Attention,self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1,20,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(20,50,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50*4*4,self.L),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L,self.D),
            nn.Tanh(),
            nn.Linear(self.D,self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
