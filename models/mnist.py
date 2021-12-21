#    Copyright 2021 DTAI Research Group

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

#    Minor changes have been made by David Friede


import torch
import torch.nn as nn
import numpy as np
from utils import gumbel_softmax


class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.centroids = nn.Embedding(19, 84)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True)
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 84)
        )
        self.add = nn.Sequential(
            nn.Linear(84 * 2, 84 * 2),
            nn.ReLU(),
            nn.Linear(84 * 2, 84),
            nn.ReLU()
        )
        self.classifier = nn.LogSoftmax(1)

#         self.classifier2 = nn.Sequential(
#             nn.Linear(84, 19, bias=False),
#             nn.LogSoftmax(1)
#         )

    def discretize(self, x, tau, beta, hard):
        centroids = self.centroids.weight
        logits = torch.mm(x, centroids.transpose(1, 0))
        a, _ = gumbel_softmax(logits, tau, beta, hard)
        return torch.mm(a, centroids).view(x.shape), a

    def forward(self, x1, x2, alpha, tau, beta, hard):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x1 = x1.view(-1, 16 * 4 * 4)
        x2 = x2.view(-1, 16 * 4 * 4)
        x1 = self.encoder2(x1)
        x2 = self.encoder2(x2)

        x1_d, _ = self.discretize(x1, tau, beta, hard)
        if np.random.random() > alpha:
            x1 = x1 + x1_d
        else:
            x1 = x1_d

        x2_d, _ = self.discretize(x2, tau, beta, hard)
        if np.random.random() > alpha:
            x2 = x2 + x2_d
        else:
            x2 = x2_d

        x = torch.cat([x1, x2], -1)
        x = self.add(x)
        out = torch.mm(x, self.centroids.weight.transpose(1, 0))
        return self.classifier(out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MNIST_Baseline(nn.Module):
    def __init__(self):
        super(MNIST_Baseline, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 11 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 19),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 11 * 4)
        x = self.classifier(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
