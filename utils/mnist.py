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
from torch.utils.data import Dataset
import numpy as np


class MNIST_Addition(Dataset):
    def __init__(self, dataset, examples):
        self.data = list()
        self.dataset = dataset
        with open(examples) as f:
            for line in f:
                line = line.split('(')[1]
                line = line.split(')')[0]
                line = line.split(',')
                self.data.append(tuple([int(i) for i in line]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        i1, i2, lab = self.data[index]
        return self.dataset[i1][0], self.dataset[i2][0], lab


def test_MNIST(net, tau, test_loader, N, device):
    confusion = np.zeros((19, 19), dtype=np.uint32)
    correct = 0
    n = 0
    for data1, data2, label in test_loader:
        data1, data2 = data1.to(device), data2.to(device)
        outputs = net.forward(data1, data2, tau=tau, beta=1,
                              alpha=1, hard=True)
        for i, output in enumerate(outputs.cpu()):
            lab = label[i]
            _, out = torch.max(output.data, -1)
            c = int(out.squeeze())
            confusion[lab, c] += 1
            if c == lab:
                correct += 1
            n += 1
    acc = correct / n
    F1 = 0
    for nr in range(19):
        TP = confusion[nr, nr]
        FP = sum(confusion[:, nr]) - TP
        FN = sum(confusion[nr, :]) - TP
        F1 += 2 * TP / (2 * TP + FP + FN) * (FN + TP) / N
    return F1, acc
