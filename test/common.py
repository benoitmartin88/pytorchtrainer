import torch
import torch.nn as nn


class XorDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.xor = [(0, 0, 0),  # x1, x2, y
                    (0, 1, 1),
                    (1, 0, 1),
                    (1, 1, 0)
                    ]

    def __len__(self):
        return len(self.xor)

    def __getitem__(self, idx):
        x1, x2, y = self.xor[idx]
        x = torch.tensor([x1, x2], dtype=torch.float32)
        y = torch.tensor([y], dtype=torch.float32)
        return x, y


class XorModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4, bias=True)
        self.fc2 = nn.Linear(4, 1, bias=True)

    def forward(self, x):
        out = torch.nn.ReLU()(self.fc1(x))
        out = self.fc2(out)
        return out


class XorMultiOutputModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4, bias=True)
        self.fc2 = nn.Linear(4, 4, bias=True)
        self.fc3a = nn.Linear(4, 4, bias=True)
        self.fc3b = nn.Linear(4, 1, bias=True)
        self.fc4 = nn.Linear(4, 1, bias=True)

        self.activation = torch.nn.ReLU()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        out1 = self.activation(self.fc3a(out))
        out1 = self.fc4(out1)
        out2 = self.fc3b(out)
        return out1, out2
