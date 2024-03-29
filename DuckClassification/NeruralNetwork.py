import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedNN(nn.Module):
    def __init__(self) -> None:
        super(FullyConnectedNN, self).__init__()
        self.input_to_hidden1 = nn.Linear(3 * 224 * 224, 36)
        self.hidden1_to_hidden2 = nn.Linear(36, 36)
        self.hidden2_to_output = nn.Linear(36, 2)

        self.flatten = nn.Flatten()

    def forward(self, e):
        e = self.flatten(e)
        e = F.relu(self.input_to_hidden1(e))
        e = F.relu(self.hidden1_to_hidden2(e))
        e = self.hidden2_to_output(e)
        return e
    
