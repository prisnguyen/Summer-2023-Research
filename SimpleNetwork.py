import torch
import torch.nn as nn

# Create a neural network
class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out

# Test with a random input
model = SimpleNetwork()
input_data = torch.randn(1, 10)
output = model(input_data)
print("Output shape:", output.shape)
print("Output values:", output)