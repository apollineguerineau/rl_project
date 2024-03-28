"""
MLP to map states to actions

goal : go from observations (~states) to a score for each action

input_dim : length of observations
output_dim : number of possible actions
"""

import torch.nn as nn

class Q_network(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim=64) -> None:
        super(Q_network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        out = self.layers(x)
        return out