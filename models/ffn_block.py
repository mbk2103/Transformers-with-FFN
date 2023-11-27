import torch.nn as nn
import torch.nn.functional as F

class FFNBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.1):
        super(FFNBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
