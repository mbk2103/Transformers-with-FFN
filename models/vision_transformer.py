import torch.nn as nn
from .ffn_block import FFNBlock

class VisionTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, num_blocks, hidden_dim, ffn_hidden_dim, dropout_prob=0.1):
        super(VisionTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.ffn_blocks = nn.ModuleList([FFNBlock(hidden_dim, ffn_hidden_dim, dropout_prob) for _ in range(num_blocks)])
        self.fc = nn.Linear(hidden_dim, num_classes)


    def forward(self, x):
        x = self.embedding(x.view(x.size(0), -1)[:,:784])  # Flatten the input tensor and take the first 784 elements
        for ffn_block in self.ffn_blocks:
            x = ffn_block(x)
        x = self.fc(x)
        return x
