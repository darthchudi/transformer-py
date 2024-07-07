import torch
from torch import nn
import math


class PositionalEncodingLayer(nn.Module):
    def __init__(self, context_length, embedding_dim):
        super().__init__()

        positional_encodings = torch.zeros(context_length, embedding_dim)

        for position in range(context_length):
            for i in range(embedding_dim // 2):
                denominator = 10_000 ** (2 * i / embedding_dim)
                positional_encodings[position, 2 * i] = math.sin(position / denominator)
                positional_encodings[position, 2 * i + 1] = math.cos(position / denominator)

        # Register the positional encodings as a buffer, so it's
        # not considered a trainable parameter
        self.register_buffer('pe', positional_encodings)

    def forward(self, x):
        context_length = x.size(1)
        return self.pe[:context_length] + x
