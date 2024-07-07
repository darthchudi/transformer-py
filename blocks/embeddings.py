from torch import nn


class EmbeddingLayer(nn.Module):
    # Initializes a look-up table for word embeddings
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_dim,
        )

    def forward(self, x):
        return self.embedding(x)
