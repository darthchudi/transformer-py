from torch import nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        # FFN(x) = max(0, xW1 + b1)W2 + b2x
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        feed_forward_output = self.layers(x)
        return feed_forward_output
