from torch import nn
from blocks.attention import MultiHeadAttention
from blocks.feed_forward_network import FeedForwardNetwork


class EncoderLayer(nn.Module):
    def __init__(self, num_heads, context_length, embedding_dim, hidden_dim, dropout_prob):
        super().__init__()

        # Initialise the encoder blocks
        self.multiHeadAttention = MultiHeadAttention(num_heads, context_length, embedding_dim)
        self.feedForwardNetwork = FeedForwardNetwork(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, mask):
        # Encoder Sublayer 1: Multi-head attention
        multi_head_attention_output, attention_heads_weights = self.multiHeadAttention.forward(x, x, x, mask)

        # Apply dropout to the output before the residual connection and normalization operation
        multi_head_attention_output = self.dropout(multi_head_attention_output)

        # Apply the residual connection and layer normalisation to the multi-head
        # attention output i.e. output = LayerNorm(Sublayer(x) + x), where Sublayer(x) = MultiHead(Q, K, V)
        multi_head_attention_output_with_residual_connection = multi_head_attention_output + x
        normalised_multi_head_attention_output = self.layer_norm(multi_head_attention_output_with_residual_connection)

        # Encoder Sublayer 2: Feed forward network
        feed_forward_output = self.feedForwardNetwork.forward(normalised_multi_head_attention_output)

        # Apply dropout to the output before the residual connection and normalization operation
        feed_forward_output = self.dropout(feed_forward_output)

        # Apply the residual connection and layer normalisation to the feed forward
        # network output i.e. output = LayerNorm(Sublayer(x) + x), where Sublayer(x) = FFN(x)
        feed_forward_with_residual_connection = feed_forward_output + normalised_multi_head_attention_output
        normalised_feed_forward_output = self.layer_norm(feed_forward_with_residual_connection)

        return normalised_feed_forward_output, attention_heads_weights


class Encoder(nn.Module):
    def __init__(self, num_layers, num_heads, context_length, embedding_dim, hidden_dim, dropout_prob):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(num_heads, context_length, embedding_dim, hidden_dim, dropout_prob) for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        layers_attention_heads_weights = []

        for encoder_layer in self.layers:
            x, attention_heads_weights = encoder_layer(x, mask)

            layers_attention_heads_weights.append(attention_heads_weights)

        return x, layers_attention_heads_weights
