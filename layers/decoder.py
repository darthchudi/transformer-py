from torch import nn
from blocks.attention import MultiHeadAttention
from blocks.feed_forward_network import FeedForwardNetwork


class DecoderLayer(nn.Module):
    def __init__(self, num_heads, context_length, embedding_dim, hidden_dim, dropout_prob):
        super().__init__()

        # Initialise the decoder blocks
        self.maskedMultiHeadAttention = MultiHeadAttention(num_heads, context_length, embedding_dim)
        self.crossMultiHeadAttention = MultiHeadAttention(num_heads, context_length, embedding_dim)
        self.feedForwardNetwork = FeedForwardNetwork(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, encoder_output, target, source_mask, target_mask):
        # Decoder Sublayer 1: Masked multi-head attention
        masked_multi_head_attention_output, decoder_masked_attention_heads_weights = self.maskedMultiHeadAttention.forward(
            q=target,
            k=target,
            v=target,
            mask=target_mask)

        # Apply dropout to the output before the residual connection and normalization operation
        masked_multi_head_attention_output = self.dropout(masked_multi_head_attention_output)

        # Apply the residual connection and layer normalisation to the multi-head
        # attention output i.e. output = LayerNorm(Sublayer(x) + x), where Sublayer(x) = MaskedMultiHead(Q, K, V)
        masked_multi_head_attention_output_with_residual_connection = masked_multi_head_attention_output + target
        normalised_masked_multi_head_attention_output = self.layer_norm(
            masked_multi_head_attention_output_with_residual_connection)

        # Decoder Sublayer 2: Cross multi-head attention
        cross_multi_head_attention_output, cross_attention_heads_weights = self.crossMultiHeadAttention.forward(
            q=normalised_masked_multi_head_attention_output, k=encoder_output, v=encoder_output, mask=source_mask)

        # Apply dropout to the output before the residual connection and normalization operation
        cross_multi_head_attention_output = self.dropout(cross_multi_head_attention_output)

        # Apply the residual connection and layer normalisation to the multi-head
        # attention output i.e. output = LayerNorm(Sublayer(x) + x), where Sublayer(x) = MultiHead(Q, K, V)
        cross_multi_head_attention_output_with_residual_connection = cross_multi_head_attention_output + normalised_masked_multi_head_attention_output
        normalised_cross_multi_head_attention_output_with_residual_connection = self.layer_norm(
            cross_multi_head_attention_output_with_residual_connection)

        # Decoder Sublayer 3: Feed forward network
        feed_forward_output = self.feedForwardNetwork.forward(
            normalised_cross_multi_head_attention_output_with_residual_connection)

        # Apply dropout to the output before the residual connection and normalization operation
        feed_forward_output = self.dropout(feed_forward_output)

        # Apply the residual connection and layer normalisation to the feed forward
        # network output i.e. output = LayerNorm(Sublayer(x) + x), where Sublayer(x) = FFN(x)
        feed_forward_with_residual_connection = feed_forward_output + normalised_cross_multi_head_attention_output_with_residual_connection
        normalised_feed_forward_output = self.layer_norm(feed_forward_with_residual_connection)

        return normalised_feed_forward_output, decoder_masked_attention_heads_weights, cross_attention_heads_weights


class Decoder(nn.Module):
    def __init__(self, num_layers, num_heads, context_length, embedding_dim, hidden_dim, dropout_prob):
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(num_heads, context_length, embedding_dim, hidden_dim, dropout_prob) for _ in range(num_layers)
        ])

    def forward(self, encoder_output, target, source_mask, target_mask):
        layers_decoder_masked_attention_heads_weights = []
        layers_cross_attention_heads_weights = []

        for decoder_layer in self.layers:
            target, decoder_masked_attention_heads_weights, cross_attention_heads_weights = decoder_layer(
                encoder_output, target, source_mask, target_mask)

            layers_decoder_masked_attention_heads_weights.append(decoder_masked_attention_heads_weights)
            layers_cross_attention_heads_weights.append(cross_attention_heads_weights)

        return target, layers_decoder_masked_attention_heads_weights, layers_cross_attention_heads_weights
