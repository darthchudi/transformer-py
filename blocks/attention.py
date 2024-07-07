from torch import nn
import torch


class Attention(nn.Module):
    def __init__(self, context_length, embedding_dim, num_heads):
        super().__init__()

        dk = embedding_dim // num_heads
        dv = embedding_dim // num_heads

        # The query projection w_q has the same dimensionality dk as the key projection
        self.w_q = nn.Linear(embedding_dim, dk)
        self.w_k = nn.Linear(embedding_dim, dk)
        self.w_v = nn.Linear(embedding_dim, dv)

        self.softmax = nn.Softmax(dim=2)
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.dk = dk

    # where x = (batch_size, context_length, embedding_dim)
    # return a self-attention tensor (batch_size, context_length, dv)
    def forward(self, k, q, v, mask=None):
        K = self.w_k(k)
        Q = self.w_q(q)
        V = self.w_v(v)

        # Compute the compatibility scores with a dot product operation
        # on the queries and keys transformation
        dot_product = torch.matmul(Q, K.transpose(1, 2))

        # Scale the dot product value based on the embedding dimension of the key vector
        # todo: fix bug here so we divide by dk
        scaled_dot_product = dot_product / torch.sqrt(torch.tensor(self.dk))

        # Apply the mask, so that masked positions get zeroed after softmax
        if mask is not None:
            # Use -1e9 as a replacement to negative infinity (-inf) for the mask value, because it
            # leads to NaN results after softmax for masked positions
            scaled_dot_product = scaled_dot_product.masked_fill(mask == 0, -1e9)

        # Perform a softmax operation on the scaled dot product value to get the attention weights
        # as (batch_size, context_length, context_length) shaped tensor.
        attention_weights = self.softmax(scaled_dot_product)

        # Apply the similarity scores on the value tensor
        attention_scores = torch.matmul(attention_weights, V)

        return attention_scores, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, context_length, embedding_dim):
        super().__init__()

        # Create the linear layer for the multi-head attention
        dv = embedding_dim // num_heads
        self.linear = nn.Linear(num_heads * dv, embedding_dim)

        # Initialise the attention heads
        self.heads = nn.ModuleList([Attention(context_length, embedding_dim, num_heads) for head in range(num_heads)])

    def forward(self, k, q, v, mask=None):
        attention_heads = []
        attention_heads_weights = []

        for head in self.heads:
            attention_scores, attention_weights = head.forward(k, q, v, mask)
            attention_heads.append(attention_scores)
            attention_heads_weights.append(attention_weights)

        # Concatenate the attention heads
        # Where each head = (batch_size, context_length, dv)
        # Concatenated = (batch_size, context_length, dv*num_heads)
        # Where dv*num_heads = embedding_dim
        concatenated_attention_heads = torch.cat(attention_heads, dim=2)

        # Apply the linear projection to the concatenated attention heads
        # to get the final multi-head attention tensor
        # MultiHead(Q, K, V ) = Concat(head1, ..., headh)W^O
        multi_head_attention = self.linear(concatenated_attention_heads)

        return multi_head_attention, attention_heads_weights
