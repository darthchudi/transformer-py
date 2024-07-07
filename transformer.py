from torch import nn
from layers.encoder import Encoder
from layers.decoder import Decoder
from blocks.embeddings import EmbeddingLayer
from blocks.positional_encodings import PositionalEncodingLayer


class Transformer(nn.Module):
    def __init__(self,
                 source_vocabulary, target_vocabulary,
                 embedding_dim, context_length, num_heads, hidden_dim,
                 visualiser, device, num_layers, dropout_prob, temperature):
        super().__init__()

        # Store the visualiser
        self.visualiser = visualiser

        # Store the vocabulary
        self.source_vocabulary = source_vocabulary
        self.target_vocabulary = target_vocabulary

        # Initialise the transformer blocks
        self.inputEmbeddingLayer = EmbeddingLayer(num_embeddings=len(source_vocabulary), embedding_dim=embedding_dim)
        self.positionalEncodingLayer = PositionalEncodingLayer(context_length, embedding_dim)
        self.encoder = Encoder(num_heads=num_heads,
                               context_length=context_length,
                               embedding_dim=embedding_dim,
                               hidden_dim=hidden_dim, num_layers=num_layers, dropout_prob=dropout_prob)

        self.outputEmbeddingLayer = EmbeddingLayer(num_embeddings=len(target_vocabulary), embedding_dim=embedding_dim)
        self.decoder = Decoder(num_heads=num_heads,
                               context_length=context_length,
                               embedding_dim=embedding_dim,
                               hidden_dim=hidden_dim, num_layers=num_layers, dropout_prob=dropout_prob)

        self.linear = nn.Linear(embedding_dim, len(target_vocabulary))
        self.softmax = nn.Softmax(dim=-1)  # (Batch Size, Context length, Vocabulary size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.temperature = temperature

        self.device = device

    def encode(self, source, source_mask):
        source_embeddings = self.inputEmbeddingLayer.forward(source)
        source_embeddings = self.positionalEncodingLayer.forward(source_embeddings)

        # Apply dropout to the sums of the embeddings and positional encoding
        source_embeddings = self.dropout(source_embeddings)

        encoder_output, encoder_layers_attention_heads_weights = self.encoder.forward(source_embeddings, source_mask)

        return encoder_output, encoder_layers_attention_heads_weights

    def decode(self, encoder_output, target, source_mask, target_mask):
        target_embeddings = self.outputEmbeddingLayer.forward(target)
        target_embeddings = self.positionalEncodingLayer.forward(target_embeddings)

        # Apply dropout to the sums of the embeddings and positional encoding
        target_embeddings = self.dropout(target_embeddings)

        decoder_output, decoder_layers_masked_attention_heads_weights, decoder_layers_cross_attention_heads_weights = self.decoder.forward(
            encoder_output,
            target_embeddings,
            source_mask,
            target_mask)

        return decoder_output, decoder_layers_masked_attention_heads_weights, decoder_layers_cross_attention_heads_weights

    def forward(self, source, target, source_mask, target_mask):
        encoder_output, encoder_layers_attention_heads_weights = self.encode(source, source_mask)
        decoder_output, decoder_layers_masked_attention_heads_weights, decoder_layers_cross_attention_heads_weights = self.decode(
            encoder_output, target, source_mask, target_mask)

        # Apply the linear transformation to the decoder output in order to derive the output probabilities
        transformed_decoder_output = self.linear(decoder_output)

        return transformed_decoder_output, encoder_layers_attention_heads_weights, decoder_layers_masked_attention_heads_weights, decoder_layers_cross_attention_heads_weights
