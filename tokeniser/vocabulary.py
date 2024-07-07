import torch
from .utils import TOKEN_UNKNOWN, TOKEN_PAD


class Vocabulary:
    def __init__(self, token_indices, reverse_token_indices, device):
        # Store the vocabulary and device
        self.data = token_indices
        self.reverse_vocabulary = reverse_token_indices
        self.device = device

    def __len__(self):
        return len(self.data)

    def get_token_from_index(self, index):
        return self.reverse_vocabulary.get(index)

    def get_token_index(self, token):
        if token not in self.data:
            return self.data[TOKEN_UNKNOWN]

        token_index = self.data.get(token)
        return token_index

    def get_tokens_indices(self, tokens):
        indices = []

        for token in tokens:
            token_index = self.get_token_index(token)
            indices.append(token_index)

        return torch.LongTensor(indices).to(self.device)

    def get_batch_tokens_indices(self, batch_tokens):
        batch_indices = []

        for row in batch_tokens:
            row_token_indices = []

            for token in row:
                token_index = None
                if token not in self.data:
                    token_index = self.data[TOKEN_UNKNOWN]
                else:
                    token_index = self.data.get(token)

                row_token_indices.append(token_index)

            batch_indices.append(row_token_indices)

        return torch.LongTensor(batch_indices).to(self.device)

    def get_padding_token_index(self):
        return self.data[TOKEN_PAD]

    def get_padding_mask(self, token_indices):
        pad_token_index = self.data[TOKEN_PAD]
        mask = token_indices != pad_token_index

        # Cast the mask to the device and expected type
        mask = mask.to(device=self.device, dtype=torch.int64)

        # Insert a new dimension so the padding mask becomes a 2D tensor
        mask = mask.unsqueeze(0)
        return mask

    def get_look_ahead_mask(self, token_indices):
        # The look ahead mask is a square shaped matrix, which is used to prevent a
        # given model from attending to future words.
        # The mask itself indicates whether a given word should attend to every
        # other given word.
        # It contains a row for each word and an entry in each column to
        # indicate if the row word should attend to the word
        # where token_indices = (context_length x embedding_dim)
        context_length = token_indices.size(0)
        look_ahead_mask = torch.ones(context_length, context_length)
        look_ahead_mask = torch.tril(look_ahead_mask, diagonal=0)

        return look_ahead_mask.to(device=self.device, dtype=torch.int64)
