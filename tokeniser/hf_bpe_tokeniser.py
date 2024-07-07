from tokenizers import Tokenizer as HFTokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from .base import Tokenizer
from .utils import TOKEN_BEGIN, TOKEN_END, TOKEN_UNKNOWN, TOKEN_PAD, SYMBOL_END_OF_WORD


class HuggingFaceTokenizer(Tokenizer):
    def __init__(self, corpus_path, vocab_size=40_000):
        # Create the tokenizer
        self.tokenizer = HFTokenizer(BPE(unk_token=TOKEN_UNKNOWN))
        self.tokenizer.pre_tokenizer = Whitespace()

        # Create the trainer
        self.trainer = BpeTrainer(
            special_tokens=[TOKEN_UNKNOWN, TOKEN_BEGIN, TOKEN_END, TOKEN_PAD],
            vocab_size=vocab_size,
            end_of_word_suffix=SYMBOL_END_OF_WORD
        )

        # Train the tokenizer
        print(f"Training BPE Tokenizer on corpus {corpus_path}")
        self.tokenizer.train([corpus_path], self.trainer)
        print(f"Done training BPE Tokenizer with vocabulary size {len(self.tokenizer.get_vocab())}")

        # Store the token indices i.e vocabulary
        self.token_indices = self.tokenizer.get_vocab()
        self.reverse_token_indices = {index: token for token, index in self.token_indices.items()}

    def tokenize(self, input_sequence, with_full_info=False):
        tokenized_sequence_output = self.tokenizer.encode(input_sequence)

        if with_full_info:
            return tokenized_sequence_output.tokens, tokenized_sequence_output
        else:
            return tokenized_sequence_output.tokens

    def decode(self, encoded_sequence):
        decoded_sequence_output = self.tokenizer.decode(encoded_sequence)
        return decoded_sequence_output
