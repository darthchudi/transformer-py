from .whitespace_tokeniser import WhitespaceTokenizer
from .hf_bpe_tokeniser import HuggingFaceTokenizer
from .vocabulary import Vocabulary


def init_whitespace_tokenizer(source_path, target_path, device=None):
    source_file = open(source_path, "r")
    target_file = open(target_path, "r")

    source_corpus = source_file.read()
    target_corpus = target_file.read()

    # Initialise the source tokenizer and vocabulary
    source_tokenizer = WhitespaceTokenizer(source_corpus)
    source_vocabulary = Vocabulary(source_tokenizer.token_indices, source_tokenizer.reverse_token_indices, device)

    # Initialise the target tokenizer and vocabulary
    target_tokenizer = WhitespaceTokenizer(target_corpus)
    target_vocabulary = Vocabulary(target_tokenizer.token_indices, target_tokenizer.reverse_token_indices, device)

    return {
        "source_tokenizer": source_tokenizer,
        "source_vocabulary": source_vocabulary,
        "target_tokenizer": target_tokenizer,
        "target_vocabulary": target_vocabulary
    }


def init_huggingface_bpe_tokenizer(device=None,
                                   vocab_size=30_000,
                                   source_path=None,
                                   target_path=None
                                   ):
    source_tokenizer = HuggingFaceTokenizer(
        corpus_path=source_path,
        vocab_size=vocab_size
    )
    source_vocabulary = Vocabulary(source_tokenizer.token_indices, source_tokenizer.reverse_token_indices, device)

    target_tokenizer = HuggingFaceTokenizer(
        corpus_path=target_path,
        vocab_size=vocab_size
    )
    target_vocabulary = Vocabulary(target_tokenizer.token_indices, target_tokenizer.reverse_token_indices, device)

    return {
        "source_tokenizer": source_tokenizer,
        "source_vocabulary": source_vocabulary,
        "target_tokenizer": target_tokenizer,
        "target_vocabulary": target_vocabulary
    }


# Initialises a tokenizer based on a given strategy
def init_tokenizer(source_path, target_path, strategy, device, vocab_size):
    if strategy == "whitespace":
        return init_whitespace_tokenizer(
            source_path,
            target_path,
            device
        )
    elif strategy == "huggingface_bpe":
        return init_huggingface_bpe_tokenizer(
            device=device,
            vocab_size=vocab_size,
            source_path=source_path,
            target_path=target_path,
        )