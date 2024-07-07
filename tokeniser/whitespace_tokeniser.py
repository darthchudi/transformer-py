from .base import Tokenizer
from .utils import special_tokens
import collections
import re


class WhitespaceTokenizer(Tokenizer):
  def __init__(self, corpus):
    max_length = 0

    # Unique set of tokens in our corpus
    vocabulary = set()

    # Split the corpus into new lines
    sentences = corpus.splitlines()

    # Add the special tokens to the vocabulary
    vocabulary.update(special_tokens)

    # Add the unique words to the vocabulary
    for sentence in sentences:
      sentence_words = self.tokenize(sentence)

      vocabulary.update(sentence_words)

      # debug
      if len(sentence_words) > max_length:
        max_length = len(sentence_words)

    # Store the vocabulary
    self.vocabulary = vocabulary

    # Initialise token indices
    self.token_indices = collections.defaultdict(int)
    self.reverse_token_indices = collections.defaultdict(str)

    # Assign an index to each token in the vocabulary
    self.set_token_indices()

  def set_token_indices(self):
    for i, word in enumerate(self.vocabulary):
      self.token_indices[word] = i
      self.reverse_token_indices[i] = word

  def tokenize(self, x):
    regex_pattern = r"[\W\s_]+" # Split by non-alphanumeric characters and white space
    return re.split(regex_pattern, x)

  def decode(self, encoded_sequence):
    return " ".join(encoded_sequence)