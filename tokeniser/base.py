from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, input_sequence):
        pass

    @abstractmethod
    def decode(self, encoded_sequence):
        pass
