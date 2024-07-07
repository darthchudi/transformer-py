from utils import TOKEN_UNKNOWN, SYMBOL_END_OF_WORD, special_tokens
from base import Tokenizer
import collections


class BPETokenizer(Tokenizer):
    def __init__(
            self, corpus, vocabulary_size, handle_unknown_characters=False, debug=False
    ):
        words = corpus.split()

        # Stores how often a given word appears in the corpus
        word_frequency = collections.defaultdict(int)

        # Unique set of tokens in our corpus
        vocabulary = set([])

        # Stores and maps learned token pairs to the merged token result
        merges = collections.defaultdict(str)

        # Maps words to the tokens used to tokenize the word
        word_lookup_table = collections.defaultdict(list)

        for token in special_tokens:
            vocabulary.add(token)

        for word in words:
            word_frequency[word] += 1

        # Build the initial vocabulary based on the unique characters
        # present in each word within the corpus
        for word in word_frequency:
            for character in word:
                vocabulary.add(character)

        # Create a map of initial tokens for each word.
        # Words will initially be tokenized with 1-grams i.e single
        # character tokens, after which we learn merge rules for
        # tokenizing with more characters based on the most frequent pairs.
        for word in word_frequency:
            word_characters = [character for character in word]

            # Add the end of word character to the symbol list for the word
            word_symbols = word_characters + [SYMBOL_END_OF_WORD]

            word_lookup_table[word] = word_symbols

        # Store the tokenizer state
        self.word_frequency = word_frequency
        self.vocabulary = vocabulary
        self.merges = merges
        self.word_lookup_table = word_lookup_table
        self.handle_unknown_characters = handle_unknown_characters

        # Learn merge rules across the corpus
        self.learn_merge_rules(vocabulary_size, debug=debug)

        # Initialise token indices
        self.token_indices = None
        self.reverse_token_indices = None

        # Assign an index to each token in the vocabulary
        self.set_token_indices()

        # Create a cache for fast look-ups opf tokenized sequences
        self.cache = {}

    # learn_merge_rules iteratively learns new merge rules over the corpus until
    # we get to the vocabulary size limit
    def learn_merge_rules(self, vocabulary_size, debug=False):
        corpus_pair_stats = self.get_corpus_pair_stats()

        while len(self.vocabulary) < vocabulary_size:
            # Get the most frequent pair in the corpus
            max_pair_in_corpus, max_pair_in_corpus_frequency = (
                self.get_most_frequent_pair_in_corpus(corpus_pair_stats)
            )

            if debug:
                print(
                    f"Vocabulary size: {len(self.vocabulary)}, max pair: {max_pair_in_corpus}, frequency: {max_pair_in_corpus_frequency}"
                )

            # No more token pairs to be merged as we've exhausted all possible merge
            # rules in the corpus, so we stop training
            if max_pair_in_corpus is None:
                break

            # Add the most frequent pair to the list of learned merges and vocabulary as a string
            max_pair_in_corpus_str = "".join(max_pair_in_corpus)
            self.merges[max_pair_in_corpus] = max_pair_in_corpus_str
            self.vocabulary.add(max_pair_in_corpus_str)

            # Replace the the two individual bytes in each word's lookup table with
            # the max byte-pair token
            corpus_pair_stats = self.merge_byte_pair_in_corpus(
                max_pair_in_corpus, max_pair_in_corpus_str, corpus_pair_stats
            )

    def get_corpus_pair_stats(self):
        pair_stats = collections.defaultdict(int)

        for word, word_bytes in self.word_lookup_table.items():
            # Count the frequency of each byte-pair in the word.
            # We iterate up until the second to last byte as that's the start of the last byte-pair.
            for i in range(len(word_bytes) - 1):
                pair_stats[word_bytes[i], word_bytes[i + 1]] += 1

        return pair_stats

    def get_most_frequent_pair_in_corpus(self, corpus_pair_stats):
        max_pair = None
        max_frequency = 0

        for pair, frequency in corpus_pair_stats.items():
            if max_pair is None or frequency > max_frequency:
                max_pair = pair
                max_frequency = frequency

        return max_pair, max_frequency

    # Replaces the the two individual bytes in each word's lookup table with
    # the max byte-pair token
    def merge_byte_pair_in_corpus(
            self, max_pair_in_corpus, max_pair_in_corpus_str, corpus_pair_stats
    ):
        for word in self.word_lookup_table:
            word_bytes = self.word_lookup_table[word]

            i = 0

            while i < len(word_bytes) - 1:
                pair_key = (word_bytes[i], word_bytes[i + 1])

                if pair_key == max_pair_in_corpus:
                    # Replace the bytes with the byte-pair token
                    word_bytes_start_chunk = word_bytes[:i]
                    word_bytes_replacement_chunk = [max_pair_in_corpus_str]
                    word_bytes_end_chunk = word_bytes[i + 2:]

                    updated_word_bytes = (
                            word_bytes_start_chunk
                            + word_bytes_replacement_chunk
                            + word_bytes_end_chunk
                    )

                    word_bytes = updated_word_bytes
                else:
                    # Increment the pointer, so we evaluate the next byte-pair
                    i += 1

            # Update the stored word bytes
            self.word_lookup_table[word] = word_bytes

            # Update the corpus pair stats for the merged byte-pair
            corpus_pair_stats = self.update_corpus_pair_stats(
                corpus_pair_stats,
                max_pair_in_corpus,
                max_pair_in_corpus_str,
                word_bytes,
            )

        return corpus_pair_stats

    def update_corpus_pair_stats(
            self, corpus_pair_stats, max_pair_in_corpus, max_pair_in_corpus_str, word_bytes
    ):
        # Decrement the frequency of the individual byte-pair
        corpus_pair_stats[max_pair_in_corpus] -= 1

        # Delete the byte-pair from the corpus pair stats if its frequency is 0
        if corpus_pair_stats[max_pair_in_corpus] == 0:
            del corpus_pair_stats[max_pair_in_corpus]

        i = 0
        while i < len(word_bytes) - 1:
            is_merged_max_pair = word_bytes[i] == max_pair_in_corpus_str
            if not is_merged_max_pair:
                i += 1
                continue

            # Update the frequency of the merged byte-pair based on the neighbouring byte-pairs
            if i > 0:
                # Update the frequency of the byte-pair to the left of the merged byte-pair
                corpus_pair_stats[word_bytes[i - 1], word_bytes[i]] += 1
            elif i < len(word_bytes) - 1:
                # Update the frequency of the byte-pair to the right of the merged byte-pair
                corpus_pair_stats[word_bytes[i], word_bytes[i + 1]] += 1

            i += 1

        return corpus_pair_stats

    def set_token_indices(self):
        # Maps tokens in the vocabulary to a unique index
        token_indices = collections.defaultdict(int)

        # Maps a token index to a token
        reverse_token_indices = collections.defaultdict(str)

        # Assign a token index to each token in the vocabulary
        for i, token in enumerate(self.vocabulary):
            token_indices[token] = i
            reverse_token_indices[i] = token

        self.token_indices = token_indices
        self.reverse_token_indices = reverse_token_indices

    def tokenize(self, input_sequence):
        # Check if the input sequence exists in the cache
        if self.cache.get(input_sequence) is not None:
            return self.cache[input_sequence]

        # The sequence doesn't exist in the cache, so we proceed with tokenisation
        tokens = []

        # Split the input sequence into individual words
        words = input_sequence.split()

        for word in words:
            # Split the word into individual characters, which represent the bytes
            word_bytes = [character for character in word]

            # Add the end of word symbol to the list of word bytes
            word_bytes.append(SYMBOL_END_OF_WORD)

            # Handle unknown characters/bytes by replacing them with the special <UNK> token
            if self.handle_unknown_characters:
                for i, byte in enumerate(word_bytes):
                    if byte not in self.vocabulary:
                        word_bytes[i] = TOKEN_UNKNOWN

            # Iterate through the merge rules and see if there are rules for the byte-pairs in the word
            for merge_rule_pair, merge_rule_pair_str in self.merges.items():
                i = 0

                while i < len(word_bytes) - 1:
                    word_byte_pair = (word_bytes[i], word_bytes[i + 1])

                    if word_byte_pair == merge_rule_pair:
                        # Merge the bytes
                        start_chunk = word_bytes[:i]
                        replacement_chunk = [merge_rule_pair_str]
                        end_chunk = word_bytes[i + 2:]

                        word_bytes = start_chunk + replacement_chunk + end_chunk
                    else:
                        i += 1

            # Add the word bytes to the flattened list of tokens
            tokens.append(word_bytes)

        flatted_tokens = []
        for word_tokens in tokens:
            flatted_tokens += word_tokens

        # Store the tokenised representation of the sequence in the cache
        self.cache[input_sequence] = flatted_tokens

        return flatted_tokens

    def decode(self, encoded_sequence):
        words = []
        current_word = ""

        for token in encoded_sequence:
            is_word_boundary_token = token == SYMBOL_END_OF_WORD or token.endswith(
                SYMBOL_END_OF_WORD
            )

            if is_word_boundary_token:
                # Append the word to the list of words if we've reached a word boundary
                current_word += token
                words.append(current_word)

                # Reset the current word
                current_word = ""
            else:
                # This token is a sub-word token, representing a byte in the word
                current_word += token

        # Remove the end of word token from each word
        words = self.sanitise_word_boundary_tokens(words)

        # Join the words to form the decoded sequence
        return " ".join(words)

    def sanitise_word_boundary_tokens(self, words):
        for i, word in enumerate(words):
            word = word.replace(SYMBOL_END_OF_WORD, "")

            words[i] = word

        return words
