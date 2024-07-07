import torch
from nltk.translate.bleu_score import corpus_bleu

TOKEN_BEGIN = "<BOS>"
TOKEN_END = "<EOS>"
TOKEN_UNKNOWN = "<UNK>"
TOKEN_PAD = "<PAD>"
TOKEN_SPACE = " "
SYMBOL_END_OF_WORD = "<w/>"

special_tokens = [
    TOKEN_BEGIN,
    TOKEN_END,
    TOKEN_UNKNOWN,
    TOKEN_PAD,
    TOKEN_SPACE,
    SYMBOL_END_OF_WORD,
]


def predictions_to_sentences(transformer, decoder_output):
    y = transformer.softmax(decoder_output)
    _, predictions = torch.max(y, dim=2)

    output_sentences = []

    for prediction in predictions:
        predicted_sentence = []
        for token_index in prediction:
            token_index = token_index.item()
            token = transformer.target_vocabulary.get_token_from_index(token_index)
            predicted_sentence.append(token)

        output_sentences.append(predicted_sentence)

    return output_sentences


def indices_to_sentences(vocabulary, token_indices_batch):
    output_sentences = []

    for token_indices in token_indices_batch:
        sentence = []
        for token_index in token_indices:
            token_index = token_index.item()
            # Retrieve the token from the vocabulary
            token = vocabulary.get_token_from_index(token_index)
            sentence.append(token)

        # Store the sentence
        output_sentences.append(sentence)

    return output_sentences


def clone_tensor(x):
    return x.clone().detach()


def sanitise_word_boundary_tokens(words):
    _words = words

    for i, word in enumerate(_words):
        word = word.replace(SYMBOL_END_OF_WORD, "")
        _words[i] = word

    return _words


def decode_bpe_sequence(encoded_sequence):
    words = []
    current_word = ""

    for token in encoded_sequence:
        is_word_boundary_token = token == SYMBOL_END_OF_WORD or token.endswith(
            SYMBOL_END_OF_WORD
        )

        is_start_or_end_token = token == TOKEN_BEGIN or token == TOKEN_END

        if is_word_boundary_token:
            # Append the word to the list of words if we've reached a word boundary
            current_word += token
            words.append(current_word)

            # Reset the current word
            current_word = ""
        elif is_start_or_end_token:
            # Handle the begin or end token
            words.append(token)

            # Reset the current word
            current_word = ""
        else:
            # This token is a sub-word token, representing a byte in the current word,
            # so it's appended to the current word
            current_word += token

    # Remove the end of word token from each word
    words = sanitise_word_boundary_tokens(words)

    # Join the words to form the decoded sequence
    return " ".join(words)


def format_tokens_for_printing(tokenizer_strategy, tokens_batch, sample_size):
    # Select the samples
    tokens_batch = tokens_batch[:sample_size]

    # Handle BPE
    is_bpe_sequence = "bpe" in tokenizer_strategy

    if is_bpe_sequence:
        return [decode_bpe_sequence(tokens) for tokens in tokens_batch]
    else:
        return [" ".join(tokens) for tokens in tokens_batch]


def sanitise_sentence_for_bleu_score(sentence):
    for i, token in enumerate(sentence):
        if token == TOKEN_END:
            # Return the sentence tokens up until the end
            # of sentence token if found
            return sentence[:i]

    # Return the sentence as is if the end of sentence
    # token is not found
    return sentence


def compute_corpus_bleu_score(predicted_sentences, target_sentences):
    predicted_sentences = predicted_sentences
    target_sentences = target_sentences

    batch_list_of_references = []
    batch_hypotheses = []

    # Predict the bleu score for each sentence
    for i, _ in enumerate(predicted_sentences):
        predicted_sentence = sanitise_sentence_for_bleu_score(predicted_sentences[i])
        target_sentence = sanitise_sentence_for_bleu_score(target_sentences[i])

        # If the predicted sentence has no tokens adfter sanitisation then we insert the unknown token N-times (where
        # N is the context length of the target sentence) so we can compute the bleu score. The function expects to
        # find at least 2 N-grams in the hypothesis. — TODO: confirm this
        if len(predicted_sentence) <= 0:
            predicted_sentence = [TOKEN_UNKNOWN] * len(target_sentence)

        # The bleu score function expects a list of target references to compare the hypothesis
        # against. In our case, we have only onee reference which is the target sentence, so we
        # append a single-list reference
        batch_list_of_references.append([target_sentence])

        # Include the predicted sentence as the hypothesis
        batch_hypotheses.append(predicted_sentence)

    corpus_bleu_score = corpus_bleu(batch_list_of_references, batch_hypotheses)
    return corpus_bleu_score

def format_attention_heads_weights_for_viz(layers_attention_heads_weights, eos_positions):
    """
  Converts a list where each entry represents a given layer's attention heads tensors as a list.
  Each entry contains a list of attention score tensors shaped (1, context_length, context_length) for each
  attention head in a given layer.
  The output value is a list of (1, num_heads, context_length, context_length) tensors which can be used by the
  BertViz library
  """

    formatted_attention_heads_weights = []

    for layer in layers_attention_heads_weights:
        attention_heads = []

        for head in layer:
            # Convert the (1, context_length, context_length) shaped attention head value to a 2d tensor
            # (context_length, context_length)
            reshaped_head_attention = head.view(head.shape[-2], head.shape[-1])

            # Remove the pad tokens
            first_eos_position = eos_positions[0]
            second_eos_position = eos_positions[1]
            reshaped_head_attention = reshaped_head_attention[:first_eos_position + 1, :second_eos_position + 1]

            attention_heads.append(reshaped_head_attention)

        # Stack the list of attention head weights in this layer
        # into a tensor with the shape (1, num_heads, context_length, context_length)
        layer_stacked_attention_heads = torch.stack(attention_heads).unsqueeze(0)

        formatted_attention_heads_weights.append(layer_stacked_attention_heads)

    return formatted_attention_heads_weights
