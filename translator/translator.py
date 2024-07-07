import torch
import math
from transformer import Transformer
from tokeniser.utils import  TOKEN_END, TOKEN_BEGIN, TOKEN_PAD, indices_to_sentences, decode_bpe_sequence
from tokeniser.init import init_tokenizer
from config.hyperparameters import embedding_dim, hidden_dim, num_heads, num_layers, dropout_prob, context_length
from .utils import safe_log


class Translator:
    def __init__(self,
                 device="cuda",
                 model_path=None,
                 dataset_config={},
                 tokenizer_strategy=None,
                 vocab_size=None,
                 context_length=None,
                 temperature=1,
                 ):
        self.device = device
        self.context_length = context_length
        self.tokenizer_strategy = tokenizer_strategy
        self.dataset_config = dataset_config
        self.temperature = temperature

        # Create the tokenizer
        tokenizer_config = init_tokenizer(strategy=tokenizer_strategy,
                                        source_path=dataset_config["source_path"],
                                        target_path=dataset_config["target_path"],
                                        device=device,
                                        vocab_size=vocab_size)

        self.source_vocabulary = tokenizer_config["source_vocabulary"]
        self.target_vocabulary = tokenizer_config["target_vocabulary"]
        self.source_tokenizer = tokenizer_config["source_tokenizer"]
        self.target_tokenizer = tokenizer_config["target_tokenizer"]

        # Initialise the model and load the weights
        transformer = Transformer(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            context_length=context_length,
            num_heads=num_heads,
            visualiser=None,
            source_vocabulary=self.source_vocabulary,
            target_vocabulary=self.target_vocabulary,
            device=device,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
            temperature=temperature
        )

        transformer.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

        # Set eval mode
        transformer.eval()
        transformer.to(device)

        self.transformer = transformer

    def tokenize_source_sentence(self, sentence, add_extra_tokens=True):
        tokens = self.source_tokenizer.tokenize(sentence)

        if add_extra_tokens:
            # Truncate the source and target tokens (if applicable) so that they have
            # enough space for the beginning of sentence and end of sentence padding tokens
            max_context_length_without_start_end_markers = self.context_length - 2
            if len(tokens) > max_context_length_without_start_end_markers:
                tokens = tokens[:max_context_length_without_start_end_markers]

            # Append the start and end token markers to the sentence tokens
            tokens.insert(0, TOKEN_BEGIN)
            tokens.append(TOKEN_END)

            # Pad the sentence tokens so that they match the context length
            while len(tokens) < context_length:
                tokens.append(TOKEN_PAD)

        # Get the token indices
        token_indices = self.source_vocabulary.get_tokens_indices(tokens)

        # Generate the mask for padding tokens, as we do not need to attend to these tokens.
        tokens_mask = self.source_vocabulary.get_padding_mask(token_indices)
        tokens_mask = tokens_mask.to(dtype=torch.float32)

        # Add an extra dimension for the batch size and move the tensor the provided device
        # This transforms the tensors to (batch size, context length, embedding dim)
        # shaped tensors, where batch size = 1 during inference
        token_indices = token_indices.unsqueeze(0).to(device=self.device)

        return token_indices, tokens_mask, tokens

    def get_initial_memory_tokens(self):
        # Get the token indices from the target vocabulary
        tokens_indices = self.target_vocabulary.get_tokens_indices([TOKEN_BEGIN])

        # Add an extra dimension for the batch size and move to the provided device
        # This transforms the tensors to (batch size, context length, embedding dim)
        # shaped tensors, where batch size  = 1 during inference
        tokens_indices = tokens_indices.unsqueeze(0).to(device=self.device)

        return tokens_indices

    def parse_memory(self, memory):
        sentences = indices_to_sentences(self.transformer.target_vocabulary, memory)

        # Return the first sentence as it's the only relevant one
        return sentences[0]

    def greedy_decode(self, encoded_tokenized_source_sentence, tokenized_source_mask, memory):
        """
    Decodes the next position N+1 in an output sequence given a memory of N tokens.
    The next position is selected based on the maximum probability for each position
    """
        # Clone the memory tensor
        memory = torch.clone(memory)

        memory_lookahead_mask = self.target_vocabulary.get_look_ahead_mask(memory)

        decoder_output, decoder_layers_masked_attention_heads_weights, decoder_layers_cross_attention_heads_weights = self.transformer.decode(
            encoded_tokenized_source_sentence,
            memory,
            tokenized_source_mask,
            memory_lookahead_mask
        )

        # Apply the linear transformation to the decoder output in order to derive the output logits
        # based on the target vocabulary
        decoder_output_logits = self.transformer.linear(decoder_output)

        # Apply the temperature before the softmax
        decoder_output_logits = decoder_output_logits / self.temperature

        # Get the probabilities for the output logits
        decoder_output_probabilities = self.transformer.softmax(decoder_output_logits)

        # Get the outputs with the maximum probability for each word position, across the vocabulary size dimension.
        # This transforms the decoder output probability tensor from a (batch size, context length, vocab size)
        # shaped tensor to a (batch size, context length) tensor, where the values along the context length dimension
        # are the the vocabulary indices of the next word at that position.
        tokens_predictions_probabilities, tokens_predictions = torch.max(decoder_output_probabilities, dim=-1)

        # Get the token for the last item, as this represents the next word
        # This fetches a (context_length) shaped tensor.
        # There's only one batch in this case, which is our inference input
        inference_batch = tokens_predictions[0]

        last_token_position_prediction = inference_batch[-1]  # Context length tensor
        next_token = last_token_position_prediction.item()

        next_token_to_be_appended = torch.tensor([next_token]).unsqueeze(0).to(self.device)

        # Add the new token to the working memory
        memory = torch.cat((memory, next_token_to_be_appended), dim=-1)

        # Parse memory
        parsed_memory = self.parse_memory(memory)
        parsed_next_token = parsed_memory[-1]

        # Get the next token probability
        next_token_probability = tokens_predictions_probabilities[0][-1]

        return {
            "memory": memory,
            "parsed_memory": parsed_memory,
            "parsed_next_token": parsed_next_token,
            "next_token_probability": next_token_probability.item(),
            "decoder_layers_masked_attention_heads_weights": decoder_layers_masked_attention_heads_weights,
            "decoder_layers_cross_attention_heads_weights": decoder_layers_cross_attention_heads_weights
        }

    def greedy_decode_until_end(self, encoded_tokenized_source_sentence, tokenized_source_mask, memory):
        """
    Generates an output target sequence given an encoded source sequence by iteratively calling the decoder until the
    context length is reached or the <EOS> end of sentence token is predicted.
    """
        # Initialise variables for attention interpretability
        decoder_layers_masked_attention_heads_weights = None
        decoder_layers_cross_attention_heads_weights = None

        # Initialise output sequence
        output_sequence = self.parse_memory(memory)
        output_sequence_token_probabilities = []

        # Counters used for iteratively calling the decoder until we encounter the ending conditions
        has_not_reached_max_context_length = True
        last_token_is_not_end_of_speech_token = True

        # Get context length limit
        limit = self.context_length - 1

        while has_not_reached_max_context_length and last_token_is_not_end_of_speech_token:
            decoder_result = self.greedy_decode(
                encoded_tokenized_source_sentence,
                tokenized_source_mask,
                memory
            )

            memory = decoder_result["memory"]
            parsed_memory = decoder_result["parsed_memory"]
            parsed_next_token = decoder_result["parsed_next_token"]
            decoder_layers_masked_attention_heads_weights = decoder_result[
                "decoder_layers_masked_attention_heads_weights"]
            decoder_layers_cross_attention_heads_weights = decoder_result[
                "decoder_layers_cross_attention_heads_weights"]
            next_token_probability = decoder_result["next_token_probability"]

            # Check if we reached the limit
            if len(parsed_memory) >= limit:
                has_not_reached_max_context_length = False

            # Get the last predicted token
            parsed_next_token = parsed_memory[-1]

            # Check if we've encoutered the EOS token
            if parsed_next_token == TOKEN_END:
                last_token_is_not_end_of_speech_token = False

            output_sequence.append(parsed_next_token)
            output_sequence_token_probabilities.append(next_token_probability)

        return {
            "output_sequence": output_sequence,
            "output_sequence_token_probabilities": output_sequence_token_probabilities,
            "decoder_layers_masked_attention_heads_weights": decoder_layers_masked_attention_heads_weights,
            "decoder_layers_cross_attention_heads_weights": decoder_layers_cross_attention_heads_weights
        }

    def translate(self, source_sentence, decoding_strategy=None, beam_search_size=2, debug=False):
        tokenized_source_sentence, tokenized_source_mask, tokenized_source_sentence_raw = self.tokenize_source_sentence(
            source_sentence)

        # Encode the source sentence tokens
        encoded_tokenized_source_sentence, encoder_layers_attention_heads_weights = self.transformer.encode(
            tokenized_source_sentence, tokenized_source_mask)

        # Get memory
        memory = self.get_initial_memory_tokens()

        # Initialise decoding parameters
        use_beam_search_decoding_strategy = decoding_strategy == "beam_search"

        if use_beam_search_decoding_strategy:
            result = self.beam_search_decode(
                encoded_tokenized_source_sentence,
                tokenized_source_mask,
                beam_search_size,
                debug
            )
        else:
            result = self.greedy_decode_until_end(
                encoded_tokenized_source_sentence,
                tokenized_source_mask,
                memory
            )

        output_sequence = result["output_sequence"]
        decoder_layers_masked_attention_heads_weights = result["decoder_layers_masked_attention_heads_weights"]
        decoder_layers_cross_attention_heads_weights = result["decoder_layers_cross_attention_heads_weights"]

        # Decode the computed sequence
        translation = decode_bpe_sequence(output_sequence)

        # Remove the start and end tokens and trim whitespace
        translation = translation.replace(TOKEN_BEGIN, "")
        translation = translation.replace(TOKEN_END, "")

        translation = translation.strip()

        return translation, {
            "encoder_layers_attention_heads_weights": encoder_layers_attention_heads_weights,
            "tokenized_source_sentence_raw": tokenized_source_sentence_raw,
            "decoder_layers_masked_attention_heads_weights": decoder_layers_masked_attention_heads_weights,
            "decoder_layers_cross_attention_heads_weights": decoder_layers_cross_attention_heads_weights,
            "tokenized_target_sentence_raw": output_sequence,
        }

    def beam_search_decode(self, encoded_tokenized_source_sentence, tokenized_source_mask, beam_size, debug):
        # Init variables for attention interpretability
        best_hypothesis_decoder_layers_masked_attention_heads_weights = None
        best_hypothesis_decoder_layers_cross_attention_heads_weights = None

        # Get memory
        memory = self.get_initial_memory_tokens()

        memory_lookahead_mask = self.target_vocabulary.get_look_ahead_mask(memory)

        decoder_output, decoder_layers_masked_attention_heads_weights, decoder_layers_cross_attention_heads_weights = self.transformer.decode(
            encoded_tokenized_source_sentence,
            memory,
            tokenized_source_mask,
            memory_lookahead_mask
        )

        # Apply the linear transformation to the decoder output in order to derive the output logits
        # based on the target vocabulary
        decoder_output_logits = self.transformer.linear(decoder_output)

        # Apply the temperature before the softmax
        decoder_output_logits = decoder_output_logits / self.temperature

        # Get the probabilities for the output logits
        decoder_output_probabilities = self.transformer.softmax(decoder_output_logits)

        # Get the top K outputs with the maximum probability for each word position, across the vocabulary size
        # dimension. This transforms the decoder output probability tensor from a (batch size, context length,
        # vocab size) shaped tensor to a (batch size, context length, beam search size) tensor, where the values
        # along the context length dimension are the the top K probabilities. This allows us implement a beam search
        # over K hypothesis for the most probable out sequence
        topk_token_prediction_probabilities, topk_token_prediction_indexes = torch.topk(decoder_output_probabilities,
                                                                                        k=beam_size, dim=-1)

        next_token_topk_prediction_indexes = topk_token_prediction_indexes[-1]

        next_token_topk_prediction_probabilities = topk_token_prediction_probabilities[0][-1]

        # Remove the context length dimension from the next token top k predictions to convert it from a
        # (1, beam search size) shaped tensor to a (beam search size) shaped tensor
        next_token_topk_prediction_indexes = next_token_topk_prediction_indexes.squeeze(0)

        best_hypothesis = ""
        best_hypothesis_output_sequence = ""
        max_seen_prob = None

        for i, probable_next_token_position in enumerate(next_token_topk_prediction_indexes):
            next_token_to_be_appended = torch.tensor([probable_next_token_position]).unsqueeze(0).to(self.device)

            # Add the new token to the working memory for this hypothesis
            hypothesis_memory = torch.cat((memory, next_token_to_be_appended), dim=-1)

            # Decode the hypothesis sequence
            result = self.greedy_decode_until_end(
                encoded_tokenized_source_sentence,
                tokenized_source_mask,
                hypothesis_memory
            )

            output_sequence = result["output_sequence"]
            output_sequence_token_probabilities = result["output_sequence_token_probabilities"]
            decoder_layers_masked_attention_heads_weights = result["decoder_layers_masked_attention_heads_weights"]
            decoder_layers_cross_attention_heads_weights = result["decoder_layers_cross_attention_heads_weights"]

            # Decode the computed sequence
            translation = decode_bpe_sequence(output_sequence)

            # Remove the start and end tokens and trim whitespace
            translation = translation.replace(TOKEN_BEGIN, "")
            translation = translation.replace(TOKEN_END, "")

            translation = translation.strip()

            # Compute the total probability for this prediction as a product of the probability
            # for each position, which is effectively the sum of the log probabilities.
            base_hypothesis_prediction_prob = safe_log(next_token_topk_prediction_probabilities[i])
            hypothesis_prediction_prob = base_hypothesis_prediction_prob + sum(
                [safe_log(prob) for prob in output_sequence_token_probabilities])

            if max_seen_prob is None or hypothesis_prediction_prob >= max_seen_prob:
                best_hypothesis = translation
                max_seen_prob = hypothesis_prediction_prob
                best_hypothesis_output_sequence = output_sequence

                best_hypothesis_decoder_layers_masked_attention_heads_weights = decoder_layers_masked_attention_heads_weights
                best_hypothesis_decoder_layers_cross_attention_heads_weights = decoder_layers_cross_attention_heads_weights

            if debug:
                print(f"Translation -> {translation}. Probability -> {hypothesis_prediction_prob}")

        if debug:
            print(f"\n✅ Best prediction -> {best_hypothesis}\nBest prediction probability -> {max_seen_prob}\n")

        return {
            "output_sequence": best_hypothesis_output_sequence,
            "translation": best_hypothesis,
            "decoder_layers_masked_attention_heads_weights": best_hypothesis_decoder_layers_masked_attention_heads_weights,
            "decoder_layers_cross_attention_heads_weights": best_hypothesis_decoder_layers_cross_attention_heads_weights
        }