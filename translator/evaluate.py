from bertviz import head_view
from datetime import datetime
from .translator import Translator
from tokeniser.utils import TOKEN_END, format_attention_heads_weights_for_viz
from data.test.sentences import base_source_sentences

def run_evaluation_on_translator(translator: Translator):
    for i, sentence in enumerate(base_source_sentences):
        source_sentence = sentence["source"]
        target_sentence = sentence["target"]

        translation, metadata = translator.translate(source_sentence, decoding_strategy="beam_search", debug=True,
                                                     beam_search_size=5)

        print(
            f"Translation {i + 1}: {translation}\nTarget Sentence: {target_sentence}\nSource Sentence: {source_sentence}\n")

        encoder_layers_attention_heads_weights = metadata["encoder_layers_attention_heads_weights"]
        tokenized_source_sentence_raw = metadata["tokenized_source_sentence_raw"]
        decoder_layers_masked_attention_heads_weights = metadata["decoder_layers_masked_attention_heads_weights"]
        decoder_layers_cross_attention_heads_weights = metadata["decoder_layers_cross_attention_heads_weights"]
        tokenized_target_sentence_raw = metadata["tokenized_target_sentence_raw"]

        # Remove the pad tokens from the source and target tokenized sentences
        source_eos_position = tokenized_source_sentence_raw.index(TOKEN_END)
        tokenized_source_sentence_raw = tokenized_source_sentence_raw[:source_eos_position + 1]

        # Default to the eos position as the final index in the target sequence if no <EOS> is present
        target_eos_position = len(tokenized_target_sentence_raw) - 1

        if TOKEN_END in tokenized_target_sentence_raw:
            target_eos_position = tokenized_target_sentence_raw.index(TOKEN_END)
            tokenized_target_sentence_raw = tokenized_target_sentence_raw[:target_eos_position + 1]

        html_head_view = head_view(
            encoder_attention=format_attention_heads_weights_for_viz(encoder_layers_attention_heads_weights,
                                                                     eos_positions=[source_eos_position,
                                                                                    source_eos_position]),
            encoder_tokens=tokenized_source_sentence_raw,
            decoder_attention=format_attention_heads_weights_for_viz(decoder_layers_masked_attention_heads_weights,
                                                                     eos_positions=[target_eos_position,
                                                                                    target_eos_position]),
            cross_attention=format_attention_heads_weights_for_viz(decoder_layers_cross_attention_heads_weights,
                                                                   eos_positions=[target_eos_position,
                                                                                  source_eos_position]),
            decoder_tokens=tokenized_target_sentence_raw[1:],  # Trim the <BOS token> from the target sentence
            html_action="return"
        )

        with open(f"./artifacts/head_view_{datetime.now()}.html", 'w') as file:
            file.write(html_head_view.data)

        print("\n========\n")