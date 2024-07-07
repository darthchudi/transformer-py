from cog import BasePredictor, Input, Path
import torch
from translator.translator import Translator
from config.hyperparameters import *

class Predictor(BasePredictor):
    def setup(self):
        self.translator = Translator(
            model_path=WEIGHTS_PATH,
            dataset_config=dataset_config,
            tokenizer_strategy=tokenizer_strategy,
            vocab_size=vocab_size,
            context_length=context_length,
            temperature=1,
            device=device
        )

    def predict(self, source_text: str = Input(description="Text to translate")) -> str:
        translation, metadata = self.translator.translate(source_text, decoding_strategy="beam_search", debug=True,
                                                     beam_search_size=5)


        return translation