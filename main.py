from config.hyperparameters import dataset_config, tokenizer_strategy, vocab_size, context_length, WEIGHTS_PATH, device
from translator.translator import Translator
from translator.evaluate import run_evaluation_on_translator

if __name__ == "__main__":
    print("🤠 Running translator evaluation...")

    translator = Translator(
        model_path=WEIGHTS_PATH,
        dataset_config=dataset_config,
        tokenizer_strategy=tokenizer_strategy,
        vocab_size=vocab_size,
        context_length=context_length,
        temperature=1,
        device=device
    )

    run_evaluation_on_translator(translator)

    print("✨ Completed!")
