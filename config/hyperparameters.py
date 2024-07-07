DATA_PATH = "./data"
WEIGHTS_PATH = "./data/mt_weights.pt"
embedding_dim = 512
num_heads = 8
hidden_dim = 2048
num_layers = 6
context_length = 30
dropout_prob = 0.1
label_smoothing = 0.1
learning_rate = 0.15  # 1e-1
num_epochs = 30
batch_size = 200
dataset_config = {
    "source_path": f"{DATA_PATH}/en-ig.ig",
    "target_path": f"{DATA_PATH}/en-ig.en",
}
tokenizer_strategy = "huggingface_bpe"
vocab_size = 10_000
train_eval_split_ratio = [0.75, 0.25]
max_dataset_size = 174_918
debug = False
device = "cpu"
use_scheduler = True
warmup_steps = 100
