# Transformer From Scratch

A PyTorch implementation of the encoder-decoder Transformer described in
"Attention Is All You Need".

![Transformer Architecture](assets/architecture.png)

## Features

- Token embeddings and sinusoidal positional encoding
- Multi-head self-attention and cross-attention
- Residual connections, layer normalization, and feed-forward blocks
- Greedy decoding for validation
- Word-level tokenizers built with Hugging Face `tokenizers`
- Training on the Hugging Face `Helsinki-NLP/opus_books` English-Italian dataset
- Automatic checkpoint saving and optional resume from the latest checkpoint

## Project Structure

- `model.py` contains the Transformer architecture and `build_transformer(...)`.
- `dataset.py` prepares padded encoder inputs, decoder inputs, labels, and masks.
- `config.py` contains the training configuration and checkpoint path helpers.
- `train.py` downloads the dataset, builds tokenizers, trains the model, validates it, and saves checkpoints.

## Requirements

- Python 3.11
- PyTorch
- Hugging Face `datasets` and `tokenizers`
- `tqdm`

The project includes a `pyproject.toml` and is configured for the Python 3.11
environment recorded in `.python-version`. With `uv` installed, create or
update the environment with:

```powershell
uv sync
```

Alternatively, activate `.venv` and install the dependencies from
`pyproject.toml` with your preferred Python package manager.

## Train

Run the training script from the project root:

```powershell
& ".venv\Scripts\python.exe" train.py
```

On the first run, the script downloads the `Helsinki-NLP/opus_books` dataset,
builds `tokenizer_en.json` and `tokenizer_it.json`, and splits the data into
90% training and 10% validation sets. The default configuration uses:

| Setting | Value |
| --- | --- |
| Source language | English (`en`) |
| Target language | Italian (`it`) |
| Batch size | 8 |
| Sequence length | 350 |
| Model size | 512 dimensions |
| Epochs | 2 |
| Learning rate | `0.0001` |

The script selects CUDA when available and otherwise runs on CPU. It saves
checkpoints under `Helsinki-NLP/opus_books_weights/`. Because `preload` is set
to `latest`, a subsequent run resumes from the most recent checkpoint.

## Create a Model Directly

```python
from model import build_transformer

model = build_transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    src_seq_len=128,
    tgt_seq_len=128,
    d_model=512,
    N=6,
    h=8,
    dropout=0.1,
    d_ff=2048,
)
```

`Note` The vocabulary sizes and sequence lengths should match the tokenizers and
dataset used by the training configuration.