# News Classifier using BERT

Fine-tunes **BERT** (`bert-base-uncased`) on the **[AG News](https://huggingface.co/datasets/ag_news)** corpus for **4-way news topic classification**: World, Sports, Business, and Sci/Tech.

## Features

- Loads AG News via [Hugging Face `datasets`](https://huggingface.co/docs/datasets)
- Optional **balanced subset** of the training split (default-oriented toward faster runs on CPU)
- **Hugging Face `Trainer`** with `DataCollatorWithPadding` for training
- Metrics: **accuracy** and **weighted F1**
- Exploratory plots for class distribution (matplotlib / seaborn)
- Checkpoints and logs under `./results` and `./logs`

## Project layout

| Path | Description |
|------|-------------|
| `code.ipynb` | Main notebook: data prep, training, evaluation |
| `results/` | Training output (`checkpoint-*`), `eval_results.json` when evaluation is saved |
| `.gitignore` | Ignores large model binaries (`.safetensors`, `.bin`, etc.) |

## Requirements

- Python 3.10+ (notebook metadata references Python 3.12)
- PyTorch with CUDA optional (the notebook selects `cuda` when available, otherwise `cpu`)

Install dependencies (example):

```bash
pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn tqdm accelerate
```

A working internet connection is needed the first time you load `ag_news` and the pretrained BERT weights from Hugging Face.

## How to run

1. Open `code.ipynb` in Jupyter, VS Code, or Cursor.
2. **Run cells in order from the top.**  
   The training section expects variables such as `model_name`, `tokenizer`, and `tokenized_datasets` to exist (standard pattern: load tokenizer, map `dataset` with a `tokenize_function`, remove the `text` column, rename `label` → `labels`, then `set_format` for PyTorch). If your copy of the notebook is missing those cells, add them before the `Trainer` cell.
3. Adjust **`TRAIN_SIZE`** and **`TEST_SIZE`** in the data-loading cell to trade off training time vs. data coverage (larger `TRAIN_SIZE` improves potential quality but increases runtime, especially on CPU).

### Training settings (as in the notebook)

- **Model:** `bert-base-uncased`, `BertForSequenceClassification` with `num_labels=4`
- **Optimizer-related:** learning rate `2e-5`, weight decay `0.01`
- **Batches:** train/eval batch size `8` per device (tuned for lighter hardware)
- **Epochs:** `2`
- **Output:** `output_dir="./results"`, evaluation and save per epoch, best checkpoint kept by **accuracy**

After training, the best checkpoint is under `results/checkpoint-*` (exact folder name depends on training steps).

### Optional: manual evaluation with fixed-length padding

The notebook includes a follow-up path that re-tokenizes with `padding="max_length"` and `max_length=128` for consistent tensor shapes when batching outside the `Trainer` collator, and can write metrics to `results/eval_results.json`.

## Example results

With the saved `results/eval_results.json` in this repo (500-sample test subset, reduced training setup), reported metrics were approximately:

- **Accuracy:** 0.89  
- **Weighted F1:** ~0.89  

Your numbers will vary with `TRAIN_SIZE`, `TEST_SIZE`, random seed, and hardware.

## Notes

- Large weight files are **gitignored**; clone the repo and train locally (or download checkpoints separately) to reproduce full artifacts.
- First-time dataset and model downloads can take several minutes depending on bandwidth.

## License

Refer to the licenses of [AG News / the dataset card on Hugging Face](https://huggingface.co/datasets/ag_news), [transformers](https://github.com/huggingface/transformers), and PyTorch for redistribution and use terms.
