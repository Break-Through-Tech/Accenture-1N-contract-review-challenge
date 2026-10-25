from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd

pd.options.mode.string_storage = "python"
pd.options.future.infer_string = False

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import chunking_util

DATA_PATH = chunking_util.DATA_PATH
NOTEBOOK_DIR = Path(chunking_util.__file__).resolve().parent
MODEL_NAME = chunking_util.MODEL_NAME
MAX_LENGTH = chunking_util.MAX_LENGTH
TARGET_CATEGORIES = 2


def build_chunk_label_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean_df = chunking_util.build_clean_df(DATA_PATH)
    required_columns = {
        "contract_id",
        "category",
        "is_impossible",
        "annotation_start",
        "annotation_text",
    }
    missing = sorted(required_columns - set(clean_df.columns))
    if missing:
        raise KeyError(f"clean_df is missing required columns: {missing}")

    token_len = chunking_util.get_token_len_fn(MODEL_NAME)
    chunker = chunking_util.build_chunker(token_len)
    chunks_df = chunking_util.build_chunks_df(chunker, clean_df)
    chunks_df = chunks_df[chunks_df["chunk_start"] >= 0].reset_index(drop=True)
    if chunks_df.empty:
        raise ValueError("No valid chunks were produced from the contract text.")

    chunk_labels_df = chunking_util.build_chunk_labels_df(clean_df, chunks_df)
    return clean_df, chunks_df, chunk_labels_df


def check_tokenizer_contract(tokenizer: AutoTokenizer, text: str = "The parties agree to comply.") -> dict:
    """Confirm the tokenizer contract ModernBERT expects for padded batches."""
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_attention_mask=True,
        return_tensors="pt",
    )
    assert "input_ids" in encoded and "attention_mask" in encoded
    assert encoded["input_ids"].shape[-1] == MAX_LENGTH
    assert encoded["attention_mask"].shape == encoded["input_ids"].shape
    return {
        "input_ids_shape": tuple(encoded["input_ids"].shape),
        "attention_mask_shape": tuple(encoded["attention_mask"].shape),
        "token_count": int(encoded["input_ids"].shape[-1]),
    }


def build_model_dataset(
    chunk_labels_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
    categories: list[str],
    sample_size: int = 32,
) -> tuple[Dataset, list[str]]:
    selected = list(dict.fromkeys(categories))
    if not selected:
        raise ValueError("No categories were provided for model training.")

    subset = chunk_labels_df[chunk_labels_df["category"].isin(selected)]
    if subset.empty:
        raise ValueError(f"No chunk-label rows found for categories: {selected}")

    pivot = subset.pivot_table(index="chunk_id", columns="category", values="label", aggfunc="max", fill_value=0)
    dataset_df = chunks_df[["chunk_id", "chunk_text"]].merge(pivot.reset_index(), on="chunk_id", how="inner")
    if dataset_df.empty:
        raise ValueError("No chunk rows survived the category merge.")

    dataset_df = dataset_df[["chunk_text", *selected]].fillna(0)
    dataset_df = dataset_df.sample(n=min(len(dataset_df), sample_size), random_state=0).reset_index(drop=True)
    return Dataset.from_pandas(dataset_df), selected


def tokenize_dataset(tokenizer: AutoTokenizer, dataset: Dataset, categories: list[str], max_length: int = MAX_LENGTH) -> Dataset:
    def _tokenize_batch(examples):
        # no padding here: padding inside dataset.map pads every row to the longest chunk in its
        # 1,000-row map batch (~750 tokens) and stores it that way. DataCollatorWithPadding pads
        # each training batch to its own longest chunk instead, so less compute goes to pad tokens.
        encoded = tokenizer(
            examples["chunk_text"],
            padding=False,
            truncation=False,
            return_attention_mask=True,
        )
        if any(len(input_ids) > max_length for input_ids in encoded["input_ids"]):
            raise ValueError(f"A chunk exceeds ModernBERT's {max_length}-token input limit")
        labels = [[float(examples[col][i]) for col in categories] for i in range(len(examples["chunk_text"]))]
        encoded["labels"] = labels
        return encoded

    tokenized = dataset.map(_tokenize_batch, batched=True, remove_columns=["chunk_text"], desc="tokenizing chunks")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def run_smoke_training() -> dict:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    check_tokenizer_contract(tokenizer)

    _, chunks_df, chunk_labels_df = build_chunk_label_frame()
    # pick the categories with the most positive chunks, so the tiny sample actually has some 1s in it
    positive_counts = chunk_labels_df.groupby("category")["label"].sum()
    candidate_categories = positive_counts[positive_counts > 0].sort_values(ascending=False).index.tolist()
    if not candidate_categories:
        raise ValueError("No positive chunk labels were produced for any category.")

    categories = candidate_categories[:TARGET_CATEGORIES]
    dataset, categories = build_model_dataset(chunk_labels_df, chunks_df, categories=categories, sample_size=32)
    tokenized = tokenize_dataset(tokenizer, dataset, categories)

    train_end = max(1, len(tokenized) // 2)
    train_ds = tokenized.select(range(train_end))
    eval_ds = tokenized.select(range(train_end, len(tokenized)))
    if len(eval_ds) == 0:
        eval_ds = train_ds

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(categories),
        problem_type="multi_label_classification",
    )
    model.config.id2label = {idx: label for idx, label in enumerate(categories)}
    model.config.label2id = {label: idx for idx, label in enumerate(categories)}

    training_args = dict(
        output_dir=str(NOTEBOOK_DIR / "tmp_smoke_train"),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="epoch",
        report_to=[],
        disable_tqdm=True,
    )
    if "overwrite_output_dir" in inspect.signature(TrainingArguments).parameters:
        training_args["overwrite_output_dir"] = True
    args = TrainingArguments(**training_args)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    train_result = trainer.train()
    eval_result = trainer.evaluate()
    return {
        "categories": categories,
        "train_rows": len(train_ds),
        "eval_rows": len(eval_ds),
        "train_loss": float(train_result.training_loss),
        "eval_loss": float(eval_result.get("eval_loss", 0.0)),
    }


if __name__ == "__main__":
    print(run_smoke_training())
