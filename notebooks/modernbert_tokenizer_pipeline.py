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
    token_len = chunking_util.get_token_len_fn(MODEL_NAME)
    chunker = chunking_util.build_chunker(token_len)
    chunks_df = chunking_util.build_chunks_df(chunker, clean_df)
    chunks_df = chunks_df[chunks_df["chunk_start"] >= 0].reset_index(drop=True)

    contract_categories = clean_df[["contract_id", "category", "is_impossible"]].drop_duplicates()
    chunk_labels_df = contract_categories.merge(chunks_df[["contract_id", "chunk_id"]], on="contract_id")

    positives = []
    evidenced = clean_df[~clean_df["is_impossible"]].copy()
    if not evidenced.empty:
        evidenced["annotation_end"] = evidenced["annotation_start"].astype(int) + evidenced["annotation_text"].str.len()
        for contract_id, group in evidenced.groupby("contract_id"):
            contract_chunks = chunks_df[(chunks_df["contract_id"] == contract_id) & (chunks_df["chunk_start"] >= 0)]
            for _, row in group.iterrows():
                start = int(row["annotation_start"])
                end = int(row["annotation_end"])
                overlaps = contract_chunks[
                    (contract_chunks["chunk_start"] < end) & (contract_chunks["chunk_end"] > start)
                ]
                for chunk_id in overlaps["chunk_id"].tolist():
                    positives.append({"chunk_id": chunk_id, "category": row["category"]})

    positive_keys = pd.DataFrame(positives, columns=["chunk_id", "category"]).drop_duplicates()
    positive_keys["label"] = 1
    chunk_labels_df = chunk_labels_df.merge(positive_keys, on=["chunk_id", "category"], how="left")
    chunk_labels_df["label"] = chunk_labels_df["label"].fillna(0).astype(int)
    chunk_labels_df = chunk_labels_df[["contract_id", "chunk_id", "category", "is_impossible", "label"]]

    assert (chunk_labels_df.loc[chunk_labels_df["is_impossible"], "label"] == 0).all(), (
        "is_impossible categories must never get positive chunk labels"
    )
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


def build_model_dataset(chunk_labels_df: pd.DataFrame, chunks_df: pd.DataFrame, categories: list[str], sample_size: int = 32) -> tuple[Dataset, list[str]]:
    selected = list(dict.fromkeys(categories))
    subset = chunk_labels_df[chunk_labels_df["category"].isin(selected)]
    pivot = subset.pivot_table(index="chunk_id", columns="category", values="label", aggfunc="max", fill_value=0)
    dataset_df = chunks_df[["chunk_id", "chunk_text"]].merge(pivot.reset_index(), on="chunk_id", how="inner")
    dataset_df = dataset_df[["chunk_text", *selected]].fillna(0)
    dataset_df = dataset_df.sample(n=min(len(dataset_df), sample_size), random_state=0).reset_index(drop=True)
    return Dataset.from_pandas(dataset_df), selected


def tokenize_dataset(tokenizer: AutoTokenizer, dataset: Dataset, categories: list[str], max_length: int = MAX_LENGTH) -> Dataset:
    def _tokenize_batch(examples):
        encoded = tokenizer(
            examples["chunk_text"],
            padding=True,
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
    categories = list(chunk_labels_df["category"].drop_duplicates().sort_values())[:TARGET_CATEGORIES]
    dataset, categories = build_model_dataset(chunk_labels_df, chunks_df, categories=categories, sample_size=32)
    tokenized = tokenize_dataset(tokenizer, dataset, categories)

    train_ds = tokenized.select(range(max(1, len(tokenized) // 2)))
    eval_ds = tokenized.select(range(max(1, len(tokenized) // 2), len(tokenized)))
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
