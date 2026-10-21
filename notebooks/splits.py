"""
Contract-level train / tune / validation split for everything trained on train_separate_questions.json.

Same two-step GroupShuffleSplit (random_state=0) that threshold_tuning.ipynb runs on its chunk
table: 80/20 into pool vs. validation, then 80/20 of the pool into train vs. tune. GroupShuffleSplit
only looks at the sorted set of contract ids, so running it on a plain contract list gives exactly
the same contracts as running it on chunks -- which means the split doesn't move when chunking does.

test.json is never touched here; it's reserved for the final held-out evaluation.

Writing the lists to data/splits/ means the transformer runs (local or Colab) read the split from a
file instead of re-deriving it, and don't need the chunking stack installed just to know which
contract goes where. Kept dependency-light on purpose (json + sklearn only) for that reason.

Run it from notebooks/ (same convention as the rest of the repo):

    cd notebooks
    python splits.py
"""

import json
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

REPO_DIR = Path(__file__).resolve().parent.parent
TRAIN_JSON = REPO_DIR / "data" / "cuad" / "train_separate_questions.json"
SPLITS_DIR = REPO_DIR / "data" / "splits"
SPLIT_NAMES = ("train", "tune", "val")
EXPECTED_SIZES = {"train": 260, "tune": 66, "val": 82}
RANDOM_STATE = 0


def load_train_contract_ids(path=TRAIN_JSON) -> list[str]:
    """Contract ids (the `title` field) of the 408 official training contracts, sorted."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    ids = sorted({doc["title"] for doc in raw["data"]})
    assert len(ids) == 408, f"expected 408 training contracts, found {len(ids)}"
    return ids


def build_contract_splits(contract_ids: list[str]) -> dict[str, list[str]]:
    # split 1 -- pool vs. validation, same call as main_pipeline.ipynb / threshold_tuning.ipynb
    ids = sorted(contract_ids)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    pool_idx, val_idx = next(splitter.split(ids, groups=ids))
    pool = [ids[i] for i in pool_idx]
    val = [ids[i] for i in val_idx]

    # split 2 -- tune set carved out of the pool, same call as threshold_tuning.ipynb
    tune_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, tune_idx = next(tune_splitter.split(pool, groups=pool))
    splits = {
        "train": sorted(pool[i] for i in train_idx),
        "tune": sorted(pool[i] for i in tune_idx),
        "val": sorted(val),
    }
    check_splits(splits)
    return splits


def check_splits(splits: dict[str, list[str]]) -> None:
    """Loud checks: expected sizes, no contract in two splits, nothing dropped."""
    sizes = {name: len(splits[name]) for name in SPLIT_NAMES}
    if sizes != EXPECTED_SIZES:
        raise ValueError(f"split sizes {sizes} don't match the expected {EXPECTED_SIZES}")
    train, tune, val = (set(splits[name]) for name in SPLIT_NAMES)
    if (train & tune) or (train & val) or (tune & val):
        raise ValueError("a contract appears in more than one split -- that's leakage")
    if len(train | tune | val) != 408:
        raise ValueError("splits don't add up to all 408 training contracts")


def write_splits(splits: dict[str, list[str]], splits_dir=SPLITS_DIR) -> None:
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        (splits_dir / f"{name}_contracts.txt").write_text("\n".join(splits[name]) + "\n", encoding="utf-8")


def load_splits(splits_dir=SPLITS_DIR) -> dict[str, list[str]]:
    """Read the saved split files back (one contract id per line) and re-run the same checks."""
    splits_dir = Path(splits_dir)
    splits = {}
    for name in SPLIT_NAMES:
        path = splits_dir / f"{name}_contracts.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run `python splits.py` from notebooks/ first")
        splits[name] = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    check_splits(splits)
    return splits


def main():
    splits = build_contract_splits(load_train_contract_ids())
    write_splits(splits)
    # round-trip through the files so we know what other code will actually read
    assert load_splits() == splits, "split files don't read back to the same contracts"
    for name in SPLIT_NAMES:
        print(f"{name:>5}: {len(splits[name])} contracts -> {SPLITS_DIR / f'{name}_contracts.txt'}")


if __name__ == "__main__":
    main()
