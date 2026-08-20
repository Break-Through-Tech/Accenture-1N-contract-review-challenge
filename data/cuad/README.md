# CUAD v1 dataset

This directory contains the machine-readable release of the Contract Understanding
Atticus Dataset (CUAD) v1. CUAD includes 510 commercial contracts annotated for 41
contract-review categories. The data is structured as SQuAD-style JSON and does not
require parsing the original PDFs.

## Files

| File | Contents |
| --- | --- |
| `CUADv1.json` | Complete dataset: 510 contracts and 13,823 annotated answer spans |
| `train_separate_questions.json` | Prepared training split: 408 contracts |
| `test.json` | Prepared test split: 102 contracts |
| `category_descriptions.csv` | Names, descriptions, answer formats, and groups for the 41 categories |
| `cuad-v1.zip` | Original `data.zip` release archive from the official repository |

Each JSON file has top-level `version` and `data` fields. Each item in `data` has a
contract `title` and one or more `paragraphs`; a paragraph contains the contract text
in `context` and annotations in `qas`.

## Provenance

- Dataset: CUAD v1 (NeurIPS 2021)
- Publisher: The Atticus Project
- Dataset page: <https://www.atticusprojectai.org/cuad/>
- Official repository: <https://github.com/TheAtticusProject/cuad>
- Release archive: <https://raw.githubusercontent.com/TheAtticusProject/cuad/master/data.zip>
- Downloaded: 2026-08-10
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

SHA-256 of the original archive:

```text
f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a  cuad-v1.zip
```

When publishing work based on these files, attribute The Atticus Project and cite the
CUAD paper as requested by the official repository.
