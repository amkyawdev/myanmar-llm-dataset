# myanmar-llm-dataset

Myanmar Language Model Instruction Dataset for fine-tuning LLM models.

## 📋 Overview

This dataset contains Myanmar language text data for training and evaluating large language models. It is designed for instruction fine-tuning of Myanmar language models.

## 📁 Dataset Files

```
data/processed/
├── train.jsonl       # Training data (20 samples)
├── validation.jsonl  # Validation data (5 samples)
└── test.jsonl       # Test data (3 samples)
```

## 🔗 Dataset Format

Each file is in JSONL format (JSON Lines):

```json
{"text": "မင်္ဂလာပါ။ မြန်မာစာနဲ့ပတ်လုံးကို ကူညီပါရန်။"}
```

## 📥 Loading Dataset

### From Hugging Face

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": "path/to/train.jsonl",
    "validation": "path/to/validation.jsonl",
    "test": "path/to/test.jsonl"
})
```

### From GitHub Raw URL

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files={
        "train": "https://raw.githubusercontent.com/amkyawdev/myanmar-llm-dataset/main/data/processed/train.jsonl",
        "validation": "https://raw.githubusercontent.com/amkyawdev/myanmar-llm-dataset/main/data/processed/validation.jsonl",
        "test": "https://raw.githubusercontent.com/amkyawdev/myanmar-llm-dataset/main/data/processed/test.jsonl"
    }
)
```

## 📊 Dataset Statistics

| Split | Size |
|-------|------|
| Train | 20 samples |
| Validation | 5 samples |
| Test | 3 samples |

## 📝 License

MIT License

---

*Dataset created by amkyawdev*