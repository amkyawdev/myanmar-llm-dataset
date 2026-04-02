# Myanmar LLM Dataset

<p align="center">
  <img src="logo.svg" width="120" height="120" alt="Logo"/>
</p>

မြန်မာဘာသာစကား Large Language Model အတွက် SFT ဒေတာအစုအပါ

## ဒေတာပါဝင်မှု

- သင်ကြားရန် (Train): နမူနာ ၅
- စမ်းသပ်ရန် (Validation): နမူနာ ၅
- အဆုံးသတ်စမ်းသပ်ရန် (Test): နမူနာ ၅

## ဖိုင်ပုံစံ

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {...}
}
```

## အသုံးပါ

```python
from datasets import load_dataset

dataset = load_dataset("amkyawdev/myanmar-llm-dataset")
```

## လိုင်စေးစနစ်

MIT License