# 🇲🇲 Myanmar LLM Dataset


<p align="center">
  <img src="logo.svg" width="720" height="520" alt="Myanmar LLM Dataset Logo"/>
</p>


<p align="center">
  <strong>မြန်မာဘာသာစကား Large Language Model အတွက် Supervised Fine-Tuning (SFT) Dataset</strong>
</p>


<p align="center">
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" alt="Status"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Language-Myanmar-orange.svg" alt="Language"/></a>
</p>


---


## 📖 ခြုံငုံသုံးသပ်ချက်


ဤ dataset သည် မြန်မာဘာသာစကားဖြင့် လေ့ကျင့်သင်ကြားထားသော **Large Language Model (LLM)** များကို **Supervised Fine-Tuning (SFT)** ပြုလုပ်ရန်အတွက် တည်ဆောက်ထားခြင်းဖြစ်ပါသည်။


```mermaid
flowchart LR
    subgraph GitHub["GitHub Repository"]
        A["mm-llm-dataset<br/>train.jsonl / validation.jsonl / test.jsonl"]
    end


    subgraph HuggingFace["Hugging Face Hub"]
        B["myanmar-llm-dataset<br/>Dataset Card"]
    end


    subgraph User["End User"]
        C["load_dataset()<br/>Python Script"]
    end


    A -- "git push" --> B
    C -- "API call" --> B
    B -- "return dataset" --> C
```


---


📊 ဒေတာပါဝင်မှု အချိုးအစား


ခွဲခြမ်း နမူနာအရေအတွက် ရှင်းလင်းချက်
Train ၅ မော်ဒယ်အား သင်ကြားရန်
Validation ၅ သင်ကြားနေစဉ် စမ်းသပ်ရန်
Test ၅ အဆုံးသတ် အကဲဖြတ်ရန်


```mermaid
pie showData
    "Train (5)" : 5
    "Validation (5)" : 5
    "Test (5)" : 5
```


---


📁 ဖိုင်ပုံစံ (JSONL)


```jsonl
{"messages": [{"role": "system", "content": "သင်သည် အထောက်အကူပြု AI လက်ထောက်တစ်ဦးဖြစ်သည်။"}, {"role": "user", "content": "မင်္ဂလာပါ"}, {"role": "assistant", "content": "မင်္ဂလာပါ။ ဘယ်လိုကူညီရမလဲ"}], "metadata": {"source": "manual", "language": "my"}}
```


အဓိက အစိတ်အပိုင်းများ


Key Type Description
messages array စကားဝိုင်း အပိုင်းအစများ
role string system / user / assistant
content string စကားဝိုင်း အကြောင်းအရာ
metadata object နောက်ဆက်တွဲ အချက်အလက်


---


🚀 အသုံးပြုနည်း


Hugging Face datasets ဖြင့်


```python
from datasets import load_dataset


# dataset တစ်ခုလုံး load လုပ်ရန်
dataset = load_dataset("amkyawdev/myanmar-llm-dataset")


# split တစ်ခုချင်းစီ load လုပ်ရန်
train = load_dataset("amkyawdev/myanmar-llm-dataset", split="train")
valid = load_dataset("amkyawdev/myanmar-llm-dataset", split="validation")
test = load_dataset("amkyawdev/myanmar-llm-dataset", split="test")
```


GitHub Raw URL ဖြင့်


```python
from datasets import load_dataset


dataset = load_dataset(
    "json",
    data_files="https://raw.githubusercontent.com/amkyawdev/myanmar-llm-dataset/main/data/processed/train.jsonl"
)
```


---


📂 Project Structure


```mermaid
flowchart TD
    subgraph Repo["mm-llm-dataset"]
        direction TB
        A["README.md"]
        B["LICENSE"]
        C["data/"]
        D["scripts/"]
    end


    subgraph Data["data/"]
        E["train.jsonl"]
        F["validation.jsonl"]
        G["test.jsonl"]
    end


    subgraph Scripts["scripts/"]
        H["preprocess.py"]
        I["validate.py"]
    end


    C --> Data
    D --> Scripts
```


---


📜 လိုင်စင်


ဤ project သည် MIT License အောက်တွင် ဖြန့်ချိထားပါသည်။


```
MIT License


Copyright (c) 2024 Am Kyaw Dev


Permission is hereby granted, free of charge, to any person obtaining a copy
...
```


---


🤝 ပါဝင်ကူညီခြင်း


ဤ dataset ကို ပိုမိုကောင်းမွန်အောင် ပူးပေါင်းပါဝင်လိုပါက Pull Request သို့မဟုတ် Issue ဖွင့်နိုင်ပါသည်။


---


<p align="center">
  Made with ❤️ for Myanmar NLP Community
</p>
