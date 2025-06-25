# AG News Classification with DistilBERT

## Project Task
Fine-tune a pre-trained transformer (DistilBERT) to classify news headlines & descriptions into one of four topics:  
- **World**  
- **Sports**  
- **Business**  
- **Sci/Tech**

## Dataset
We use the **AG News** corpus (M. Jabreel & Q. Zhang), publicly available in CSV form:  
- **Train**: 120,000 articles  
- **Test**: 7,600 articles  
Each example has a _title_ and a _description_, which we merge into a single `text` field, then shift labels from 1–4 → 0–3.

## Pre-trained Model
We fine-tune **DistilBERT-base-uncased** (a smaller, faster BERT) via Hugging Face’s Transformers:  
- Tokenizer: `DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")`  
- Model: `DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", config=...)`  
- ID ↔ label mapping injected via `id2label={0:"World",1:"Sports",2:"Business",3:"Sci/Tech"}`.

## Performance Metrics
We report on:  
- **Accuracy**  
- **Weighted Precision / Recall / F1**  
- **Per-class classification report**  

| Split           | Accuracy | Precision | Recall | F1    |
|-----------------|:--------:|:---------:|:------:|:-----:|
| 100×20 smoke-test | 0.45     | 0.33      | 0.45   | 0.35  |
| Full train/test (3 epochs) | ~0.93     | ~0.93      | ~0.93   | ~0.93  |




## Hyperparameters
- **Max sequence length**: 128 tokens  
- **Batch size**: 16 train / 32 eval (smoke-test uses 8/16)  
- **Epochs**: 3 (smoke-test uses 1)  
- **Learning rate**: 5 × 10⁻⁵ (default)  
- **Logging steps**: every 100 steps  
- **Save limit**: keep last 2 checkpoints  

---

## How to Run

1. Click the badge to open in Colab and run **2-representation.ipynb** :

   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ipeakk/LHL_LLM_Project_/blob/main/notebooks/2-representation.ipynb)

2. Locally:

   jupyter notebook notebooks/1-preprocessing.ipynb
   jupyter notebook notebooks/2-representation.ipynb
