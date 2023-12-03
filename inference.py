# %%
import torch
from transformers import (
    BertTokenizer,
    BertTokenizerFast,
    BertForTokenClassification,
    AdamW,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
)
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import datasets
import numpy as np
from transformers import pipeline

tokenizer = BertTokenizerFast.from_pretrained("tokenizer")

model_fine_tuned = AutoModelForTokenClassification.from_pretrained("ner_model")

nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer)

# %%
# import transformers
# print(transformers.__version__)

example = "how about a flight from milwaukee to st. louis that leaves monday night"
ner_results = nlp(example)
for x in ner_results:
    print(x["entity"], x["word"])
    # print(type(x))

print("DONE")

# %%
