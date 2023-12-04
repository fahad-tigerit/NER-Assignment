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

tokenizer = BertTokenizerFast.from_pretrained("tokenizer_only_train")

model_fine_tuned = AutoModelForTokenClassification.from_pretrained(
    "ner_model_only_train"
)

nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer)

# %%
dataset = load_dataset("tuetschek/atis")
test_dataset = dataset["train"]
test_texts = test_dataset["text"]
print(type(test_dataset))


# %%
print(len(test_texts))
for lines in test_texts:
    example = lines
    ner_results = nlp(example)
    for x in ner_results:
        # print(x["entity"], x["word"])
        if "B-return" in x["entity"] or "I-return" in x["entity"]:
            print(x["entity"], x["word"])

# %%
# import transformers
# print(transformers.__version__)

example = "i need a return flight from philadelphia to boston"
ner_results = nlp(example)
for x in ner_results:
    # print(x["entity"], x["word"])

    print(x["entity"], x["word"])
    # print(type(x))
