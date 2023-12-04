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

dataset = load_dataset("tuetschek/atis")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

custom_tokens = ["o'clock", "st."]
tokenizer.add_tokens(custom_tokens)

unique_labels = set()

for example in dataset["train"]["slots"]:
    tokens = example.split(" ")
    for label in tokens:
        unique_labels.add(label)

for example in dataset["test"]["slots"]:
    tokens = example.split(" ")
    for label in tokens:
        unique_labels.add(label)

num_labels = len(unique_labels)
print(num_labels)

lookup_table = dict()
for i, x in enumerate(unique_labels):
    lookup_table[x] = i
label_list = {value: key for key, value in lookup_table.items()}


def tokenize_and_align_labels(examples, label_all_tokens=True):
    tokenized_inputs = tokenizer(examples["text"], truncation=True)
    labels = []

    for i, label in enumerate(examples["slots"]):
        label = label.split(" ")
        indexes = []
        temp_tokens = examples["text"][i].split(" ")
        for j in range(len(temp_tokens)):
            if temp_tokens[j][0] == "'":
                indexes.append(j)

        for k in range(len(indexes)):
            indexes[k] += k
            label.insert(indexes[k], "O")

        for j in range(len(label)):
            label[j] = lookup_table[label[j]]

        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)

            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_datasets = dataset["train"].map(tokenize_and_align_labels, batched=True)
tokenized_datasets_test = dataset["test"].map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = datasets.load_metric("seqeval")


def compute_metrics(eval_preds):
    pred_logits, labels = eval_preds

    pred_logits = np.argmax(pred_logits, axis=2)
    predictions = [
        [
            label_list[eval_preds]
            for (eval_preds, l) in zip(prediction, label)
            if l != -100
        ]
        for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
        [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]
    results = metric.compute(predictions=predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# eval_dataset=tokenized_datasets["validation"],
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased", num_labels=num_labels
)
model.resize_token_embeddings(len(tokenizer))

args = TrainingArguments(
    "test-ner",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    num_train_epochs=15,
    weight_decay=0.01,
)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets_test,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
model.save_pretrained("ner_model")
tokenizer.save_pretrained("tokenizer")
id2label = {str(i): label for i, label in enumerate(lookup_table)}
label2id = {label: str(i) for i, label in enumerate(lookup_table)}

from transformers import pipeline
import json


config = json.load(open("ner_model/config.json"))
config["id2label"] = id2label
config["label2id"] = label2id
json.dump(config, open("ner_model/config.json", "w"))

# %%


model_fine_tuned = AutoModelForTokenClassification.from_pretrained("ner_model")
nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer)

# %%

example = "what 's the cheapest flight from san francisco to milwaukee"

ner_results = nlp(example)
for x in ner_results:
    print(x["entity"], x["word"])
    # print(type(x))

print("DONE")
