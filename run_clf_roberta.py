# -*- coding: utf-8 -*-

'''Setups'''
import os
# import zipfile
import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from senti import SentiDataset
from utils import compute_metrics

print("code start")

# GPU detection
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# OS setup for GPUs
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

# discarded code
'''
!wget https://raw.githubusercontent.com/huggingface/transformers/main/tests/deepspeed/ds_config_zero3.json


with zipfile.ZipFile("/scratch/yl5865/ds301_final_project/trainingandtestdata.zip","r") as zip_ref:
    zip_ref.extractall("/scratch/yl5865/ds301_final_project/data/")

# Read-in the Training Tweets
raw_data_df = pd.read_csv("/scratch/yl5865/ds301_final_project/data/training.1600000.processed.noemoticon.csv",encoding="ISO-8859-1")
raw_data_df.columns = ["labels","ids","date","query","user","tweets"]
data_df = raw_data_df.drop(["ids","date","query","user"],axis=1)
data_df["labels"] = data_df["labels"]/4
TOTAL = data_df.shape[0]
TTSPLIT_FACTOR = 0.1

# small_data_df = data_df.sample(1500, random_state=7)
# small_train_data_df = small_data_df.iloc[:1300,:]
# small_val_data_df = small_data_df.iloc[-200:,:]
# small_val_data_df
data_df = data_df.sample(TOTAL, random_state=7)
train_data_df = data_df.iloc[:int(TOTAL*(1-TTSPLIT_FACTOR)),:]
val_data_df = data_df.iloc[-int(TOTAL*(1-TTSPLIT_FACTOR)):,:]

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# small_train_data = SentiDataset(small_train_data_df,tokenizer)
# small_val_data = SentiDataset(small_val_data_df,tokenizer)
train_data = SentiDataset(train_data_df,tokenizer)
val_data = SentiDataset(val_data_df,tokenizer)
'''
from datasets import load_dataset
dataset = load_dataset("sentiment140")

print("data loaded")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

print("right after tokenize")
data = tokenized_datasets["train"]
data = data.map(lambda example: {'labels':1} if example['sentiment']==4 else {"labels":0}, remove_columns=["date","user","query","sentiment","text"])
SPLIT_FACTOR = 0.85
TOTAL_TRAIN = len(data)

data = data.shuffle(seed=7)

train_dataset = data.select(range(0,int(TOTAL_TRAIN*SPLIT_FACTOR)))
eval_dataset = data.select(range(int(TOTAL_TRAIN*SPLIT_FACTOR),TOTAL_TRAIN))

print("right before Classifier")

'''Train the Classifier'''
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)

if device is "cpu":
  # when cpu is used
  os.environ.clear()
  trainArgs = TrainingArguments(evaluation_strategy="epoch",
                          save_strategy="epoch",
                          num_train_epochs=3,
                          load_best_model_at_end=True,
                          output_dir="/scratch/yl5865/ds301_final_project/model",
                          output_dir="test_trainer")
else:
  # when gpu is used
  trainArgs = TrainingArguments(evaluation_strategy="epoch",
                          save_strategy="epoch",
                          num_train_epochs=3,
                          per_device_train_batch_size=32,
                          per_device_eval_batch_size=16,
                          load_best_model_at_end=True,
                          output_dir="/scratch/yl5865/ds301_final_project/model",
                          deepspeed="/scratch/yl5865/ds301_final_project/configs/ds_config_zero3.json")

trainer = Trainer(model=model,
                  args=trainArgs,
                  train_dataset=train_dataset,
                  eval_dataset=eval_dataset,
                  compute_metrics=compute_metrics)

print("right before trainer started")

trainer.train()
trainer.save_pretrained('/scratch/yl5865/ds301_final_project/tuned_roberta_model')
