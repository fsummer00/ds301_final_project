# -*- coding: utf-8 -*-

'''Setups'''
import os
import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from senti import SentiDataset
from utils import compute_metrics

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
!wget http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip

import zipfile
with zipfile.ZipFile("trainingandtestdata.zip","r") as zip_ref:
    zip_ref.extractall("./data/")
'''


'''Read-in the Training Tweets'''
raw_data_df = pd.read_csv("data/training.1600000.processed.noemoticon.csv",encoding="ISO-8859-1")
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


'''Prepare the Dataset'''
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# small_train_data = SentiDataset(small_train_data_df,tokenizer)
# small_val_data = SentiDataset(small_val_data_df,tokenizer)
train_data = SentiDataset(train_data_df,tokenizer)
val_data = SentiDataset(val_data_df,tokenizer)


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
                          load_best_model_at_end=True,
                          output_dir="/scratch/yl5865/ds301_final_project/model",
                          deepspeed="/scratch/yl5865/ds301_final_project/configs/ds_config_zero3.json")

trainer = Trainer(model=model,
                  args=trainArgs,
                  train_dataset=train_data,
                  eval_dataset=val_data,
                  compute_metrics=compute_metrics)

trainer.train()
trainer.save_model('/scratch/yl5865/ds301_final_project/tuned_roberta_model')
