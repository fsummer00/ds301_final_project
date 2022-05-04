import torch
from torch.utils.data import Dataset
from utils import encode_data, extract_labels

class SentiDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_seq_length=256):
        self.encoded_data = encode_data(dataframe,tokenizer,max_seq_length=max_seq_length)
        self.label_list = extract_labels(dataframe)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, i):
        dic = {}
        dic["input_ids"] = self.encoded_data[0][i]
        dic["attention_mask"] = self.encoded_data[1][i]
        dic["labels"] = self.label_list[i]
        return dic
