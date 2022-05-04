import sklearn
import torch

def encode_data(dataset, tokenizer, max_seq_length=256):
    temp = tokenizer(dataset["tweets"].tolist(), padding="max_length", truncation=True, max_length=max_seq_length, return_offsets_mapping=True, return_tensors = "pt")
    input_ids = temp["input_ids"]
    attention_mask = temp["attention_mask"]
    return input_ids, attention_mask

def extract_labels(dataset):
    return dataset["labels"].astype("int").tolist()

def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a
    transformers.trainer_utils.EvalPrediction object.
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    result = sklearn.metrics.precision_recall_fscore_support(labels,preds,average="binary")
    return dict({"accuracy":sklearn.metrics.accuracy_score(labels,preds,normalize=True), "precision":result[0], "recall":result[1], "f1":result[2]})
