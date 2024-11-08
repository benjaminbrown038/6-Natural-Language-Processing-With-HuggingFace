from huggingface_hub import notebook_login
notebook_login()

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
import evaluate
import numpy as np
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import create_optimizer
from transformers import TFAutoModelForTokenClassification
import tensorflow as tf
from transformers.keras_callbacks import KerasMetricCallback
from transformers.keras_callbacks import PushToHubCallback
from transformers import pipeline
from transformers import AutoModelForTokenClassification

wnut = load_dataset("wnut_17")
wnut["train"][0]

label_list = wnut["train"].features[f"ner_tags"].feature.names
label_list

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
example = wnut["train"][0]
tokenized_input = tokenizer
tokens

def tokenize_and_align_labels():
    return tokenized_inputs

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors = "tf")

seqeval = eval.load("seqeval")
lavels = 
