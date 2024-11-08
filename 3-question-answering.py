!pip install transformers datasets evaluate

from huggingface_hub import notebook_login
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import create_optimizer
from transformers import TFAutoModelForQuestionAnswering
import tensorflow as tf
from transformers import pipeline
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForQuestionAnswering


notebook_login()

