!pip install transformers datasets evaluate
from huggingface_hub import notebook_login
from datasets import load_dataset
from transformers import AutoTokenizer
from dataclasses import dataclass 
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union 
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
import tensorflow as tf
import evaluate
import numpy as np
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers import create_optimizer
from transformers import TFAutoModelForMultipleChoice
from transformers.keras_callbacks import KerasMetricCallback



