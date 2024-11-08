!pip install transformers datasets evaluate

from huggingface_hub import notebook_login
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCasualLM, TrainingArguments, Trainer
import math
from transformers import create_optimizer, AdamWeightDecay
from transformers import TFAutoModelForCasualLM
import tensorflow as tf
from transformers.keras_callbacks import PushToHubCallback
from transformers import AutoTokenizer
from transformers import AutoModelForCasualLM



notebook_login()
