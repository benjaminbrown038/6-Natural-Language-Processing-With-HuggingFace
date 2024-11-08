!pip install transformers datasets evaluate

from huggingface_hub import notebook_login

notebook_login()

from datasets import load_datset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM
import math
from transformers import create_optimizer, AdamWeightDecay
from transformers import TFAutoModelForMaskedLM
import tensorflow as tf
from transformers.keras_callbacks import PushToHubCallback
from transformers import pipeline
