!pip install transformers datasets evaluate accelerate

from huggingface_hub import notebook_login
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import create_optimizer
from transformers import TFAutoModelForSequenceClassification
from transformers import create_optimizer
import tensorflow as tf
import evaluate
import numpy as np








imbd = load_dataset("imbd")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_imdb = imdb.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions,axis=1)
    return accuracy.compute(predictions = predictions, references=labels)

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id)

training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics)

trainer.train()
trainer.push_to_hub()

batch_size = 16
num_epochs = 5
batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

model = TFAutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id)

tf_train_set = model.prepare_tf_dataset(
    tokenized_imdb["train"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator)

tf_validation_set = model.prepare_tf_dataset(
    tokenized_imdb["test"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator)

model.compile(optimizer=optimizer) 

metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
push_to_hub_callback = PushToHubCallback(output_dir="my_awesome_model", tokenizer=tokenizer)

callbacks = [metric_callback, push_to_hub_callback]
model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3, callbacks=callbacks)

text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
classifier(text)

tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
inputs = tokenizer(text, return_tensors="pt")

model = AutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")

with torch.no_grad():
    logits = model(**inputs).logits
    
predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]


tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
inputs = tokenizer(text, return_tensors="tf")

model = TFAutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")
logits = model(**inputs).logits

predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
model.config.id2label[predicted_class_id]
