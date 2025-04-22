import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import math


# DEVICE SETUP
######################
#CHANGE TO COMMENTED LINE POR FAVOR (also delete these ass comments (not the useful ones))
#############
device = 'cpu'  #torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess the fat amount of data
# CSV FILE PATHS
#############
#CHANGE TO FILE PATHES YOU CAN KEEP EVERYTHING AFTER NLP BUT NOT NLP
##########################
train_csv_path = "/Users/robbie/Desktop/NLP/NLP-Project-Group-3/data/one_hot_targets/train_data.csv"  # Update if needed
test_csv_path  = "/Users/robbie/Desktop/NLP/NLP-Project-Group-3/data/one_hot_targets/test_data.csv"   # Update if needed

# marco del datos
df_train = pd.read_csv(train_csv_path)
df_test = pd.read_csv(test_csv_path)

# this is just for something later i need a way to identify them and it cant just be 
# the date bc some are same so it also need ticker and this was easiest
df_train["transcript_id"] = df_train["file_path"].apply(lambda x: os.path.basename(x))
df_test["transcript_id"] = df_test["file_path"].apply(lambda x: os.path.basename(x))


def load_transcript(file_path):
    ####################
    #CHANGE THE F STRING TO HAVE WHEREVER UR THING IS
    #KEEP EVERYTHING AFTER NLP BTU NOT NLP
    ##################
    with open(f'/Users/robbie/Desktop/NLP/NLP-Project-Group-3/{file_path}', "r", encoding="utf-8") as f:
        return f.read()

# preload the text bc the map function is trash and so is my load transcript but blame el mapo
# basically it stops it from constantly opening files and wasting everyones time
df_train["text"] = df_train["file_path"].apply(load_transcript)
df_test["text"] = df_test["file_path"].apply(load_transcript)

# rename for label
df_train = df_train.rename(columns={"combined_label": "labels"})
df_test  = df_test.rename(columns={"combined_label": "labels"})

# stupid hugging face gotta have they own stuff
dataset_train = Dataset.from_pandas(df_train)
dataset_test = Dataset.from_pandas(df_test)


# Token and chunk section (must precompute before training)
# model name
#####################
#FIRST MODEL IS (aparently) BETTER BUT SECOND SEEMS TO RUN FASTER
#SEE HOW LONG FIRST FEW ITERATIONS TAKE AND DECIDE WHICH TO USE
############################################
model_name = "yiyanghkust/finbert-tone" # "yiyanghkust/finbert-tone" #"ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_chunk(batch):
    output_examples = {"input_ids": [], "attention_mask": [], "labels": [], "transcript_id": []}
    
    for text, label, transcript_id in zip(batch["text"], batch["labels"], batch["transcript_id"]):
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=512,
            stride=128,
            return_overflowing_tokens=True,
            return_special_tokens_mask=False,
        )
        for input_ids, attention_mask in zip(encoding["input_ids"], encoding["attention_mask"]):
            output_examples["input_ids"].append(input_ids)
            output_examples["attention_mask"].append(attention_mask)
            output_examples["labels"].append(label)
            output_examples["transcript_id"].append(transcript_id)
    
    return output_examples

# hate these columns fr
columns_to_remove = ["file_path", "text", "surprise_pct", "volatility_change", "report_date", "ticker"]

# places to save the datasets so we dont need to reload everytime
###################
#CHANGE TO WHEREVER YOU WANT TO KEEP TEH TOKEN DATA WHEN RUNNING
###################################
train_dataset_path = "/Users/robbie/Desktop/NLP/processed_train_dataset"
test_dataset_path  = "/Users/robbie/Desktop/NLP/processed_test_dataset"

# processs + save
if not os.path.exists(train_dataset_path):
    dataset_train_chunked = dataset_train.map(
        tokenize_and_chunk,
        batched=True,
        batch_size=8,      # INCREASE THIS OR LOWER IT DEPENDING ON HOW LONG THINGS TAKE
        remove_columns=columns_to_remove,
        ##################################
        #INCREASE NUM_PROC IF U CAN JUST FIND OUT HOW MANY CPU CORES YOU GOT
        #####################
        num_proc=8,        
        load_from_cache_file=False
    )
    # this part actually saves the stuff
    dataset_train_chunked.save_to_disk(train_dataset_path)
else:
    # this loads the stuff if it exists hopefully it does
    dataset_train_chunked = load_from_disk(train_dataset_path)

# same shit different story
if not os.path.exists(test_dataset_path):
    dataset_test_chunked = dataset_test.map(
        tokenize_and_chunk,
        batched=True,
        batch_size=8,       # INCREASE THIS OR LOWER IT DEPENDING ON HOW LONG THINGS TAKE
        remove_columns=columns_to_remove,
        ###########################
        #INCREASE NUM_PROC IF U CAN JUST FIND OUT HOW MANY CPU CORES YOU GOT
        #####################
        num_proc=8,
        load_from_cache_file=False
    )
    #saving stuff to file
    dataset_test_chunked.save_to_disk(test_dataset_path)
else:
    #loading stuff if it there
    dataset_test_chunked = load_from_disk(test_dataset_path)

# apartently we gotta set they own dataset class to be a certain way bc it bugs out otherwise
# it just makes the dataset be of pytorch tensors
dataset_train_chunked.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "transcript_id"])
dataset_test_chunked.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "transcript_id"])

counts = {
    lbl: dataset_train_chunked.filter(lambda ex, lbl=lbl: ex["labels"] == lbl).num_rows
    for lbl in range(4)
}
max_count = max(counts.values())

balanced_splits = []
print(counts)
for lbl, cnt in counts.items():
    # pull out all examples of this label
    ds_lbl = dataset_train_chunked.filter(lambda ex, lbl=lbl: ex["labels"] == lbl)
    if cnt < max_count:
        # repeat until we have at least max_count, then truncate
        reps = math.ceil(max_count / cnt)
        ds_lbl = concatenate_datasets([ds_lbl] * reps).select(range(max_count))
    balanced_splits.append(ds_lbl)
dataset_train_balanced = concatenate_datasets(balanced_splits).shuffle(seed=42)
# Training time 

# Load model 
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4, ignore_mismatched_sizes=True)
model.to(device)

# train args 
###################
#IF YOU CHANGED BATCH SIZE ABOVE I THINK DO SAME HERE
##################
training_args = TrainingArguments(
    output_dir="./finbert_chunked_output",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    warmup_ratio=0.1,                
    weight_decay=0.01,
)

# Define metric
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
# precision_metric = evaluate.load("precision")
# recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    # precision = precision_metric.compute(predictions=predictions, references=labels, average="macro")
    # recall = recall_metric.compute(predictions=predictions, references=labels, average="macro")
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        # "precision": precision["precision"],
        # "recall": recall["recall"]
    }

# I hate this trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train_balanced,
    eval_dataset=dataset_test_chunked,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# TRAINING TIME BABY
trainer.train()
trainer.save_model("./final‑finetuned‑model")
tokenizer.save_pretrained("./final‑finetuned‑model")
# Evaluate lol i doubt well ever get to this part of the program
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)
