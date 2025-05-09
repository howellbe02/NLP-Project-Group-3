import os
import pandas as pd
from datasets import Dataset, load_from_disk
import torch
from transformers import BigBirdTokenizer, BigBirdConfig, BigBirdForSequenceClassification, TrainingArguments, Trainer
import evaluate

# DEVICE SETUP
################ CHANGE TO COMMENTED LINE
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess the fat amount of data
# CSV FILE PATHS
## CHANGE ########|------------------------| <----- That part
train_csv_path = "/Users/robbie/Desktop/NLP/NLP-Project-Group-3/data/one_hot_targets/train_data.csv"  # Update if needed
test_csv_path  = "/Users/robbie/Desktop/NLP/NLP-Project-Group-3/data/one_hot_targets/test_data.csv"   # Update if needed


# marco del datos
df_train = pd.read_csv(train_csv_path)
df_test = pd.read_csv(test_csv_path)

# transcript IDs (I'm unsure if we still need but put in just in case)
df_train["transcript_id"] = df_train["file_path"].apply(lambda x: os.path.basename(x))
df_test["transcript_id"] = df_test["file_path"].apply(lambda x: os.path.basename(x))

# Function to load transcript text
def load_transcript(file_path):
    ####################
    #CHANGE THE F STRING TO HAVE WHEREVER UR THING IS
    #KEEP EVERYTHING AFTER NLP BTU NOT NLP
    ############|-------------------------| <-- this part
    with open(f'/Users/robbie/Desktop/NLP/NLP-Project-Group-3/{file_path}', "r", encoding="utf-8") as f:
        return f.read()

# preload the text bc basically it stops it from constantly opening files and wasting everyones time
df_train["text"] = df_train["file_path"].apply(load_transcript)
df_test["text"]  = df_test["file_path"].apply(load_transcript)

# rename for label bc we train on label
df_train = df_train.rename(columns={"combined_label": "labels"})
df_test  = df_test.rename(columns={"combined_label": "labels"})

# hugging face bull
dataset_train = Dataset.from_pandas(df_train)
dataset_test  = Dataset.from_pandas(df_test)

# TOKENIZATION SECTION (so much easier this time)
# there are a few bigbirds, ill see if they all work and whoever trains can pick
model_name = "google/bigbird-roberta-base" #"google/bigbird-roberta-base" #"l-yohai/bigbird-roberta-base-mnli"
tokenizer = BigBirdTokenizer.from_pretrained(model_name)

# dont care after 4096 tokens, so dont have to worry about chunking stuff
# paper did this so i am too bc that chunking stuff was a headache
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=4096,
        return_attention_mask=True,
    )

# Columns to remove after tokenization bc trainer is picky
columns_to_remove = ["file_path", "text", "surprise_pct", "volatility_change", "report_date", "ticker"]

######################
# CHANGE              |-------------------------| <------ this part (or whole thing i dont think it matters)
train_dataset_path = "/Users/robbie/Desktop/NLP/processed_bigbird_train_dataset"
test_dataset_path  = "/Users/robbie/Desktop/NLP/processed_bigbird_test_dataset"

# processs + save
if not os.path.exists(train_dataset_path):
    dataset_train = dataset_train.map(
        tokenize_fn,
        batched=True,
        batch_size=4,      
        remove_columns=columns_to_remove,
        num_proc=8,        
        load_from_cache_file=True
    )
    # this part actually saves the stuff
    dataset_train.save_to_disk(train_dataset_path)
else:
    # this loads the stuff if it exists hopefully it does
    dataset_train = load_from_disk(train_dataset_path)

# same shit different story
if not os.path.exists(test_dataset_path):
    dataset_test = dataset_test.map(
        tokenize_fn,
        batched=True,
        batch_size=4,       
        remove_columns=columns_to_remove,
        num_proc=8,
        load_from_cache_file=True
    )
    #saving stuff to file
    dataset_test.save_to_disk(test_dataset_path)
else:
    #loading stuff if it there
    dataset_test = load_from_disk(test_dataset_path)

# stupid hugging face gotta make everything perfect for it 
dataset_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "transcript_id"])
dataset_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "transcript_id"])

# config stuff, I made block size= 64 and random blocks=4 bc the paper tuned for block size = {32,64,128}
# and random blocks={3,4,5} so i just chose the middle one unsure how much it matters but i did it 
config = BigBirdConfig.from_pretrained(
    model_name,
    num_labels=4,
    block_size=64,
    num_random_blocks=4
)

# el model
model = BigBirdForSequenceClassification.from_pretrained(
    model_name,
    config=config,
    ignore_mismatched_sizes=True
)
model.to(device)

for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

print("Trainable parameters:")
for n, p in model.named_parameters():
    if p.requires_grad:
        print("  ", n)
    else:
        print('not ', n)

# trainer args
training_args = TrainingArguments(
    output_dir="./bigbird_output",
    num_train_epochs=10,
    per_device_train_batch_size=4,     
    per_device_eval_batch_size=4,      
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# define metrics (left the other two out bc i dont care)
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
# just a function for the trainer to compute metrics bc you gotta do everything for it
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy.compute(predictions=preds, references=labels)
    f1_score = f1.compute(predictions=preds, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}

# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# TRAIN 
trainer.train()
# Evaluate lol i doubt well ever get to this part of the program
eval_results = trainer.evaluate()
print("Evaluation results:\n", eval_results)