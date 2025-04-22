import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from safetensors.torch import load_file
import pandas as pd
import os
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

# Model and data loading
model_name = "yiyanghkust/finbert-tone"
model_path = "/content/final‑finetuned‑model/model.safetensors"
test_dataset_path = "/content/gdrive/MyDrive/finbert-nlp/processed_test_dataset"
test_csv_path = "/content/NLP-Project-Group-3/data/one_hot_targets/test_data.csv"

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4, ignore_mismatched_sizes=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = load_file(model_path)
model.classifier = torch.nn.Linear(model.classifier.in_features, 4)  # Update classifier
for key in state_dict:
    state_dict[key] = state_dict[key].to(device)
model.load_state_dict(state_dict, strict=False)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load test dataset
dataset_test_chunked = load_from_disk(test_dataset_path)
df_test = pd.read_csv(test_csv_path)

# Function to load the transcript text
def load_transcript(file_path):
    with open(f'/content/NLP-Project-Group-3/{file_path}', "r", encoding="utf-8") as f:
        return f.read()

# Create a dictionary mapping transcript_id to the full transcript text
transcript_texts = {os.path.basename(file_path): load_transcript(file_path) for file_path in df_test["file_path"]}

# Add the 'text' column back to the dataset
def add_text_column(example):
    example["text"] = transcript_texts[example["transcript_id"]]
    return example

dataset_test_chunked = dataset_test_chunked.map(add_text_column)

# Prediction function
def predict(text):
    encoding = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    return predictions[0]  # Return single prediction value

# Test functions
def test_first_512(example):
    return predict(example["text"][:512])

def test_last_512(example):
    return predict(example["text"][-512:])

def test_random_512(example):
    text = example["text"]
    start_idx = np.random.randint(0, len(text) - 512) if len(text) > 512 else 0
    return predict(text[start_idx:start_idx + 512])

def test_mean_pooling(example):
    chunks = [example["text"][i:i + 512] for i in range(0, len(example["text"]), 512)]
    chunk_predictions = [predict(chunk) for chunk in chunks]
    return int(round(np.mean(chunk_predictions))) # Calculate mean and round to nearest integer

def test_max_pooling(example):
    chunks = [example["text"][i:i + 512] for i in range(0, len(example["text"]), 512)]
    chunk_predictions = [predict(chunk) for chunk in chunks]
    return int(np.max(chunk_predictions)) # Calculate max

# Run tests and store results
results = defaultdict(list)
for example in dataset_test_chunked:
    transcript_id = example["transcript_id"]
    results[transcript_id].append({
        "first_512": test_first_512(example),
        "last_512": test_last_512(example),
        "random_512": test_random_512(example),
        "mean_pooling": test_mean_pooling(example),
        "max_pooling": test_max_pooling(example),
    })

# Run tests and store results
all_predictions = defaultdict(list)
ground_truth_labels = []
for example in dataset_test_chunked:
    all_predictions["first_512"].append(test_first_512(example))
    all_predictions["last_512"].append(test_last_512(example))
    all_predictions["random_512"].append(test_random_512(example))
    #all_predictions["mean_pooling"].append(test_mean_pooling(example))
    #all_predictions["max_pooling"].append(test_max_pooling(example))
    ground_truth_labels.append(example["labels"])

# Calculate and print overall metrics
print("Overall Metrics for Test Dataset:")
for method in ["first_512", "last_512", "random_512", "mean_pooling", "max_pooling"]:
    predictions = all_predictions[method]
    accuracy = np.mean(np.array(predictions) == np.array(ground_truth_labels))
    precision = precision_score(ground_truth_labels, predictions, average='macro')
    recall = recall_score(ground_truth_labels, predictions, average='macro')
    f1 = f1_score(ground_truth_labels, predictions, average='macro')

    print(f"  {method}:")
    print(f"    Accuracy: {accuracy}")
    print(f"    Precision (macro): {precision}")
    print(f"    Recall (macro): {recall}")
    print(f"    F1-score (macro): {f1}")
    print("-" * 20)