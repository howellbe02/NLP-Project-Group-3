import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, TrainingArguments, Trainer
import evaluate
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from functools import partial
from nltk.tokenize import sent_tokenize

##### THIS FILE NEEDS TO BE IN THE SAME DIRECTORY AS THE DATA DIRECTORY OR UPDATE THE 'BASE_DIR'

class HierarchicalTransformer(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(HierarchicalTransformer, self).__init__()
        self.segment_encoder = AutoModel.from_pretrained(pretrained_model_name) # Segment encoder (pretrained model)
        hidden_size = self.segment_encoder.config.hidden_size
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=6,  # As mentioned in the paper
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.document_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3) # Document encoder (transformer layers)

        self.num_classes = num_classes # For our purposes this = 4
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        ) # Classification layers
    
    def forward(self, input_ids, attention_mask, labels=None, transcript_id=None, **kwargs):
        # Process all segments
        outputs = self.segment_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token embeddings for each segment
        unique_transcripts = list(set(transcript_id))
        all_document_embeddings = []
        all_labels = []
        
        for trans_id in unique_transcripts:
            indices = [i for i, t_id in enumerate(transcript_id) if t_id == trans_id]  # Get indices of segments belonging to this transcript

            if not indices:
                continue
                
            transcript_embeddings = cls_embeddings[indices]  # Get embeddings, Shape: [num_segments, hidden_size]

            document_output = self.document_encoder(transcript_embeddings.unsqueeze(0))  # TransformerEncoder expects [batch_size=1, seq_len=num_segments, hidden_size]
            
            first_token = document_output[0, 0, :]
            max_pooled = torch.max(document_output[0], dim=0)[0]  # Max over all segments
            doc_representation = torch.cat([first_token, max_pooled], dim=0)
            all_document_embeddings.append(doc_representation)
            
            if labels is not None:
                all_labels.append(labels[indices[0]])
        
            
        document_representations = torch.stack(all_document_embeddings)  # Convert lists to tensors
        logits = self.classifier(document_representations)  # Classify
        
        # Handle loss calculation (when in training mode)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            batch_labels = torch.tensor(all_labels, device=logits.device)
            loss = loss_fct(logits, batch_labels)
            return {"loss": loss, "logits": logits}
        
        return logits


def load_transcript(file_path, base_path):
    with open(f'{base_path}/{file_path}', "r", encoding="utf-8") as f:
        return f.read()



def tokenize_and_chunk(batch, tokenizer, max_length=512): # Greedy Sentence Chunking like in the paper
    output = {
        "input_ids": [], 
        "attention_mask": [], 
        "labels": [], 
        "transcript_id": []
    }
    
    for idx in range(len(batch["text"])):
        text = batch["text"][idx]
        label = batch["labels"][idx]
        transcript_id = batch["transcript_id"][idx]

        sentences = sent_tokenize(text)
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            tokens = tokenizer(sentence, add_special_tokens=False)["input_ids"] # Get token count
            sentence_length = len(tokens)
            
            # Check if adding this sentence would exceed the limit
            if current_length + sentence_length > max_length - 2:
                if current_chunk:  # Process current chunk if not empty
                    encoded = tokenizer(
                        " ".join(current_chunk),
                        truncation=True,
                        max_length=max_length,
                        padding="max_length",
                        return_tensors="pt"
                    ) # Tokenize the whole chunk at once
                    
                    output["input_ids"].append(encoded["input_ids"][0])
                    output["attention_mask"].append(encoded["attention_mask"][0])
                    output["labels"].append(label)
                    output["transcript_id"].append(transcript_id)
                    
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Handle remaining text
        if current_chunk:
            encoded = tokenizer(
                " ".join(current_chunk),
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            output["input_ids"].append(encoded["input_ids"][0])
            output["attention_mask"].append(encoded["attention_mask"][0])
            output["labels"].append(label)
            output["transcript_id"].append(transcript_id)
    
    return output

def compute_metrics(eval_pred, eval_dataset):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    logits, labels = eval_pred
    transcript_ids = eval_dataset["transcript_id"]
    
    # Get unique transcript IDs that exist in the current evaluation batch
    unique_transcripts = {}
    for i, t_id in enumerate(transcript_ids):
        if t_id not in unique_transcripts and i < len(logits):
            unique_transcripts[t_id] = i
    
    # For each unique transcript, take one prediction and one label
    doc_predictions = []
    doc_labels = []
    
    for t_id, idx in unique_transcripts.items():
        # Make sure the index is within bounds
        if idx < len(logits):
            # Take prediction and label from this segment
            pred = np.argmax(logits[idx])
            label = labels[idx]
            
            doc_predictions.append(pred)
            doc_labels.append(label)
    
    # Calculate metrics on document-level predictions
    accuracy = accuracy_metric.compute(predictions=doc_predictions, references=doc_labels)
    f1 = f1_metric.compute(predictions=doc_predictions, references=doc_labels, average="macro")
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
    }

# Custom data collator with padding and handles missing keys
class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, features):
        batch = {}
        for key in ["input_ids", "attention_mask"]:
            batch[key] = torch.stack([feature[key] for feature in features]) # Stack tensors for input_ids and attention_mask
        
        batch["labels"] = torch.tensor([feature["labels"] for feature in features]) # Convert labels to tensor
        batch["transcript_id"] = [feature["transcript_id"] for feature in features] # Pass through transcript_id as list

        return batch

def load_data(base_dir):
    # CSV FILE PATHS
    train_csv_path = os.path.join(base_dir, "data", "one_hot_targets", "train_data.csv")
    test_csv_path = os.path.join(base_dir, "data", "one_hot_targets", "test_data.csv")

    # Load data
    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # Add transcript ID for identification
    df_train["transcript_id"] = df_train["file_path"].apply(lambda x: os.path.basename(x))
    df_test["transcript_id"] = df_test["file_path"].apply(lambda x: os.path.basename(x))
    
    # Load transcript content
    df_train["text"] = df_train["file_path"].apply(lambda x: load_transcript(x, base_dir))
    df_test["text"] = df_test["file_path"].apply(lambda x: load_transcript(x, base_dir))

    # Rename label column if it's not already called "labels"
    df_train = df_train.rename(columns={"combined_label": "labels"})
    df_test = df_test.rename(columns={"combined_label": "labels"})

    # Convert to Hugging Face datasets
    dataset_train = Dataset.from_pandas(df_train)
    dataset_test = Dataset.from_pandas(df_test)
    
    return dataset_train, dataset_test

def process_datasets(dataset_train, dataset_test, tokenizer, base_dir):
    columns_to_remove = ["file_path", "text", "surprise_pct", "volatility_change", "report_date", "ticker"]
    columns_to_remove = [col for col in columns_to_remove if col in dataset_train.features]

    # Processed dataset paths
    train_dataset_path = os.path.join(base_dir, "processed_train_dataset")
    test_dataset_path = os.path.join(base_dir, "processed_test_dataset")
    
    # Define tokenizer function
    max_seq_length = 512
    tokenize_func = partial(tokenize_and_chunk, tokenizer=tokenizer, max_length=max_seq_length)

    # Process and save training dataset
    if not os.path.exists(train_dataset_path):
        print("Processing training dataset...")
        dataset_train_chunked = dataset_train.map(
            tokenize_func,
            batched=True,
            batch_size=8,
            remove_columns=columns_to_remove,
            num_proc=8,
            load_from_cache_file=True
        )
        dataset_train_chunked.save_to_disk(train_dataset_path)
    else:
        print("Loading processed training dataset from disk...")
        dataset_train_chunked = load_from_disk(train_dataset_path)

    # Process and save test dataset
    if not os.path.exists(test_dataset_path):
        print("Processing test dataset...")
        dataset_test_chunked = dataset_test.map(
            tokenize_func,
            batched=True,
            batch_size=8,
            remove_columns=columns_to_remove,
            num_proc=8,
            load_from_cache_file=True
        )
        dataset_test_chunked.save_to_disk(test_dataset_path)
    else:
        print("Loading processed test dataset from disk...")
        dataset_test_chunked = load_from_disk(test_dataset_path)
    
    # Set dataset format to PyTorch tensors
    dataset_train_chunked.set_format(
        type="torch", 
        columns=["input_ids", "attention_mask", "labels", "transcript_id"]
    )
    dataset_test_chunked.set_format(
        type="torch", 
        columns=["input_ids", "attention_mask", "labels", "transcript_id"]
    )
    
    return dataset_train_chunked, dataset_test_chunked

def setup_model_and_trainer(model_name, dataset_train_chunked, dataset_test_chunked, tokenizer, device):
    # Create HierarchicalTransformer model
    model = HierarchicalTransformer(
        pretrained_model_name=model_name,
        num_classes=4  # Make sure this matches your actual number of classes
    )
    
    # Add config attribute needed by Trainer
    model.config = type('obj', (object,), {
        'num_labels': model.num_classes
    })
    
    model.to(device)

    # Training arguments, these are for training
    training_args = TrainingArguments(
        output_dir="./hierarchical_transformer_output",
        num_train_epochs=2,                     
        per_device_train_batch_size=4,         
        per_device_eval_batch_size=4,         
        learning_rate=2e-5,                    
        weight_decay=0.01,                      
        evaluation_strategy="steps",          
        eval_steps=0.2,                        
        save_strategy="steps",                  
        save_steps=0.2,                         
        load_best_model_at_end=True,            
        metric_for_best_model="f1"              
    )


    data_collator = CustomDataCollator(tokenizer)

    def compute_metrics_wrapper(eval_pred): # Some BS to avoid nonsense with how this trainer works, there's always something 
        return compute_metrics(eval_pred, dataset_test_chunked)
    
    # Initialize standard Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train_chunked,
        eval_dataset=dataset_test_chunked,
        compute_metrics=compute_metrics_wrapper,
        data_collator=data_collator
    )
    
    return trainer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and process data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_train, dataset_test = load_data(BASE_DIR)
    dataset_train_chunked, dataset_test_chunked = process_datasets(
        dataset_train, dataset_test, tokenizer, BASE_DIR
    )
    
    trainer = setup_model_and_trainer(
        model_name, dataset_train_chunked, dataset_test_chunked, tokenizer, device
    )
    trainer.train() 
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    trainer.save_model("./best_hierarchical_transformer_model")
    print("Training and evaluation completed. Model saved to './best_hierarchical_transformer_model'")


if __name__ == "__main__":
    main()
