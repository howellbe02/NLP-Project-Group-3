import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time
# CSV
csv_path = "/Users/robbie/Desktop/NLP/NLP-Project-Group-3/data/one_hot_targets/test_data.csv"
df = pd.read_csv(csv_path)

# load full transcript 
def load_transcript(fp):
    with open(f'/Users/robbie/Desktop/NLP/NLP-Project-Group-3/{fp}', "r", encoding="utf-8") as f:
        return f.read()
df["text"] = df["file_path"].apply(load_transcript)

# rename for HF
df = df.rename(columns={"combined_label": "labels"})

# file path is trans id
ds = Dataset.from_pandas(
    df[["text","labels","file_path"]],
    preserve_index=False
).rename_column("file_path","transcript_id")

# model 
model_dir = '/Users/robbie/Desktop/NLP/final‑finbert‑model'
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# no overlap
def tokenize_no_overlap(example):
    enc = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",   
        max_length=512,
        stride=0,               
        return_overflowing_tokens=True,
        return_attention_mask=True,
    )
    n = len(enc["input_ids"])
    # each segment has label bc stupid HF, i fix later
    return {
        "input_ids":      enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels":         [example["labels"]] * n,
        "transcript_id":  [example["transcript_id"]] * n,
    }

# map 
ds_chunked = ds.map(
    tokenize_no_overlap,
    batched=False,
    remove_columns=["text"],
)

# format
ds_chunked.set_format(
    type="torch",
    columns=["input_ids","attention_mask","labels","transcript_id"]
)

print("Produced", len(ds_chunked), "chunks across", len(df), "documents.")


# does both 
def evaluate_pooling(ds_docs):
    docs_cls = {}   # trans id: list of chunk CLS embeddings
    docs_lbl = {}   
    start = time.perf_counter()
    k = 1
    with torch.no_grad():
        for ex in ds_docs:
            
            chunks_ids = ex["input_ids"]      
            chunks_mask = ex["attention_mask"] 
            tids = ex["transcript_id"]        
            # each entry in labels is same 
            lbl = ex["labels"][0].item() if isinstance(ex["labels"], torch.Tensor) else ex["labels"][0]

            # iterate each chunk
            for i in range(chunks_ids.size(0)):
                input_ids = chunks_ids[i].unsqueeze(0).to(device)
                attention_mask = chunks_mask[i].unsqueeze(0).to(device)
                bert_out = model.bert(input_ids=input_ids, attention_mask=attention_mask)
                cls_emb = bert_out.pooler_output.squeeze(0).cpu()
                tid = tids[i]
                docs_cls.setdefault(tid, []).append(cls_emb)
                docs_lbl.setdefault(tid, lbl)
            print(f'{k}: {time.perf_counter()-start}')
            k+=1

    # pool per document and classify
    y_true, y_pred_mean, y_pred_max = [], [], []
    for tid, embs in docs_cls.items():
        stacked = torch.stack(embs, dim=0)  
        if k == 101:
            print(stacked.shape)
            k+= 1
        # mean pool
        doc_emb_mean = stacked.mean(dim=0)
        doc_emb_mean = doc_emb_mean.unsqueeze(0).to(device)
        logits_mean = model.classifier(doc_emb_mean)
        pred_mean = logits_mean.argmax(dim=-1).item()
        y_pred_mean.append(pred_mean)
         # max pool
        doc_emb_max, _ = stacked.max(dim=0)
        doc_emb_max = doc_emb_max.unsqueeze(0).to(device)
        logits_max = model.classifier(doc_emb_max)
        pred_max = logits_max.argmax(dim=-1).item()
        y_pred_max.append(pred_max)

        y_true.append(docs_lbl[tid])
    # Mean pool
    acc = accuracy_score(y_true, y_pred_mean)
    f1 = f1_score(y_true, y_pred_mean, average="macro")
    print(f"[MEAN POOL] Acc: {acc:.4f}, Macro-F1: {f1:.4f}")
    print(classification_report(y_true, y_pred_mean, zero_division=0))
    # Max pool
    acc = accuracy_score(y_true, y_pred_max)
    f1 = f1_score(y_true, y_pred_max, average="macro")
    print(f"[MAX POOL] Acc: {acc:.4f}, Macro-F1: {f1:.4f}")
    print(classification_report(y_true, y_pred_max, zero_division=0))

    # Per-dimension (ES / VC) metrics
    # mapping: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    es_true = [lbl//2 for lbl in y_true]
    es_pred_mean = [p//2 for p in y_pred_mean]
    es_pred_max = [p//2 for p in y_pred_max]

    vc_true = [lbl%2 for lbl in y_true]
    vc_pred_mean = [p%2 for p in y_pred_mean]
    vc_pred_max = [p%2 for p in y_pred_max]

    print(f"\n MEAN POOL")
    print(f"\n ES performance:")
    print(classification_report(es_true, es_pred_mean, labels=[0,1],
                                target_names=["ES=0","ES=1"], zero_division=0))
    print(f"\n VC performance:")
    print(classification_report(vc_true, vc_pred_mean, labels=[0,1],
                                target_names=["VC=0","VC=1"], zero_division=0))
    print(f"\n MAX POOL")
    print(f"\n ES performance:")
    print(classification_report(es_true, es_pred_max, labels=[0,1],
                                target_names=["ES=0","ES=1"], zero_division=0))
    print(f"\n VC performance:")
    print(classification_report(vc_true, vc_pred_max, labels=[0,1],
                                target_names=["VC=0","VC=1"], zero_division=0))

evaluate_pooling(ds_chunked)

