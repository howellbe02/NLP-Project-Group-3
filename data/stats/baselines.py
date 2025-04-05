# imports
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.multioutput import MultiOutputClassifier
from tabulate import tabulate  

# Loads all CSV files from directory and makes one massive thing
def load_data(data_dir):   
    dfs = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)
            dfs.append(df)
    if not dfs:
        print("Bruh nothing here")
        return None
    return pd.concat(dfs, ignore_index=True)


# Computes the counts and proportions for labels
# Earnings surprise data dist is a little concerning
def compute_label_distribution(data, target):
   
    counts = data[target].value_counts().to_dict()
    counts.setdefault(0, 0)
    counts.setdefault(1, 0)
    total = counts[0] + counts[1]
    distribution = {
        'Count 0': counts[0],
        'Count 1': counts[1],
        'Total Samples': total,
        'Proportion 0': counts[0] / total,
        'Proportion 1': counts[1] / total,
    }
    return distribution

# gets DummyClassifier metrics, easy baselines 
# Note: there is no real input feature used bc im not loading all those .txts so i just used row idx it dont really matter
def evaluate_dummy_classifier(data, target, strategy):
    dummy = DummyClassifier(strategy=strategy, random_state=0)

    X = data.index.values.reshape(-1, 1)  # <-----------------  dummy feature 
    y = data[target].values
    
    dummy.fit(X, y)
    predictions = dummy.predict(X)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)

    return accuracy, precision, recall, f1

def print_baselines_and_stats():
    # Path to the folder, dont forget to change unless you me
    data_dir = '/Users/robbie/Desktop/NLP/NLP-Project-Group-3/data/one_hot_targets'
    data = load_data(data_dir)
    if data is None:
        return

    # calc dummy metrics for separate task
    tasks = ['surprise_pct', 'volatility_change']
    strategies = ['most_frequent', 'stratified', 'uniform']

    for target in tasks:
        print(f"\n===== Statistics for '{target}' =====")
        
        # label dist
        distribution = compute_label_distribution(data, target)
        dist_table = [[k, f"{v:.4f}" if isinstance(v, float) else v] for k, v in distribution.items()]
        print("Label Distribution:")
        print(tabulate(dist_table, headers=["Metric", "Value"], tablefmt="grid"))
        
        # dummy metrics
        results = []
        for strategy in strategies:
            accuracy, precision, recall, f1 = evaluate_dummy_classifier(data, target, strategy)
            results.append([strategy, f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])
        
        print("\nDummy Classifier Baselines:")
        headers = ["Strategy", "Accuracy", "Precision", "Recall", "F1 Score"]
        print(tabulate(results, headers=headers, tablefmt="grid"))
    
    # correlate 
    if all(task in data.columns for task in tasks):
        corr = data[tasks].corr().iloc[0, 1]
        print(f"\nCorrelation between '{tasks[0]}' and '{tasks[1]}': {corr:.4f}")

    # dummy metrics for both task same time
    print("\n===== Combined Baselines for Both Tasks =====")
    combined_results = []
    X = data.index.values.reshape(-1, 1)  # dummy again bc i didnt want to change the other func, someone else can if they want
    y_combined = data[tasks].values         

    for strategy in strategies:
        multi_dummy = MultiOutputClassifier(DummyClassifier(strategy=strategy, random_state=0))

        multi_dummy.fit(X, y_combined)
        y_pred = multi_dummy.predict(X)

        joint_accuracy = accuracy_score(y_combined, y_pred)
        joint_precision = precision_score(y_combined, y_pred, average='macro', zero_division=0)
        joint_recall = recall_score(y_combined, y_pred, average='macro', zero_division=0)
        joint_f1 = f1_score(y_combined, y_pred, average='macro', zero_division=0)

        combined_results.append([strategy, f"{joint_accuracy:.4f}", f"{joint_precision:.4f}", 
                                 f"{joint_recall:.4f}", f"{joint_f1:.4f}"])
    
    print(tabulate(combined_results, headers=["Strategy", "Accuracy", "Precision", "Recall", "F1 Score"], tablefmt="grid"))

if __name__ == '__main__':
    print_baselines_and_stats()
