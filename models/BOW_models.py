import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# BASE DIR, change to yours
BASE_DIR = '/Users/robbie/Desktop/NLP/NLP-Project-Group-3/'
# function to load up all the data since I stopped formatting it nicely at some point
def load_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    texts = []
    labels_comb = []
    labels_comb_sep = [] # keeps it as two separate binary classification tasks
    labels_ES = []
    labels_VC = []
    for idx, row in df.iterrows():
        file_path = BASE_DIR + row['file_path']
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        texts.append(text)
        labels_comb.append(row['combined_label'])
        labels_comb_sep.append([row['surprise_pct'], row['volatility_change']])
        labels_ES.append(row['surprise_pct'])
        labels_VC.append(row['volatility_change'])
    return np.array(texts), np.array(labels_comb), np.array(labels_ES), np.array(labels_VC), np.array(labels_comb_sep)

# this is just where I had it before i pushed it to github, def change if trying to run
train_csv = "/Users/robbie/Desktop/NLP/train_data.csv"
test_csv  = "/Users/robbie/Desktop/NLP/test_data.csv"
# load up them train and tests
train_texts, train_combined, train_surprise, train_volatility, train_comb_sep = load_data(train_csv)
test_texts, test_combined, test_surprise, test_volatility, test_comb_sep = load_data(test_csv)
labels = ['ES', 'VC']
# TF-IDF VOCAB SET UP
vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=50000, max_df=0.8)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# 1) Logistic Regression for separate binary classification tasks bc i wanted to
# a) Earnings Surprise
log_reg_ES = LogisticRegression(max_iter=10000)
log_reg_ES.fit(X_train, train_surprise)
pred_ES = log_reg_ES.predict(X_test)
print(f'Logistic Regression on ES:')
print(classification_report(test_surprise, pred_ES, zero_division=0))
print(f'--------------------------------------')

# b) Volaltility Change
log_reg_VC = LogisticRegression(max_iter=10000)
log_reg_VC.fit(X_train, train_volatility)
pred_VC = log_reg_VC.predict(X_test)
print(f'Logistic Regression on VC:')
print(classification_report(test_volatility, pred_VC, zero_division=0))
print(f'--------------------------------------')

# I later found out you can use Logistic Regression on multiclass classification 
# so I also did it for the combined labels, but now we got more stuff i guess
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train, train_combined)
pred_comb = log_reg.predict(X_test)
print(f'Logistic Regression on both classes:')
print(classification_report(test_combined, pred_comb, zero_division=0))
print(f'--------------------------------------')

# Guess what, i read more and turns out we can predict the classes separately,
# this is a lot fr but idk whats best so i just did everything
log_reg_sep = MultiOutputClassifier(LogisticRegression(max_iter=10000))
log_reg_sep.fit(X_train, train_comb_sep)
log_reg_sep_pred = log_reg_sep.predict(X_test)
print(f'Logistic Regression on both classes separate:')
for i in range(2):
    print(f'Classification report for {labels[i]}')
    print(classification_report(test_comb_sep[:, i], log_reg_sep_pred[:, i], zero_division=0))
print(f'--------------------------------------')

# 2) Simple NN model and GBDT on both tasks combined
# NOTE: didnt realize logistic regression can do multiclass so i included a simple NN
#       later realized it does so I added it above but still kept this

# a) simple NN
# didn't know what i wanted for hidden layers so i randomly decided, discuss if change is needed
mlp = MLPClassifier(hidden_layer_sizes=(100,75,50), max_iter=10000)
mlp.fit(X_train, train_combined)
mlp_pred = mlp.predict(X_test)
print(f'Simple NN on both classes:')
print(classification_report(test_combined, mlp_pred, zero_division=0))
print(f'--------------------------------------')

# b) Gradient Boosted Decision Tree (epic name)
xgb = XGBClassifier(tree_method='hist', eval_metric='logloss')
xgb.fit(X_train, train_combined)
xgb_pred = xgb.predict(X_test)
print(f'GBDT on both classes combined:')
print(classification_report(test_combined, xgb_pred, zero_division=0))
print(f'--------------------------------------')

# Okay so (b) got a whole lot bigger bc i read into the documentation
# So I'm including a GBDT for multiple outputs and a random forest of GBDTs
xgb_sep = XGBClassifier(tree_method='hist', eval_metric='logloss', multi_strategy="multi_output_tree")
xgb_sep.fit(X_train, train_comb_sep)
xgb_sep_pred = xgb_sep.predict(X_test)
print(f'GBDT on both classes separated:')
for i in range(2):
    print(f'Classification report for {labels[i]}')
    print(classification_report(test_comb_sep[:, i], xgb_sep_pred[:, i], zero_division=0))
print(f'--------------------------------------')

# These are just random forest versions bc i was interested
xgb_rf = XGBRFClassifier(n_estimators=100, max_depth=5)
xgb_rf.fit(X_train, train_combined)
xgb_rf_pred = xgb_rf.predict(X_test)
print(f'GBDT Random Forest on both classes combined:')
print(classification_report(test_combined, xgb_rf_pred, zero_division=0))
print(f'--------------------------------------')

xgb_rf_sep = XGBRFClassifier(n_estimators=100, max_depth=5, multi_strategy="multi_output_tree")
xgb_rf_sep.fit(X_train, train_comb_sep)
xgb_rf_sep_pred = xgb_rf_sep.predict(X_test)
print(f'GBDT Random Forest on both classes separated:')
for i in range(2):
    print(f'Classification report for {labels[i]}')
    print(classification_report(test_comb_sep[:, i], xgb_rf_sep_pred[:, i], zero_division=0))
print(f'--------------------------------------')