import os, json, torch
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

if __name__ == "__main__":
    root_dir = "/home/acsguser/Codes/SwinBERT/"
    df = pd.read_csv(os.path.join(root_dir, "datasets", "Crime", "caption_embedding.csv"), index_col=0)

    # split train and test set
    train_df = df[df.split == 'train']
    test_df = df[df.split == 'test']
    train_json, test_json, train_labels, test_labels = train_df.embedding.values.tolist(), test_df.embedding.values.tolist(), train_df.label.values.tolist(), test_df.label.values.tolist()
    print(len(train_json), len(test_json))

    train_inpputs, test_inputs = [], []
    for emb in train_json:
        train_inpputs.append(json.loads(emb))
    for emb in test_json:
        test_inputs.append(json.loads(emb))

    # clf = svm.SVC(probability=True)
    # clf = RandomForestClassifier()
    auc_res = []
    for i in [88]:
        clf = MLPClassifier(hidden_layer_sizes=[512,256,128], batch_size=16, learning_rate="invscaling",random_state=i, max_iter=2)
        clf.fit(train_inpputs, train_labels)

        preds = clf.predict(test_inputs)
        acc = accuracy_score(test_labels, preds)
        f1 = f1_score(test_labels, preds)

        probas = clf.predict_proba(test_inputs)
        auc = roc_auc_score(test_labels, probas[:, 1])
        auc_res.append(auc)
        print("seed={}, acc={}, test f1={}, test auc={}".format(i, f1, acc, auc))

    print("max={}, avg={}".format(max(auc_res), sum(auc_res) / len(auc_res)))
