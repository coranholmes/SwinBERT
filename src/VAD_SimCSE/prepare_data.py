import os, torch, json
import pandas as pd
from transformers import AutoModel, AutoTokenizer


if __name__ == "__main__":
    ds_root = "/home/acsguser/Codes/SwinBERT/datasets/Crime/"
    df = pd.DataFrame(columns=['split', 'path', 'caption', 'label'])

    train = []
    with open(os.path.join(ds_root, 'split', 'Anomaly_Detection_splits', 'Anomaly_Train.txt')) as f:
        for line in f:
            line = line.strip()
            train.append(line)
    print("Training data:", len(train))

    test = []
    with open(os.path.join(ds_root, 'split', 'Anomaly_Detection_splits', 'Anomaly_Test.txt')) as f:
        for line in f:
            line = line.strip()
            test.append(line)
    print("Test data:", len(test))

    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

    # generate record for dataframe
    with open(os.path.join(ds_root, 'captions.txt')) as f:
        for line in f:
            line = line.strip()
            path, caption = line.split("\t")
            label = 0 if "Normal" in path else 1
            split = 'train' if path in train else 'test'
            # record = pd.Series([split, path, caption, label], index=['split', 'path', 'caption', 'label'])
            record = {
                'split': split,
                'path': path,
                'caption': caption,
                'label': label,
            }
            df = df.append(record, ignore_index=True)

    texts = df['caption'].tolist()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    df['embedding'] = embeddings.tolist()
    df.to_csv(os.path.join(ds_root, 'caption_embedding.csv'))
