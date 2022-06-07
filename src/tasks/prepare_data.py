import os
import pandas as pd

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
                'label': label
            }

            df = df.append(record, ignore_index=True)
    df.to_csv(os.path.join(ds_root, 'data.csv'))
