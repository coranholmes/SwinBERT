import os,pickle

# 根据w
if __name__ == "__main__":
    ds_root = "/home/acsguser/Codes/SwinBERT/datasets/Crime/"
    captions_pkl = open(os.path.join(ds_root, "captions.pkl"), "rb")
    cap_dict = pickle.load(captions_pkl)
    # train_file = os.path.join(ds_root, "split", "Anomaly_Detection_splits", "Anomaly_Train.txt")
    label_pkl = open(os.path.join(ds_root, "labels.pkl"), "wb")
    label_dict = dict()
    print("No. of captions: ", len(cap_dict))

    for k,v in cap_dict.items():
            label_dict[k] = 0 if "Normal" in k else 1

    print("No. of labels: ", len(label_dict))

    pickle.dump(label_dict, label_pkl)





