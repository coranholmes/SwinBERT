import os

if __name__ == "__main__":
    ds_root = "/home/acsguser/Codes/SwinBERT/datasets/Crime/"
    with open(os.path.join(ds_root, "captions.txt")) as f:
        max_len = 0
        sent = ''
        for line in f:
            line = line.strip()
            arr = line.split("\t")
            cur_len = len(arr[1])
            if cur_len > max_len:
                max_len = cur_len
                sent = arr[1]
    print(max_len, sent)