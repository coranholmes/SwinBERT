import pickle, os

# 读取保存的captions，生成pickle文件
if __name__ == "__main__":
    # read from pickle
    # with open("/home/acsguser/Codes/SwinBERT/datasets/captions.pkl", "rb") as f:
    #     caps = pickle.load(f)
    # print(len(caps))
    caps = dict()
    # read captions to be added from txt file
    with open("/home/acsguser/Codes/SwinBERT/datasets/Crime/captions.txt") as f:
        for line in f:
            line = line.strip()
            arr = line.split("\t")
            print(arr)
            caps[arr[0]] = arr[1]

    # save to captions2.pkl
    print(len(caps))
    with open("/home/acsguser/Codes/SwinBERT/datasets/Crime/captions.pkl", "wb") as f:
        pickle.dump(caps, f)
