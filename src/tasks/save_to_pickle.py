import pickle, os

if __name__ == "__main__":
    # read from pickle
    with open("/home/acsguser/Codes/SwinBERT/datasets/captions.pkl", "rb") as f:
        caps = pickle.load(f)
    print(len(caps))
    # read captions to be added from txt file
    with open("/home/acsguser/Codes/SwinBERT/datasets/captions.txt") as f:
        i = 0
        for line in f:
            line = line.strip()
            arr = line.split("\t")
            caps[arr[0]] = arr[1]
            i += 1
            # print(i)
    # save to captions2.pkl
    print(len(caps))
    with open("/home/acsguser/Codes/SwinBERT/datasets/captions.pkl", "wb") as f:
        pickle.dump(caps, f)
