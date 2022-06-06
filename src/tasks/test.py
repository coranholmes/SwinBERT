import os,pickle

if __name__ == "__main__":
    root_dir = "/home/acsguser/Codes/SwinBERT/"
    ds_dir = os.path.join(root_dir, "datasets", "Crime")
    g = os.walk(ds_dir)

    cap_pkl = open("/home/acsguser/Codes/SwinBERT/datasets/captions.pkl", "rb")
    cap_file = open("/home/acsguser/Codes/SwinBERT/datasets/captions.txt", "w")
    cap_dict = pickle.load(cap_pkl)

    for k, v in cap_dict.items():
        cap_file.writelines(k+"\t"+v+"\n")

    cap_file.close()
    cap_pkl.close()

