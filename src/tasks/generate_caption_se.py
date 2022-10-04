# generate the sentence embeddings of the captions
import os, json, sys, torch, argparse
from transformers import AutoModel, AutoTokenizer
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sentence embeddings of the captions")
    parser.add_argument("--dataset", type=str, choices=['ucf', 'shanghai', 'violence'] ,help="dataset to generate caption embeddings")
    parser.add_argument("--is_test", action='store_true', default=False, help="whether to generate test caption embeddings")
    args = parser.parse_args()
    is_test = "test" if args.is_test else "train"
    if args.dataset == "ucf":
        ds_name = "Crime"
        caption_path = "/home/acsguser/Codes/SwinBERT/datasets/Crime/RTFM_train_caption/all_captions.txt"
    elif args.dataset == "shanghai":
        ds_name = "Shanghai"
        caption_path = "/home/acsguser/Codes/SwinBERT/datasets/" + ds_name + "/RTFM_train_caption/" + is_test + "_captions.txt"
    elif args.dataset == "violence":
        ds_name = "Violence"
        caption_path = "/home/acsguser/Codes/SwinBERT/datasets/Violence/RTFM_train_caption/all_captions.txt"
    else:
        raise ValueError("dataset should be either ucf, shanghai, or violence")

    print("Loading captions from ", caption_path)
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    with open(caption_path) as f:
        for line in f:
            captions = json.loads(line)
            key = []
            for k in captions:
                key.append(k)
            assert len(key) == 1
            key = key[0]
            if key.endswith("mp4") or key.endswith("avi"):
                vid_name = os.path.split(key)[1][:-4]
            else:
                vid_name = os.path.split(key)[-1]
            print(vid_name)
            emb_path = "/home/acsguser/Codes/SwinBERT/datasets/" + ds_name + "/RTFM_train_caption/sent_emb_n/" + vid_name + "_emb.npy"
            texts = captions[key]
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
            np.save(emb_path, embeddings)
