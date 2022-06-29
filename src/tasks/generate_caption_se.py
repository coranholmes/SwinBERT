# generate the sentence embeddings of the captions
import os, json, sys, torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

if __name__ == "__main__":
    caption_path = "/home/acsguser/Codes/SwinBERT/datasets/Crime/RTFM_train_caption/dense_captions_16.txt"
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
            vid_name = os.path.split(key)[1][:-4]
            print(vid_name)
            emb_path = "/home/acsguser/Codes/SwinBERT/datasets/Crime/RTFM_train_caption/sent_emb_16/" + vid_name + "_emb.npy"
            texts = captions[key]
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
            np.save(emb_path, embeddings)
