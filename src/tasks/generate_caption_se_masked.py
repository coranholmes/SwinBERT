#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 7/11/22 16:04
# @Author  : CHEN Weiling
# @File    : generate_caption_se_masked.py
# @Software: PyCharm
# @Comments: given a video, generate its caption with different masks

import os, json, sys, torch, argparse
from transformers import AutoModel, AutoTokenizer
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate sentence embeddings of the captions")
    parser.add_argument("--dataset", type=str, choices=['ucf', 'shanghai', 'violence', 'ped2'] ,help="dataset to generate caption embeddings")
    parser.add_argument("--is_test", action='store_true', default=False, help="whether to generate test caption embeddings")
    parser.add_argument("--caption_path", type=str, default='', help="path to save the generated embeddings")
    parser.add_argument("--output_path", type=str, default='', help="path to save the generated embeddings")
    parser.add_argument("--model", type=str, default='sup-simcse-bert-base-uncased', help="name of the pretrained model")
    parser.add_argument("--vid_name", type=str, default='', help="name of the video")
    args = parser.parse_args()

    if args.dataset == "ucf":
        ds_name = "Crime"
        caption_path = ["/home/acsguser/Codes/SwinBERT/datasets/Crime/RTFM_train_caption/all_captions.txt"]
    elif args.dataset == "shanghai":
        ds_name = "Shanghai"
        caption_path = [
            "/home/acsguser/Codes/SwinBERT/datasets/Shanghai/RTFM_train_caption/train_captions.txt",
            "/home/acsguser/Codes/SwinBERT/datasets/Shanghai/RTFM_train_caption/train_captions.txt"
        ]
    elif args.dataset == "violence":
        ds_name = "Violence"
        caption_path = ["/home/acsguser/Codes/SwinBERT/datasets/Violence/RTFM_train_caption/all_captions.txt"]
    elif args.dataset == "ped2":
        ds_name = "UCSDped2"
        caption_path = [
            "/home/acsguser/Codes/SwinBERT/datasets/UCSDped2/RTFM_train_caption/train_captions.txt",
            "/home/acsguser/Codes/SwinBERT/datasets/UCSDped2/RTFM_train_caption/test_captions.txt"
        ]
    else:
        raise ValueError("dataset should be either ucf, shanghai, or violence")

    if args.caption_path != '':
        caption_path = args.caption_path

    print("Loading captions from ", caption_path)
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/" + args.model)
    model = AutoModel.from_pretrained("princeton-nlp/" + args.model)
    for cp in caption_path:
        with open(cp) as f:
            for line in f:
                captions = json.loads(line)
                key = []
                for k in captions:
                    key.append(k)
                assert len(key) == 1
                key = key[0]
                if args.vid_name in key:
                    if key.endswith("mp4") or key.endswith("avi"):
                        vid_name = os.path.split(key)[1][:-4]
                    else:
                        vid_name = os.path.split(key)[-1]
                    print(vid_name)
                    texts = captions[key]
                    break
    max_len = 20
    # for cc in texts:
    #     cur_len = len(cc.split())
    #     max_len = cur_len if cur_len > max_len else max_len
    list_file = open("/home/acsguser/Codes/RTFM/list/explainability/" + ds_name + "_" + args.vid_name + ".txt", "w")
    for i in range(max_len):
        print("Masking the {}-th word".format(i))
        masked_texts = []
        for cc in texts:
            words = cc.split()
            if i < len(words):
                words[i] = "[MASK]"
            masked_texts.append(" ".join(words))
        print(masked_texts)
        inputs = tokenizer(masked_texts, padding=True, truncation=True, return_tensors="pt")
        if args.output_path != "":
            emb_path = args.output_path + vid_name + "_emb.npy"
        else:
            os.makedirs("/home/acsguser/Codes/RTFM/save/" + ds_name + "/sent_emb_mask/" + vid_name, exist_ok=True)
            emb_path = "/home/acsguser/Codes/RTFM/save/" + ds_name + "/sent_emb_mask/" + vid_name + "/" + vid_name + "_" + str(i) +"_emb.npy"
        with torch.no_grad():
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        # write emb_path to list_file
        list_file.write(emb_path + "\n")
        np.save(emb_path, embeddings)
        print("Save to ", emb_path)
