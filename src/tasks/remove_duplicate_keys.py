# 生成dense caption的时候不知何故有些文件生成了两次，以文件名为key进行去重，重新整理成最终的all_captions.txt
import json

if __name__=="__main__":
    video_set = set()
    cap_file = open("/home/acsguser/Codes/SwinBERT/datasets/Crime/RTFM_train_caption/all_captions_wo_dup.txt", "w")
    i = 0
    with open ("/home/acsguser/Codes/SwinBERT/datasets/Crime/RTFM_train_caption/all_captions.txt") as f:
        for line in f:
            line_dict = json.loads(line.strip())
            for k in line_dict.keys():
                k = k.strip()
                if k not in video_set:
                    video_set.add(k.strip())
                    cap_file.writelines(line)
                    i += 1
    print(i)