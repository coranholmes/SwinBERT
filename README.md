# Dense generation using SwinBERT
Refer to `README.md` for more information about setting up the environment.

## Requirements 
pre-requisites:
```bash
sudo apt-get install libopenmpi-dev
pip install -r requirements.txt
```
To install apex:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
```

## Download
Refer to the `Download` section in the original `README_orig.md`. For pretrained model, I am currently using `VATEX` only.

## Dense caption generation
Take the ucf-crime dataset as an example, set the paths in the below command and run accordingly
```bash
CUDA_VISIBLE_DEVICES=2 python ./src/tasks/dense_caption_mass.py \
--resume_checkpoint /home/acsguser/Codes/SwinBERT/models/table1/tvc/best-checkpoint/model.bin \
--eval_model_dir /home/acsguser/Codes/SwinBERT/models/table1/tvc/best-checkpoint/ \
--dataset_path /home/acsguser/Codes/SwinBERT/datasets/Crime/data/ \
--caption_file /home/acsguser/Codes/SwinBERT/datasets/Crime/RTFM_train_caption/all_captions.txt \
--file_type video \
--file_format mp4 \
--do_lower_case \
--dense_caption \
--do_test
```

## Generate sentence embeddings based on captions
Run
```bash
python ./src/tasks/generate_caption_se.py --dataset ucf --is_test
```
