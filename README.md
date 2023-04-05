# Dense generation using SwinBERT
These codes are used in TEVAD paper for generate text features. However, these codes are not actively maintained and use at your own risk.
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
python ./src/tasks/dense_caption_mass.py \
--resume_checkpoint path/to/SwinBERT/models/table1/vatex/best-checkpoint/model.bin \
--eval_model_dir path/to/SwinBERT/models/table1/vatex/best-checkpoint/ \
--dataset_path path/to/SwinBERT/datasets/Crime/data/ \
--caption_file path/to/SwinBERT/datasets/Crime/RTFM_train_caption/vatex_all_captions.txt \
--file_type video \
--file_format mp4 \
--do_lower_case \
--dense_caption \
--do_test
```

## Generate sentence embeddings based on captions
Run
```bash
python src/tasks/generate_caption_se.py --dataset ucf \
--caption_path path/to/SwinBERT/datasets/Crime/RTFM_train_caption/vatex_all_captions.txt \
--output_path path/to/SwinBERT/datasets/Crime/RTFM_train_caption/sent_emb_n/
```
