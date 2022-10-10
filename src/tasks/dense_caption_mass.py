from run_caption_VidSwinBert_inference import *
from run_caption_VidSwinBert_inference import _online_video_decode, _transforms
import pickle, json, sys
import gc, cv2


def inference(args, video_path, model, tokenizer, tensorizer):
    cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token,
                                         tokenizer.pad_token, tokenizer.mask_token, '.'])

    model.float()
    model.eval()

    if args.file_type == "video": # video
        frames_lst = _online_video_decode(args, video_path)  # {list: no. of segments}, each shape=[T,C,W,H] T is the no of frames (=64)
    elif args.file_type == "image":  # image
        frames = []
        file_list = os.listdir(video_path)
        for img_path in file_list:
            img_path = os.path.join(video_path, img_path)
            img = Image.open(img_path)
            # img = cv2.imread(img_path)
            frames.append(img)
        print(len(frames))

        frame_lst = []
        for i in range(0, len(frames), 16):
            frame_lst.append(frames[i: min(i+args.dense_caption_num, len(frames))])
        del frames
        gc.collect()
        res = []
        print("frame_lst successfully prepared: ", len(frame_lst))
        i = 0
        while len(frame_lst) > 0: # {list: no. of segments}
            i += 1
            # print(i)
            ind_frames = frame_lst.pop(0)
            # ind_frames = [frame.to_rgb().to_ndarray() for frame in ind_frames]
            ind_frames = torch.as_tensor(np.stack(ind_frames))  # {list:64}
            if len(ind_frames) < args.dense_caption_num:
                # repeat last frame until the size becomes 64
                repeat_last = ind_frames[-1].repeat(args.dense_caption_num - len(ind_frames), 1, 1, 1)
                ind_frames = torch.cat([ind_frames, repeat_last], dim=0)
            # print("before append")
            res.append(ind_frames)
        frames_lst = res  # {list: no. of segments}, each list: (64, 480, 856, 3)
    else:
        raise ValueError("file_type should be either video or image")

    if not isinstance(frames_lst, list):
        frames_lst = [frames_lst]

    print("Length of frames list:", len(frames_lst))

    res = []
    for frames in frames_lst:
        preproc_frames = _transforms(args, frames)  # shape=[T,C,224,224] 对视频帧进行变形，小视频变为224x224
        data_sample = tensorizer.tensorize_example_e2e('', preproc_frames)  # {tuple:5}
        data_sample = tuple(t.to(args.device) for t in data_sample)
        with torch.no_grad():
            inputs = {'is_decode': True,
                      'input_ids': data_sample[0][None, :],  # shape=[1,max_len], 初始全为[MASK]
                      'attention_mask': data_sample[1][None, :],  # shape=[1,1598,1598]
                      'token_type_ids': data_sample[2][None, :],  # shape=[1,max_len]
                      'img_feats': data_sample[3][None, :],  # shape=[1,64,3,224,224]
                      'masked_pos': data_sample[4][None, :],  # shape=[1,max_len]
                      'do_sample': False,
                      'bos_token_id': cls_token_id,
                      'pad_token_id': pad_token_id,
                      'eos_token_ids': [sep_token_id],
                      'mask_token_id': mask_token_id,
                      # for adding od labels
                      'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,
                      # hyperparameters of beam search
                      'max_length': args.max_gen_length,
                      'num_beams': args.num_beams,
                      "temperature": args.temperature,
                      "top_k": args.top_k,
                      "top_p": args.top_p,
                      "repetition_penalty": args.repetition_penalty,
                      "length_penalty": args.length_penalty,
                      "num_return_sequences": args.num_return_sequences,
                      "num_keep_best": args.num_keep_best,
                      }
            tic = time.time()
            outputs = model(**inputs)

            time_meter = time.time() - tic
            all_caps = outputs[0]  # batch_size * num_keep_best * max_len
            all_confs = torch.exp(outputs[1])

            for caps, confs in zip(all_caps, all_confs):
                for cap, conf in zip(caps, confs):
                    cap = tokenizer.decode(cap.tolist(), skip_special_tokens=True)
                    res.append(cap)
    return res

def main(args):
    args = update_existing_config_for_inference(args)  # models/table1/vatex/log/args.json 注意 max_gen_len

    # if it is a rerun, read the existing captions from the file
    if args.rerun:
        video_set = set()

        with open(args.old_caption_file) as f:
            for line in f:
                line = json.loads(line.strip())
                for k in line.keys():
                    if k.startswith('/home'):
                        video_set.add(k.strip()[53:])  # TODO: for Violence dataset, use video_set.add(k.strip()[53:])
                    else:
                        video_set.add(k.strip())
    cap_file = open(args.caption_file, "w")

    # global training_saver
    args.device = torch.device(args.device)
    # Setup CUDA, GPU & distributed training
    dist_init(args)
    check_arguments(args)
    set_seed(args.seed, args.num_gpus)
    fp16_trainning = None
    logger.info(
        "device: {}, n_gpu: {}, rank: {}, "
        "16-bits training: {}".format(
            args.device, args.num_gpus, get_rank(), fp16_trainning))

    if not is_main_process():
        logger.disabled = True

    logger.info(f"Pytorch version is: {torch.__version__}")
    logger.info(f"Cuda version is: {torch.version.cuda}")
    logger.info(f"cuDNN version is : {torch.backends.cudnn.version()}")

    # Get Video Swin model
    swin_model = get_swin_model(args)
    # Get BERT and tokenizer
    bert_model, config, tokenizer = get_bert_model(args)
    # build SwinBERT based on training configs
    vl_transformer = VideoTransformer(args, config, swin_model, bert_model)
    vl_transformer.freeze_backbone(freeze=args.freeze_backbone)

    # load weights for inference
    logger.info(f"Loading state dict from checkpoint {args.resume_checkpoint}")
    cpu_device = torch.device('cpu')
    pretrained_model = torch.load(args.resume_checkpoint, map_location=cpu_device)

    if isinstance(pretrained_model, dict):
        vl_transformer.load_state_dict(pretrained_model, strict=False)
    else:
        vl_transformer.load_state_dict(pretrained_model.state_dict(), strict=False)

    vl_transformer.to(args.device)
    vl_transformer.eval()

    tensorizer = build_tensorizer(args, tokenizer, is_train=False)

    ds_dir = args.dataset_path
    # Deal with images in subfolders
    if args.file_type == "image":
        logger.info(f"Loading images from {args.dataset_path}")
        file_list = os.listdir(args.dataset_path)
        for file_name in file_list:
            if file_name.endswith(args.file_format):
                args.test_video_fname = os.path.join(args.dataset_path, file_name)
                if args.rerun and args.test_video_fname in video_set:
                    print("Already process " + args.test_video_fname)
                    continue
                print("processing " + args.test_video_fname)
                cap = inference(args, args.test_video_fname, vl_transformer, tokenizer, tensorizer)
                print(cap)
                print("Length of caption list:", len(cap))
                text = {
                    args.test_video_fname: cap
                }
                cap_file.writelines(json.dumps(text) + "\n")
                cap_file.flush()

    # Deal with videos
    elif args.file_type == "video":
        logger.info(f"Loading videos from {args.dataset_path}")
        g = os.walk(ds_dir)

        violence_issue_lst = ["v=8cTqh9tMz_I__#1_label_A.mp4", "v=9eME1y6V-T4__#01-12-00_01-18-00_label_A.mp4", "Saving.Private.Ryan.1998__#02-29-31_02-30-55_label_B2-G-0.mp4"]

        for path, dir_list, file_list in g:
            for file_name in file_list:
                file_name = file_name.strip()
                args.test_video_fname = os.path.join(path, file_name)
                p_group = args.test_video_fname.split("/")
                path_new = "/".join([p_group[-2], p_group[-1]])

                if args.rerun and path_new in video_set:
                    print("Already process " + path_new)
                    continue

                # Exclude the corrupted video from Violence dataset
                skip_cur_video = False
                for corrupted_video in violence_issue_lst:
                    if corrupted_video in args.test_video_fname:
                        print("Skip " + args.test_video_fname)
                        skip_cur_video = True
                if skip_cur_video:
                    continue

                print("processing " + path_new)
                cap = inference(args, args.test_video_fname, vl_transformer, tokenizer, tensorizer)
                print(cap)
                print("Length of caption list:", len(cap))
                text = {
                    path_new: cap
                }
                cap_file.writelines(json.dumps(text) + "\n")
                cap_file.flush()
    else:
        raise ValueError("file_type should be either image or video")

    # with open("/home/acsguser/Codes/SwinBERT/datasets/captions2.pkl", "wb") as f:
    #     pickle.dump(cap_dict, f)


if __name__ == "__main__":
    shared_configs.shared_video_captioning_config(cbs=True, scst=True)
    args = get_custom_args(shared_configs)
    main(args)
