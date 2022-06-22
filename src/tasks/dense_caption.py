from run_caption_VidSwinBert_inference import *
from run_caption_VidSwinBert_inference import _online_video_decode, _transforms

def inference(args, video_path, model, tokenizer, tensorizer):
    cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token,
        tokenizer.pad_token, tokenizer.mask_token, '.'])

    model.float()
    model.eval()
    frames_lst = _online_video_decode(args, video_path)  # shape=[T,C,W,H] T is the no of frames (=64)

    for frames in frames_lst:
        preproc_frames = _transforms(args, frames)  # shape=[T,C,224,224] 对视频帧进行变形，小视频变为224x224
        data_sample = tensorizer.tensorize_example_e2e('', preproc_frames)  # {tuple:5}
        data_sample = tuple(t.to(args.device) for t in data_sample)
        with torch.no_grad():

            inputs = {'is_decode': True,
                'input_ids': data_sample[0][None,:],  # shape=[1,max_len], 初始全为[MASK]
                'attention_mask': data_sample[1][None,:],  # shape=[1,1598,1598]
                'token_type_ids': data_sample[2][None,:],  # shape=[1,max_len]
                'img_feats': data_sample[3][None,:],  # shape=[1,64,3,224,224]
                'masked_pos': data_sample[4][None,:],  # shape=[1,max_len]
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
                    logger.info(f"Prediction: {cap}")
                    # logger.info(f"Conf: {conf.item()}")

        # logger.info(f"Inference model computing time: {time_meter} seconds")

def main(args):
    args = update_existing_config_for_inference(args)  # models/table1/vatex/log/args.json 注意 max_gen_len
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
    logger.info(f"cuDNN version is : {torch.backends.cudnn.version()}" )

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
    inference(args, args.test_video_fname, vl_transformer, tokenizer, tensorizer)


if __name__ == "__main__":
    shared_configs.shared_video_captioning_config(cbs=True, scst=True)
    args = get_custom_args(shared_configs)
    main(args)