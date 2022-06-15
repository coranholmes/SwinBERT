from classify_train import *


def evaluate(model):
    model.eval()  # 表示进入测试模式

    outputs = []
    labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[
                2].long().to(device)

            output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            outputs.append(output[0])
            labels.append(b_labels)
        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)
        acc, f1, auc = compute_metrics(outputs, labels)
    return acc, f1, auc, outputs


if __name__ == "__main__":
    root_dir = "/home/acsguser/Codes/SwinBERT/"
    df = pd.read_csv(os.path.join(root_dir, "datasets", "Crime", "data.csv"), index_col=0)

    # split train and test set
    train_df = df[df.split == 'train']
    test_df = df[df.split == 'test']
    train_inputs, test_inputs, train_labels, test_labels = train_df.caption.values.tolist(), test_df.caption.values.tolist(), train_df.label.values.tolist(), test_df.label.values.tolist()
    print(len(train_inputs), len(test_inputs))

    cache_dir = '/home/acsguser/Codes/SwinBERT/models/captioning/bert-base-uncased/'
    tokenizer = BertTokenizer.from_pretrained(cache_dir)

    test_inputs = torch.tensor([convert_text_to_token(tokenizer, sen) for sen in test_inputs])

    test_masks = torch.tensor(attention_masks(test_inputs))

    test_labels = torch.tensor(test_labels)

    # create DataLoader
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

    # create model and optimizer
    # checkpoint = torch.load(os.path.join(root_dir, "models", "classification", "classify_epoch_0.pth"))
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model_path = os.path.join(root_dir, "models", "classification", "lr3e-5", "classify_3_16.pth")
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_acc, test_f1, test_auc, outputs = evaluate(model)
    preds = torch.max(outputs, dim=1)[1]
    test_df['pred'] = preds
    test_df.to_csv(os.path.join(root_dir, 'datasets','Crime','test_predict.csv'))
    print("test acc={}, test f1={}, test auc={}".format(test_acc, test_f1, test_auc))