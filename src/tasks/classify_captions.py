import numpy as np
import os, random, time, datetime
import torch
import pandas as pd
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

SEED = 123
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-2
EPSILON = 1e-8
epochs = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# 将每一句转成数字 （大于limit_size做截断，小于limit_size做 Padding，加上首位两个标识，长度总共等于limit_size+2）
def convert_text_to_token(tokenizer, sentence, limit_size=128):  # max_len=128
    tokens = tokenizer.encode(sentence[:limit_size])  # 直接截断
    if len(tokens) < limit_size + 2:  # 补齐（pad的索引号就是0）
        tokens.extend([0] * (limit_size + 2 - len(tokens)))
    return tokens


# 建立mask
def attention_masks(input_ids):
    atten_masks = []
    for seq in input_ids:  # [1900, 32]
        seq_mask = [float(i > 0) for i in seq]  # PAD: 0; 否则: 1
        atten_masks.append(seq_mask)
    return atten_masks


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))  # 返回 hh:mm:ss 形式的时间


def compute_metrics(preds, labels):  # preds.shape=(16, 2) labels.shape=torch.Size([16, 1])
    prob = F.softmax(preds, dim=1)[:, 1]
    try:
        auc = roc_auc_score(labels, prob.detach())
    except ValueError:
        # print("ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.")
        auc = None
    preds = torch.max(preds, dim=1)[1]
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return acc, f1, auc


def train(model, optimizer):
    t0 = time.time()
    avg_loss, avg_acc, avg_f1, avg_auc = [], [], [], []

    model.train()
    for step, batch in enumerate(train_dataloader):
        # 每隔10个batch 输出一下所用时间.
        if step % 10 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[
            2].long().to(device)

        output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss, logits = output[0], output[1]

        avg_loss.append(loss.item())

        acc, f1, auc = compute_metrics(logits, b_labels)
        avg_acc.append(acc)
        avg_f1.append(f1)
        if auc is not None:
            avg_auc.append(auc)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)  # 大于1的梯度将其设为1.0, 以防梯度爆炸
        optimizer.step()  # 更新模型参数
        scheduler.step()  # 更新learning rate

    avg_acc = np.array(avg_acc).mean()
    avg_f1 = np.array(avg_f1).mean()
    avg_auc = np.array(avg_auc).mean()
    avg_loss = np.array(avg_loss).mean()
    return avg_loss, avg_acc, avg_f1, avg_auc


def evaluate(model):
    avg_acc, avg_f1, avg_auc = [], [], []
    model.eval()  # 表示进入测试模式

    with torch.no_grad():
        for batch in test_dataloader:
            b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[
                2].long().to(device)

            output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            # TODO concat output[0], 最后才计算Metrics
            acc, f1, auc = compute_metrics(output[0], b_labels)
            avg_acc.append(acc)
            avg_f1.append(f1)
            if auc is not None:
                avg_auc.append(auc)
    avg_acc = np.array(avg_acc).mean()
    avg_f1 = np.array(avg_f1).mean()
    avg_auc = np.array(avg_auc).mean()
    return avg_acc, avg_f1, avg_auc


def predict(sen):
    input_id = convert_text_to_token(tokenizer, sen)
    input_token = torch.tensor(input_id).long().to(device)  # torch.Size([128])

    atten_mask = [float(i > 0) for i in input_id]
    attention_token = torch.tensor(atten_mask).long().to(device)  # torch.Size([128])

    output = model(input_token.view(1, -1), token_type_ids=None,
                   attention_mask=attention_token.view(1, -1))  # torch.Size([128])->torch.Size([1, 128])否则会报错
    # print(output[0])
    return torch.max(output[0], dim=1)[1]


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

    train_inputs = torch.tensor([convert_text_to_token(tokenizer, sen) for sen in train_inputs])
    test_inputs = torch.tensor([convert_text_to_token(tokenizer, sen) for sen in test_inputs])
    train_masks = torch.tensor(attention_masks(train_inputs))
    test_masks = torch.tensor(attention_masks(test_inputs))
    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)

    # create DataLoader
    train_data = TensorDataset(train_inputs, train_masks, train_labels)  # 相当于zip，把inputs,masks和labels打包
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE*8)

    # create model and optimizer
    # checkpoint = torch.load(os.path.join(root_dir, "models", "classification", "classify_epoch_0.pth"))
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # num_labels表示2个分类，好评和差评
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=EPSILON)

    # training steps 的数量: [number of batches] x [number of epochs].
    total_steps = len(train_dataloader) * epochs

    # 设计 learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        train_loss, train_acc, train_f1, train_auc = train(model, optimizer)
        print('epoch={},train acc={},train f1={}, train auc={}, loss={}'.format(epoch, train_acc, train_f1, train_auc,
                                                                                train_loss))

        # save model
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler}
        torch.save(checkpoint, os.path.join(root_dir, "models", "classification", "classify_" + str(epoch) + "_" + str(BATCH_SIZE) + ".pth"))

        test_acc, test_f1, test_auc = evaluate(model)
        print("epoch={},test acc={}, test f1={}, test auc={}".format(epoch, test_acc, test_f1, test_auc))
