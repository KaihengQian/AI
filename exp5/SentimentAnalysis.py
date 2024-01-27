import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from collections import defaultdict

from PIL import Image
from nltk.tokenize import word_tokenize

from model.Multimodal import MultimodalModel


# 初始化
def initialize():
    current_directory = os.getcwd()

    folder_name1 = "evaluation"
    folder_path1 = os.path.join(current_directory, folder_name1)
    if not os.path.exists(folder_path1):
        os.mkdir(folder_path1)
    else:
        print(f"Folder {folder_name1} has existed.")

    folder_name2 = "result"
    folder_path2 = os.path.join(current_directory, folder_name2)
    if not os.path.exists(folder_path2):
        os.mkdir(folder_path2)
    else:
        print(f"Folder {folder_name2} has existed.")

    # 如果计算机安装有CUDA，则使用CUDA进行接下来的全部训练，否则使用CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('using device:', device)
    print()

    return device


# 词表
class Vocab:
    def __init__(self, text, min_freq=1, reserved_tokens=None):
        self.idx2token = list()
        self.token2idx = {}
        token_freqs = defaultdict(int)
        self.UNK_TOKEN = '<UNK>'

        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1

        unique_tokens = [self.UNK_TOKEN]
        if reserved_tokens:
            unique_tokens += reserved_tokens
        # 过滤掉出现频率过低的词
        unique_tokens += [token for token, freq in token_freqs.items() if freq >= min_freq]

        for token in unique_tokens:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1
        self.unk = self.token2idx[self.UNK_TOKEN]

    def __len__(self):
        return len(self.idx2token)

    def __getitem__(self, token):
        return self.token2idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.idx2token[idx] for idx in ids]


class SentimentDataset(Dataset):
    def __init__(self, dataframe, vocab, text_path, image_path, transform=None):
        self.guids = dataframe['guid'].values
        self.tags = dataframe['tag'].values
        self.vocab = vocab
        self.text_path = text_path
        self.image_path = image_path
        self.transform = transform
        self.label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}  # 标签映射

    def __len__(self):
        return len(self.guids)

    def __getitem__(self, index):
        guid = self.guids[index]

        text_path = self.text_path + str(guid) + '.txt'
        with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = [line.strip() for line in f.readlines() if line.strip()]  # 去除空行
        text = ' '.join(text)

        tokens = word_tokenize(text)  # 分词
        tokens = [word for word in tokens if word.isalpha()]  # 去除标点符号和数字
        tokens = [word.lower() for word in tokens]  # 转换为小写字母
        ids = self.vocab.convert_tokens_to_ids(tokens)  # 利用词表进行转换
        ids = torch.tensor(ids, dtype=torch.long)

        image_path = self.image_path + str(guid) + '.jpg'
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        tag = self.tags[index]
        if pd.isna(tag):
            label = 3
        else:
            label = self.label_mapping[tag]

        return ids, image, label


def collate_fn(batch, pad_token):
    text = [sample[0] for sample in batch]
    padded_text = pad_sequence(text, batch_first=True, padding_value=pad_token)
    attention_mask = (padded_text != pad_token).type(torch.float32)

    images = torch.stack([sample[1] for sample in batch])

    labels = torch.tensor([sample[2] for sample in batch], dtype=torch.long)

    return padded_text, attention_mask, images, labels


# 加载数据集
def load_dataset(train_set_path, test_set_path, text_path, image_path, vocab, PAD_TOKEN):
    # 读取训练数据
    train_df = pd.read_csv(train_set_path, index_col=False)

    # 将训练数据划分为训练集和验证集，固定划分（8:2）
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # 读取测试数据，即测试集
    test_df = pd.read_csv(test_set_path, index_col=False)

    num_train = train_df.shape[0]
    num_val = val_df.shape[0]
    num_test = test_df.shape[0]

    print('X_train: ', num_train)
    print('X_val: ', num_val)
    print('X_test: ', num_test)
    print()

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = SentimentDataset(train_df, text_path, image_path, image_transform)
    val_dataset = SentimentDataset(val_df, text_path, image_path, image_transform)
    test_dataset = SentimentDataset(test_df, text_path, image_path, image_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              collate_fn=lambda batch: collate_fn(batch, pad_token=vocab[PAD_TOKEN]))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                            collate_fn=lambda batch: collate_fn(batch, pad_token=vocab[PAD_TOKEN]))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,
                             collate_fn=lambda batch: collate_fn(batch, pad_token=vocab[PAD_TOKEN]))

    return train_loader, val_loader, test_loader, num_train, num_val, num_test


# 检查模型准确率
def check_accuracy(device, model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for text, attention_mask, images, labels in loader:
            text = text.to(device)
            attention_mask = attention_mask.to(device)
            images = images.to(device)
            labels = labels.to(device)
            scores = model(text, attention_mask, images)
            _, preds = scores.max(1)
            num_correct += (preds == labels).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        return acc


# 训练模型
def train_model(device, model, optimizer, train_loader, val_loader, num, epochs):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for text, attention_mask, images, labels in train_loader:
            # 将待更新参数的梯度置为零
            optimizer.zero_grad()

            text = text.to(device)
            attention_mask = attention_mask.to(device)
            images = images.to(device)
            labels = labels.to(device)

            scores = model(text, attention_mask, images)

            loss = F.cross_entropy(scores, labels)

            # 反向传播，计算梯度
            loss.backward()

            # 利用梯度更新参数
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num
        print(f'Epoch: {epoch + 1}/{epochs}')
        print(f'Train Loss: {avg_loss}')

    val_acc = check_accuracy(device, model, val_loader)

    return model, val_acc


def predict(device, model, loader, test_set_path, result_path):
    all_preds = []

    model.eval()
    with torch.no_grad():
        for text, attention_mask, images, labels in loader:
            text = text.to(device)
            attention_mask = attention_mask.to(device)
            images = images.to(device)
            scores = model(text, attention_mask, images)
            _, preds = scores.max(1)
            all_preds.append(preds.cpu().numpy())

    tag = np.concatenate(all_preds)
    label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}  # 标签映射
    tag_mapping = []
    for i in range(len(tag)):
        tag_mapping.append(label_mapping[tag[i]])

    test_df = pd.read_csv(test_set_path, index_col=False)
    test_df['tag'] = tag
    test_df.to_csv(result_path, index=False)


def main(args):
    train_set_path = 'dataset/train.txt'
    test_set_path = 'dataset/test_without_label.txt'
    text_path = 'dataset/data/'
    image_path = 'dataset/data/'
    evaluation_path = 'evaluation/' + args.text.upper() + '_' + args.image.upper() + '_evaluation.txt'
    result_path = 'result/prediction.txt'

    device = initialize()

    # 根据训练数据建立词表
    train_df = pd.read_csv(train_set_path, index_col=False)
    guids = train_df['guid'].values
    train_text = []
    for guid in guids:
        path = text_path + str(guid) + '.txt'
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = [line.strip() for line in f.readlines() if line.strip()]  # 去除空行
        text = ' '.join(text)
        tokens = word_tokenize(text)  # 分词
        tokens = [word for word in tokens if word.isalpha()]  # 去除标点符号和数字
        tokens = [word.lower() for word in tokens]  # 转换为小写字母
        train_text.append(tokens)

    PAD_TOKEN = '<PAD>'  # 填充项
    reserved_tokens = [PAD_TOKEN]
    vocab = Vocab(train_text, reserved_tokens=reserved_tokens)

    train_loader, val_loader, test_loader, num_train, num_val, num_test = (
        load_dataset(train_set_path, test_set_path, text_path, image_path, vocab, PAD_TOKEN))

    vocab_size = len(vocab)
    text_embed_dim = 256
    text_hidden_dim = 512
    output_dim = 128
    num_classes = 3

    model = MultimodalModel(args.text, args.image, vocab_size, text_embed_dim, text_hidden_dim, output_dim, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    model, val_acc = train_model(device, model, optimizer, train_loader, val_loader, num_train, args.epochs)
    val_acc = round(val_acc * 100, 2)

    print('Validation:')
    print('Validation Accuracy: ', val_acc, '%')

    with open(evaluation_path, 'w') as f:
        f.write('Validation:\n')
        f.write('Validation Accuracy: ' + str(val_acc) + '%')

    predict(device, model, test_loader, test_set_path, result_path)
