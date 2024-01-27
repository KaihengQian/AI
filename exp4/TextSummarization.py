import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

from model.Seq2Seq import Seq2Seq, train, evaluate
from model.BART import BART


# 初始化
def initialize():
    current_directory = os.getcwd()

    folder_name1 = "evaluation"
    folder_path1 = os.path.join(current_directory, folder_name1)
    if not os.path.exists(folder_path1):
        os.mkdir(folder_path1)
    else:
        print(f"Folder {folder_name1} has existed.")

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


class DiagnosisDataset(Dataset):
    def __init__(self, dataframe, vocab, SOS_TOKEN, EOS_TOKEN):
        self.data = dataframe
        description = self.data['description'].str.split()
        diagnosis = self.data['diagnosis'].str.split()
        self.description_ids = [vocab.convert_tokens_to_ids(sentence) for sentence in description]
        self.diagnosis_ids = [[vocab[SOS_TOKEN]] + vocab.convert_tokens_to_ids(sentence) + [vocab[EOS_TOKEN]] for
                              sentence in diagnosis]

    def __len__(self):
        return len(self.description_ids)

    def __getitem__(self, index):
        return self.description_ids[index], self.diagnosis_ids[index]


def collate_fn(batch, pad_token):
    # 填充原始文本序列
    description = [torch.tensor(sample[0]) for sample in batch]
    description_len = torch.tensor([len(sample[0]) for sample in batch])
    padded_description = pad_sequence(description, batch_first=True, padding_value=pad_token)

    # 填充摘要序列
    diagnosis_input = [torch.tensor(sample[1][:-1]) for sample in batch]  # 去除末尾终止标记'<EOS>'，作为解码器输入
    diagnosis_output = [torch.tensor(sample[1][1:]) for sample in batch]  # 去除头部起始标记'<SOS>'，作为解码器输出
    padded_diagnosis_input = pad_sequence(diagnosis_input, batch_first=True, padding_value=pad_token)
    padded_diagnosis_output = pad_sequence(diagnosis_output, batch_first=True, padding_value=pad_token)
    # 为了方便后续计算，将摘要序列填充至与原始文本序列同长度
    padding_len = padded_description.size(1) - padded_diagnosis_input.size(1)
    padding = torch.full((padded_diagnosis_input.size(0), padding_len), pad_token)
    padded_diagnosis_input = torch.cat([padded_diagnosis_input, padding], dim=1)
    padded_diagnosis_output = torch.cat([padded_diagnosis_output, padding], dim=1)

    return padded_description, description_len, padded_diagnosis_input, padded_diagnosis_output


# 加载数据集
def load_dataset(train_set_path, test_set_path, vocab, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN):
    # 读取训练数据
    train_df = pd.read_csv(train_set_path, index_col=0)

    # 将训练数据划分为训练集和验证集，固定划分（8:2）
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # 读取测试数据，即测试集
    test_df = pd.read_csv(test_set_path, index_col=0)

    num_train = train_df.shape[0]
    num_val = val_df.shape[0]
    num_test = test_df.shape[0]

    print('X_train: ', num_train)
    print('X_val: ', num_val)
    print('X_test: ', num_test)
    print()

    train_dataset = DiagnosisDataset(train_df, vocab, SOS_TOKEN, EOS_TOKEN)
    val_dataset = DiagnosisDataset(val_df, vocab, SOS_TOKEN, EOS_TOKEN)
    test_dataset = DiagnosisDataset(test_df, vocab, SOS_TOKEN, EOS_TOKEN)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=lambda batch: collate_fn(batch, pad_token=vocab[PAD_TOKEN]))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=lambda batch: collate_fn(batch, pad_token=vocab[PAD_TOKEN]))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=lambda batch: collate_fn(batch, pad_token=vocab[PAD_TOKEN]))

    return train_loader, val_loader, test_loader, num_train, num_val, num_test


class DiagnosisDatasetBART(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        description = self.data.iloc[index]['description']
        diagnosis = self.data.iloc[index]['diagnosis']

        return description, diagnosis


# 加载数据集
def load_dataset_bart(train_set_path, test_set_path):
    # 读取训练数据
    train_df = pd.read_csv(train_set_path, index_col=0)

    # 将训练数据划分为训练集和验证集，固定划分（8:2）
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # 读取测试数据，即测试集
    test_df = pd.read_csv(test_set_path, index_col=0)

    num_train = train_df.shape[0]
    num_val = val_df.shape[0]
    num_test = test_df.shape[0]

    print('X_train: ', num_train)
    print('X_val: ', num_val)
    print('X_test: ', num_test)
    print()

    train_dataset = DiagnosisDatasetBART(train_df)
    val_dataset = DiagnosisDatasetBART(val_df)
    test_dataset = DiagnosisDatasetBART(test_df)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader, num_train, num_val, num_test


def main(args):
    train_set_path = 'data/train.csv'
    test_set_path = 'data/test.csv'
    evaluation_path = 'evaluation/' + args.encoder.upper() + '_' + args.decoder.upper() + '_evaluation.txt'

    device = initialize()

    if args.encoder == 'bart' and args.decoder == 'bart':  # 使用BART模型
        train_loader, val_loader, test_loader, num_train, num_val, num_test = load_dataset_bart(train_set_path, test_set_path)

        model = BART()

        train_loss = model.train(device, train_loader, num_train, args.epochs)

        print('Validation:')
        val_bleu4, val_rouge2 = model.evaluate(device, val_loader, num_val)
        print('Test:')
        test_bleu4, test_rouge2 = model.evaluate(device, test_loader, num_test)

    else:  # 使用Seq2Seq模型
        # 根据训练数据建立词表
        train_df = pd.read_csv(train_set_path, index_col=0)
        train_description = train_df['description'].str.split()
        train_diagnosis = train_df['diagnosis'].str.split()

        PAD_TOKEN = '<PAD>'  # 填充项
        SOS_TOKEN = '<SOS>'  # 起始标记
        EOS_TOKEN = '<EOS>'  # 终止标记
        reserved_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
        vocab = Vocab(train_description + train_diagnosis, reserved_tokens=reserved_tokens)

        train_loader, val_loader, test_loader, num_train, num_val, num_test = load_dataset(
            train_set_path, test_set_path, vocab, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN)

        vocab_size = len(vocab)
        embed_size = 256
        hidden_size = 512

        model = Seq2Seq(vocab_size, vocab_size, embed_size, hidden_size, args.encoder, args.decoder, vocab[SOS_TOKEN],
                        vocab[EOS_TOKEN], vocab[PAD_TOKEN])
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        train_loss, model = train(device, model, optimizer, train_loader, num_train, args.epochs, vocab[PAD_TOKEN])

        print('Validation:')
        val_bleu4, val_rouge2 = evaluate(device, model, val_loader, num_val, vocab, vocab[EOS_TOKEN], vocab[PAD_TOKEN], 'val')
        print('Test:')
        test_bleu4, test_rouge2 = evaluate(device, model, test_loader, num_test, vocab, vocab[EOS_TOKEN], vocab[PAD_TOKEN], 'test')

    with open(evaluation_path, 'w') as f:
        f.write('Encoder: ' + args.encoder.upper() + ', Decoder: ' + args.decoder.upper() + '\n')
        f.write('Validation:\n')
        f.write('BLEU-4: ' + str(val_bleu4) + '\n')
        f.write('ROUGE-2: ' + str(val_rouge2) + '\n')
        f.write('Test:\n')
        f.write('BLEU-4: ' + str(test_bleu4) + '\n')
        f.write('ROUGE-2: ' + str(test_rouge2))
