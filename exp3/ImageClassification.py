import os
from mlxtend.data import loadlocal_mnist
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as T

from classifiers.AlexNet_classifier import AlexNet
from classifiers.LeNet_classifier import LeNet
from classifiers.ResNet_classifier import BasicBlock, ResNet
from classifiers.DenseNet_classifier import DenseNet
from classifiers.MobileNet_classifier import MobileNetV1


# 初始化
def initialize():
    current_directory = os.getcwd()

    folder_name1 = "models"
    folder_path1 = os.path.join(current_directory, folder_name1)
    if not os.path.exists(folder_path1):
        os.mkdir(folder_path1)
    else:
        print(f"Folder {folder_name1} has existed.")

    folder_name2 = "evaluation"
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


class MnistDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# 加载数据集
def load_dataset(train_images_path, train_labels_path, test_images_path, test_labels_path):
    # 读取训练数据
    X_train, y_train = loadlocal_mnist(
        images_path=train_images_path,
        labels_path=train_labels_path)
    # 调整图像形状
    X_train = X_train.reshape(-1, 28, 28, 1)

    # 将训练数据划分为训练集和验证集，固定划分（8:2）
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 读取测试数据，即测试集
    X_test, y_test = loadlocal_mnist(
        images_path=test_images_path,
        labels_path=test_labels_path)
    # 调整图像形状
    X_test = X_test.reshape(-1, 28, 28, 1)

    print('X_train: ', X_train.shape)
    print('y_train: ', y_train.shape)
    print('X_val: ', X_val.shape)
    print('y_val: ', y_val.shape)
    print('X_test: ', X_test.shape)
    print('y_test: ', y_test.shape)
    print()

    # 将训练集、验证集、测试集分别包装为Dataloader
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

    train_dataset = MnistDataset(X_train, y_train, transform=transform)
    val_dataset = MnistDataset(X_val, y_val, transform=transform)
    test_dataset = MnistDataset(X_test, y_test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader


# 训练分类器
def train_classifier(device, train_loader, val_loader, classifier_type, lr, dropout, epochs, model_path):
    if classifier_type == 'alexnet':
        model = AlexNet(dropout=dropout)
        path = model_path[0]

    elif classifier_type == 'lenet':
        model = LeNet(dropout=dropout)
        path = model_path[1]

    elif classifier_type == 'resnet':
        model = ResNet(BasicBlock, num_blocks=[3, 4, 6, 3], dropout=dropout)
        path = model_path[2]

    elif classifier_type == 'densenet':
        model = DenseNet(block_config=[6, 12, 24, 16], dropout=dropout)
        path = model_path[3]

    elif classifier_type == 'mobilenet':
        model = MobileNetV1(dropout=dropout)
        path = model_path[4]

    else:
        print("Nonexistent model type!")
        return

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    model = model.to(device=device)
    for e in range(epochs):
        for t, (x, y) in enumerate(train_loader):
            model.train()
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # 将待更新参数的梯度置为零
            optimizer.zero_grad()

            # 反向传播，计算梯度
            loss.backward()

            # 利用梯度更新参数
            optimizer.step()

        print("Epoch %d/%d:" % (e + 1, epochs))
        train_acc = check_accuracy_alt(device, train_loader, model)
        val_acc = check_accuracy_alt(device, val_loader, model)
        print("Train Accuracy: %f Validation Accuracy: %f" % (train_acc, val_acc))

    # 保存训练好的模型参数
    torch.save(model.state_dict(), path)

    return model


# 检查模型准确率（使用验证集或测试集）
def check_accuracy(device, loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        res = 'Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc)
        return res


def check_accuracy_alt(device, loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        return acc


# 评价分类器
def evaluate_classifier(device, val_loader, test_loader, model, classifier_type, evaluation_path):
    if classifier_type == 'alexnet':
        path = evaluation_path[0]

    elif classifier_type == 'lenet':
        path = evaluation_path[1]

    elif classifier_type == 'resnet':
        path = evaluation_path[2]

    elif classifier_type == 'densenet':
        path = evaluation_path[3]

    elif classifier_type == 'mobilenet':
        path = evaluation_path[4]

    else:
        print("Nonexistent model type!")
        return

    with open(path, 'w') as f:
        optimizer_type = "Adam"
        # 使用验证集
        f.write(f"{classifier_type} with {optimizer_type} on validation set:\n")
        val_res = check_accuracy(device, val_loader, model)
        f.write(val_res + '\n\n')
        # 使用测试集
        f.write(f"{classifier_type} with {optimizer_type} on test set:\n")
        test_res = check_accuracy(device, test_loader, model)
        f.write(test_res)


def main(args):
    train_images_path = 'data/train-images.idx3-ubyte'
    train_labels_path = 'data/train-labels.idx1-ubyte'
    test_images_path = 'data/t10k-images.idx3-ubyte'
    test_labels_path = 'data/t10k-labels.idx1-ubyte'
    model_path = ['models/AlexNet_classifier.pth',
                  'models/LeNet_classifier.pth',
                  'models/ResNet_classifier.pth',
                  'models/DenseNet_classifier.pth',
                  'models/MobileNet_classifier.pth']
    evaluation_path = ['evaluation/AlexNet_evaluation.txt',
                       'evaluation/LeNet_evaluation.txt',
                       'evaluation/ResNet_evaluation.txt',
                       'evaluation/DenseNet_evaluation.txt',
                       'evaluation/MobileNet_evaluation.txt']

    device = initialize()

    train_loader, val_loader, test_loader = load_dataset(train_images_path, train_labels_path, test_images_path,
                                                         test_labels_path)

    model = train_classifier(device, train_loader, val_loader, args.model, args.lr, args.dropout, args.epochs, model_path)

    evaluate_classifier(device, val_loader, test_loader, model, args.model, evaluation_path)
