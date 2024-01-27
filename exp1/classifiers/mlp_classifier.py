import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import ExponentialLR


# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)  # 第一个隐藏层
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # 第二个隐藏层
        # self.dropout = nn.Dropout(p=0.1)  # 定义Dropout层，设置Dropout率
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        return x


# 训练MLP模型
def train_mlp_model(X_train, X_val, y_train, y_val, classifier_path, report_path, classifier_index):
    # 转换为torch tensor
    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
    y_val_tensor = torch.tensor(y_val, dtype=torch.int64)

    # 设置超参数
    input_dim = X_train.shape[1]
    output_dim = len(set(y_train))
    r = int(np.cbrt(input_dim / output_dim))
    hidden_dim1 = output_dim * np.square(r)
    hidden_dim2 = output_dim * r
    learning_rate = 0.01
    epochs = 13

    # 实例化模型、损失和优化器
    model = MLP(input_dim, hidden_dim1, hidden_dim2, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 执行He初始化
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

    # 设置学习率衰减
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    # 训练模型
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        # 执行学习率衰减
        scheduler.step()

    # 测试模型
    with torch.no_grad():
        predictions = model(X_val_tensor)
        _, predicted = torch.max(predictions, 1)
        accuracy = (predicted == y_val_tensor).sum().item() / y_val_tensor.size(0)
    print(f"准确度: {accuracy:.3f}")
    # 保存分类报告，包括精确度、召回率和F1分数
    report = classification_report(predicted.numpy(), y_val_tensor.numpy(), digits=3)
    with open(report_path[classifier_index], "w") as f:
        f.write(report)

    # 保存MLP分类器
    joblib.dump(model, classifier_path[classifier_index])


# 评价MLP模型
def evaluate_mlp_model(X_train, X_val, y_train, y_val):
    # 转换为torch tensor
    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
    y_val_tensor = torch.tensor(y_val, dtype=torch.int64)

    # 设置超参数
    input_dim = X_train.shape[1]
    output_dim = len(set(y_train))
    r = int(np.cbrt(input_dim / output_dim))
    hidden_dim1 = output_dim * np.square(r)
    hidden_dim2 = output_dim * r
    learning_rate = 0.01
    epochs = 13

    # 实例化模型、损失和优化器
    model = MLP(input_dim, hidden_dim1, hidden_dim2, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 执行He初始化
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

    # 设置学习率衰减
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    # 训练模型
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        # 执行学习率衰减
        scheduler.step()

    # 测试模型
    with torch.no_grad():
        predictions = model(X_val_tensor)
        _, predicted = torch.max(predictions, 1)
        accuracy = (predicted == y_val_tensor).sum().item() / y_val_tensor.size(0)

    return accuracy
