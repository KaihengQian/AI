# 目录结构
```
exp3/
│  README.md                        // 帮助文档
│  requirements.txt                 // 环境依赖
│  ImageClassification.py           // 主调用程序
│  main.py                          // 脚本程序
│  
├─adjust_parameters                 // 调参方法及过程
│      AlexNet.ipynb
│      DenseNet.ipynb
│      LeNet.ipynb
│      MobileNet.ipynb
│      ResNet.ipynb
│      
├─classifiers                       // CNN模型封装
│      AlexNet_classifier.py        // AlexNet
│      DenseNet_classifier.py       // DenseNet
│      LeNet_classifier.py          // LeNet
│      MobileNet_classifier.py      // MobileNet
│      ResNet_classifier.py         // ResNet
│          
├─data                              // 实验数据
│      t10k-images-idx3-ubyte.gz
│      t10k-images.idx3-ubyte
│      t10k-labels-idx1-ubyte.gz
│      t10k-labels.idx1-ubyte
│      train-images-idx3-ubyte.gz
│      train-images.idx3-ubyte
│      train-labels-idx1-ubyte.gz
│      train-labels.idx1-ubyte
│      
├─evaluation                        // 程序运行时创建，存放模型评估结果
└─models                            // 程序运行时创建，存放训练好的模型
```
# 项目部署
**（推荐使用Anaconda的PyTorch环境）**
```
(base) C:\Users\user> activate pytorch
```
## 进入工作目录
```
(pytorch) C:\Users\user> cd /your/path/exp3
```
## 安装环境依赖
```
(pytorch) /your/path/exp3> pip install -r requirements.txt
```
## 运行脚本程序
例如
```
(pytorch) /your/path/exp3> python main.py --model alexnet --lr 0.01 --dropout 0.0 --epochs 10
```
此处提供了四种参数可供输入：

1. --model：模型架构种类（可填alexnet，lenet，resnet，densenet，mobilenet），无默认值，必填。
2. --lr：学习率，默认值为0.01，选填。
3. --dropout：Dropout层的p值，默认值为0.0，选填。
4. --epochs：训练周期数，默认值为10，选填。
