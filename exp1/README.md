# 目录结构
```
exp1/
│  MyScript.py                              // 脚本程序
│  README.md                                // 帮助文档
│  requirements.txt                         // 环境依赖
│  TextClassification.py                    // 主调用程序
│  
├─classifiers                               // 分类器相关函数封装
│      decision_tree_classifier.py          // 决策树
│      logistic_regression_classifier.py    // 逻辑回归
│      mlp_classifier.py                    // 多层感知机
│      svm_linear_classifier.py             // 使用线性内核的支持向量机
│      svm_poly_classifier.py               // 使用多项式内核的支持向量机
│      svm_rbf_classifier.py                // 使用径向基函数内核的支持向量机
│      __init__.py
│      
├─evaluation                                // 程序运行时创建，存放K折交叉验证评估结果
├─exp1_data                                 // 实验数据
│      submit_sample.txt                    // 提交样例
│      test.txt                             // 测试数据
│      train_data.txt                       // 训练数据
│      
├─models                                    // 程序运行时创建，存放训练好的模型
├─report                                    // 程序运行时创建，存放验证集评估结果
└─result                                    // 程序运行时创建，存放预测结果
```
# 项目部署
**（推荐使用Anaconda的PyTorch环境）**
```
(base) C:\Users\user> activate pytorch
```
## 进入工作目录
```
(pytorch) C:\Users\user> /your/path/exp1
```
## 安装环境依赖
```
(pytorch) /your/path/exp1> pip install -r requirements.txt
```
## 运行脚本程序
```
(pytorch) /your/path/exp1> python MyScript.py
```
依次使用6种不同的机器学习分类算法进行文本分类，运行总时长约为7~8分钟。
