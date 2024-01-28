# 目录结构

---

```
exp5/
│  main.py                           // 脚本程序
│  requirements.txt                  // 环境依赖
│  README.md                         // 帮助文档
│  SentimentAnalysis.py              // 主调用程序
│          
├─adjust_parameters                  // 调参方法及过程
│      EarlyFusion.ipynb
│      ImageModel.ipynb
│      LateFusion.ipynb
│      TextModel.ipynb
│      
├─dataset                            // 实验数据
│  │  test_without_label.txt
│  │  train.txt
│  │  
│  └─data
│          
├─evaluation                         // 程序运行时创建，存放模型评估结果
│      
├─model                              // 模型相关类、函数封装
│  │  ImageModel.py                  // 图像模型封装类
│  │  Multimodal.py                  // 多模态模型封装类
│  │  TextModel.py                   // 文本模型封装类
│  │  
│  └─models--bert-base-uncased       // 从transformers库下载的BERT模型
│      ├─blobs
│      ├─refs
│      │      main
│      │      
│      └─snapshots
│          └─1dbc166cf8765166998eff31ade2eb64c8a40076
│                  config.json
│                  model.safetensors
│                  
└─result                             // 程序运行时创建，存放测试集预测结果
```
# 项目部署

---

**（推荐使用Anaconda的PyTorch环境）**
```
(base) C:\Users\user> activate pytorch
```
## 进入工作目录
```
(pytorch) C:\Users\user> cd /your/path/exp5
```
## 安装环境依赖
```
(pytorch) /your/path/exp5> pip install -r requirements.txt
```
## 下载预训练模型
本实验使用到了从transformers库下载的'bert-base-uncased'预训练模型，请先点击链接下载models--bert-base-uncased文件夹，大小约为420M。
链接：[https://pan.baidu.com/s/1zwEHKVoSibVaWNxqQv_xPg?pwd=ydzp](https://pan.baidu.com/s/1zwEHKVoSibVaWNxqQv_xPg?pwd=ydzp)
然后将整个文件夹放在/your/path/exp5/model/路径下。
## 运行脚本程序
例如
```
(pytorch) /your/path/exp5> python main.py --text gru --image mobilenet --epochs 30
```
此处提供了两种参数可供输入：

1. --text：文本模型种类（可填rnn，lstm，gru，bert），默认值为gru，选填。
2. --image：图像模型种类（可填alexnet, mobilenet），默认值为mobilenet，选填。
3. --epochs：训练周期数，默认值为30，选填。

注：如果选择BERT作为文本模型，对于显存要求较高。
