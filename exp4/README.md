# 目录结构
```
exp4/
│  main.py                           // 脚本程序
│  requirements.txt                  // 环境依赖
│  README.md                         // 帮助文档
│  TextSummarization.py              // 主调用程序
│  
├─adjust_parameters                  // 调参方法及过程
│      BART.ipynb
│      Seq2Seq.ipynb
│      
├─data                               // 实验数据
│      test.csv
│      train.csv
│      
├─evaluation                         // 程序运行时创建，存放模型评估结果
└─model                              // 模型相关类、函数封装
    │  BART.py                       // BART模型封装类
    │  evaluation.py                 // 评估指标计算函数
    │  Seq2Seq.py                    // Seq2Seq模型封装类
    │  
    └─models--facebook--bart-base    // 从transformer库下载的BART模型 
        ├─.no_exist
        │  └─aadd2ab0ae0c8268c7c9693540e9904811f36177
        │          added_tokens.json
        │          generation_config.json
        │          special_tokens_map.json
        │          tokenizer_config.json
        │          
        ├─blobs
        ├─refs
        │      main
        │      
        └─snapshots
            └─aadd2ab0ae0c8268c7c9693540e9904811f36177
                    config.json
                    merges.txt
                    model.safetensors
                    vocab.json
```
# 项目部署
**（推荐使用Anaconda的PyTorch环境）**
```
(base) C:\Users\user> activate pytorch
```
## 进入工作目录
```
(pytorch) C:\Users\user> cd /your/path/exp4
```
## 安装环境依赖
```
(pytorch) /your/path/exp4> pip install -r requirements.txt
```
## 下载预训练模型
本实验使用到了从transformer库下载的'facebook/bart-base'预训练模型，请先点击链接下载models--facebook--bart-base文件夹，大小约为500M。
链接：[https://pan.baidu.com/s/1kNE2povKZfrRHlpmBHVgrQ?pwd=2i47](https://pan.baidu.com/s/1kNE2povKZfrRHlpmBHVgrQ?pwd=2i47) 
然后将整个文件夹放在/your/path/exp4/model/路径下。
## 运行脚本程序
例如
```
(pytorch) /your/path/exp4> python main.py --encoder lstm --decoder lstm --epochs 50
```
此处提供了两种参数可供输入：

1. --encoder：编码器种类（可填rnn，lstm，gru，bart），默认值为lstm，选填。
2. --decoder：解码器种类（可填rnn，lstm，gru，bart），默认值为lstm，选填。
3. --epochs：训练周期数，默认值为50，选填。

注：可运行的编码器/解码器组合包括rnn/rnn，rnn/gru，lstm/lstm，gru/rnn，gru/gru，bart/bart，请从中选择。如果选择bart/bart，建议设置epochs为10。
