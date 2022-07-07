## 快速上手

### 环境依赖

> python == 3.8

- oneflow == 0.8.0
- hydra-core == 1.0.6
- tensorboard == 2.4.1
- matplotlib == 3.4.1
- scikit-learn == 0.24.1
- transformers == 3.4.0
- jieba == 0.42.1


### 使用pip安装

- 安装依赖: ```pip install -r requirements.txt```


### 使用数据进行训练预测

- 存放数据： 可先下载数据 ```wget 120.27.214.45/Data/re/standard/data.tar.gz```在此目录下

  在 `data/origin` 文件夹下存放训练数据：

  - `train.csv`：存放训练数据集

  - `valid.csv`：存放验证数据集

  - `test.csv`：存放测试数据集

  - `relation.csv`：存放关系种类

- 开始训练：```python run.py``` (训练所用到参数都在conf文件夹中，**可通过修改 conf/config.yaml 文件中 参数 model 来指定使用以下模型之一（cnn, transformer, gcn)**)

- 每次训练的日志保存在 `logs` 文件夹内，模型结果保存在 `checkpoints` 文件夹内。

- 进行预测 ```python predict.py```, **可通过修改 conf/predict.yaml 文件中 参数 fp 来指定 checkpoint**


## 模型内容
1、PCNN

2、GCN

3、Transformer

