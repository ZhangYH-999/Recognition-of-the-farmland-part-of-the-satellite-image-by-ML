# 遥感图像地块分割与提取

## 开发环境

```bash
Windows 10
Python 3.7.4
g++ (x86_64-posix-seh-rev0, Built by MinGW-W64 project) 8.1.0
```

## 快速开始

### 安装前置 python 库文件

```\bash
pip install -r requirements.txt
```

如果您无法使用 pip 命令正确安装 gdal，请从[官网](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)下载相关 wheel 文件后手动安装

### 使用

可使用 `python Main.py --help` 命令查看使用说明

```bash
You can use '--help' to get help.
--retrain [model_type] [pad]: training model after regenerating dataset with specified pad
--evaluate [model_path]: evaluate the model
--predict [model_path] [file_path]: using model to predict image
```

#### 评估现有模型

我们在 `model` 文件夹下提供了可直接使用的随机森林模型，该模型使用步长为 5 的 csv 训练集训练得到，你也可以评估你重新训练的新模型，评估数据集为比赛所给的 8 张 tif 遥感图片和参考标签

示例如下：

```bash
python Main.py --evaluate ./model/RandomForest.pkl
```

#### 重新训练模型

我们提供了决策树和随机森林两种训练模型类型，您还可以根据需求设置不同步长来训练自己的模型，其将会被放置在 `model` 文件夹下，注意，新模型会覆盖旧模型，如需保存，请将旧模型保存至其他文件夹

示例如下：

```bash
python Main.py --retrain DecisionTree 5
```

```bash
python Main.py --retrain RandomForest 5
```

#### 预测

如果想使用您训练好的模型对图片进行预测，请将其放置在任意空文件夹下后，使用命令进行预测，其结果将会被储存至 `result/label` 文件夹下，可视化结果将会被放置在 `result/label_view` 文件夹下

示例如下：

```bash
python Main.py --predict ./model/RandomForest.pkl ./dataset/test
```

## 其他

若运行中报 `segment.exe` 相关错误，请在 `segment` 目录下执行

```bash
g++ -o segment ./segment.cpp
```

生成 `segment.exe` 文件，并将其移至 `segment/bin/release` 下重试

## 感谢

[Graph Based Image Segmentation by P. Felzenszwalb, D. Huttenlocher](http://cs.brown.edu/people/pfelzens/segment/)

