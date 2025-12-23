# 任务：CNN分类器
## 任务需求
1. 我想实现一个CNN的分类器，来对岩石种类进行区分。具体的种类有四种，详见数据集：homework/train。其中garbage代表不是目标岩石的垃圾类。还有test dataset和val dataset

## 实施要求
1. model.py写CNN模型文件：
    1. 可以参考下面的代码写一个比较简单的cnn，修改这个实现分类功能：
        import torch
        import torch.nn as nn
        class LithologyCNN(nn.Module):
            def __init__(self, num_classes):
                super(LithologyCNN, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(256 * 28 * 28, 1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(1024, num_classes),
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
2. train.py写训练脚本：
    1. 基于最简单的pytorch实现，loss采用基础的分类loss，optimizer和其他超参数你自己决定
    2. 每隔一定步数进行一次valiadation，记录在val dataset里面取一定数量data的准确率。
    3. 记录loss，val正确率等关键数据，存在项目目录下,.csv格式。实时写入
    4. 支持多卡训练
    5. 训练前数据增强。类似这种：
        datagen = ImageDataGenerator(
                rotation_range=20,      # 随机旋转角度范围
                width_shift_range=0.1,  # 随机水平平移范围
                height_shift_range=0.1, # 随机垂直平移范围
                shear_range=0.1,        # 随机剪切强度
                zoom_range=0.1,         # 随机缩放范围
                horizontal_flip=True,   # 随机水平翻转
                vertical_flip=True,     # 随机垂直翻转
                fill_mode='nearest'     # 填充模式
            )

3. inference.py写推理脚本。输入是一张图片，输出是什么类别

## 修改
### roll 1
1. 增加保存检查点的功能，每个epoch都存一下
2. 没有实现数据增强。实现以下功能的数据增强：
    rotation_range=20,      # 随机旋转角度范围
    width_shift_range=0.1,  # 随机水平平移范围
    height_shift_range=0.1, # 随机垂直平移范围
    shear_range=0.1,        # 随机剪切强度
    zoom_range=0.1,         # 随机缩放范围
    horizontal_flip=True,   # 随机水平翻转
    vertical_flip=True,     # 随机垂直翻转
3. 多卡训练没有实现

### roll 2
1. 实现inference.py

### roll 3
1. 实现README.md。介绍如何所需的环境，以及如何启动训练和推理。尽可能简洁明了