---
title: NLP中的迁移学习
date: 2019-06-11 23:10:53
categories: 
- 郭宇
tags: 
- 自然语言处理
- 迁移学习
---

# 1. 简介

## 1.1. 什么是迁移学习

* 假如一个人从来没有见过猫，那么这个人只需要最多10张照片，就能学习到什么是猫。
* 当你拿到一张照片的时候，在认识猫之前你已经认识了太多的东西了。

![image](http://wx2.sinaimg.cn/large/007mOCDngy1g3wa9yjg5fj30ns0hedwy.jpg)

![image](http://wx1.sinaimg.cn/large/007mOCDngy1g3v0v1d2nmj30w60mednx.jpg)

## 1.2. 为什么需要迁移学习
* 许多任务共享相同的语言知识
* 有标注的数据较少
* 经验上，引入迁移学习取得巨大成功

![image](http://ws2.sinaimg.cn/large/007mOCDngy1g3v13llgpnj31w80ucwo3.jpg)

## 1.3. 迁移学习分类

把我们当前要处理的NLP任务叫做T（T称为目标任务），迁移学习技术做的事是利用另一个任务S（S称为源任务）来提升任务T的效果，也即把S的信息迁移到T中。
* S无监督，且源数据和目标数据同时用于训练：此时主要就是自监督（self-supervised）学习技术，代表工作有CVT。
* **S无监督，且先使用源数据训练，再使用目标数据训练（序贯训练）：此时主要就是以BERT为代表的无监督模型预训练技术，代表工作有ELMo、ULMFiT、GPT/GPT-2、BERT、MASS、UNILM。（NLP中的迁移学习大多数属于该类）**
* S有监督，且源数据和目标数据同时用于训练：此时主要就是多任务（multi-task）学习技术，代表工作有MT-DNN。
* S有监督，且先使用源数据训练，再使用目标数据训练（序贯训练）：此时主要就是有监督模型预训练技术，类似CV中在ImageNet上有监督训练模型，然后把此模型迁移到其他任务上去的范式。代表工作有CoVe。

## 1.4. 序贯训练

![image](http://ws3.sinaimg.cn/large/007mOCDngy1g3v1figi17j31pa0py421.jpg)

### 1.4.1. 无监督预训练

* 易于得到大量无标注语料：维基百科，新闻，社交媒体...
* 使用语言模型且基于假设：你可以通过某个词的上下文得知该词的意思

### 1.4.2. 有监督预训练

* CV中常见，NLP中缺乏大量标注数据集故不常用
* 机器翻译
* S与T相关程度高（eg.从一个Q&A数据集训练出的模型迁移到另一个数据集上）

### 1.4.3. 目标任务
* 情感分析，问答，文本分类...

# 2. 预训练

![image](http://ws4.sinaimg.cn/large/007mOCDngy1g3v1tsrjpcj31v00w4dt9.jpg)

## 2.1. 为什么语言模型可以work

* 即使是对于人来说，这样是比较困难的任务
* 语言模型会将可能词汇压缩进一个向量（e.g.“They walked down the street to ???”）
* 为了实现该任务，模型将被迫学习语义，情感等信息

## 2.2. 生成向量学到了什么

![image](http://ws3.sinaimg.cn/large/007mOCDngy1g3vuw0x2c7j30j80goq4w.jpg)

![image](http://wx4.sinaimg.cn/large/007mOCDngy1g3vv3r7kzbj30t80ikq61.jpg)

**BERT:fine tuning>最后四层连接>最后四层相加>倒数第二层>倒数第一层**

# 3. 迁移

* 模型结构修改：预训练的模型应该修改到怎样的程度来做下游任务？
* 优化策略：哪些层的参数需要训练，又该按怎样的顺序？

## 3.1. 模型

* 不调整预训练模型中间层
* 修改模型中间层

### 3.1.1. unchanged

1. 移除预训练模型最后一层

![image](http://ws4.sinaimg.cn/large/007mOCDngy1g3vwih3cjzj30k20i0afu.jpg)

2. 根据特定任务添加相关层

![image](http://wx2.sinaimg.cn/large/007mOCDngy1g3vwmzntf4j30na0q0wmx.jpg)

### 3.1.2. changed

* 原因：目标任务与预训练任务要求结构差距较大（S:单句输入，T：多句输入）
* Use the pretrained model weights to initialize as much as possible of a structurally different target task model

![image](http://ws4.sinaimg.cn/large/007mOCDngy1g3wapk82mlj30t40lqdrr.jpg)

## 3.2. 优化

可选操作
* 哪些层的权重需要更新(feature based ,fine tuning)
* 什么时候更新，怎样更新(from top to bottom,gradual ufreezing)


### 3.2.1. which weights

**to tune or not to tune(the pretrained weights)**

1.不微调
![image](http://wx4.sinaimg.cn/large/007mOCDngy1g3w4rmaw3ij30ne0k0tg9.jpg)
![image](http://wx3.sinaimg.cn/large/007mOCDngy1g3w4tcfdqnj30sw0kunbt.jpg)

2.微调


### 3.2.3. 更新策略

动机：避免重写有用信息，最大化正向迁移
相关概念：catastrophic forgetting（resNet）

* 整个训练阶段冻结除最后层之外所有层
* 自底向上每次训练一层
* 自上向下逐步解冻