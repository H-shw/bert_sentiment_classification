2022.7.31

增加知识蒸馏(KD)的方法 (`stu_train.py` 文件)。

由 预训练的 `bert-base-chinese` BERT 蒸馏给一个 参数量更小的随机初始化的 BERT (学习 Attention , Hidden_States 的表示) ， 同时结合标签数据的训练 。

2022.7.24

参考 https://blog.csdn.net/weixin_44750512/article/details/123236934 提供的 实现方法。

按照自己的方法重构了一下框架。