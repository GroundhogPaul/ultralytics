## 一、观察到反冲大和收敛慢

176x1000epoch_MileStone（以下称为训练1）和176xPruneStep2_1500epoch_L1shrinkBN（以下称为训练2）是两组训练，其中，训练2是以训练1的结果作为pretrain参数继续训练得到的，但没有用resume

在这两组训练中，都出现了这样一个情况。result中的train/box_loss和pose_loss都在接近训练后期的时候，下降得越来越快。具体的，以100为分段，直线拟合两次训练的500~1500epoch的loss，有如下结果

训练1：

epoch: 500, lr = 0.004555, slopeBox = -0.000183, slopePose = -0.001397
epoch: 600, lr = 0.003565, slopeBox = -0.000204, slopePose = -0.001501
epoch: 700, lr = 0.002575, slopeBox = -0.000248, slopePose = -0.001762
epoch: 800, lr = 0.001585, slopeBox = -0.000313, slopePose = -0.002220
epoch: 900, lr = 0.000595, slopeBox = -0.000470, slopePose = -0.003422

训练2：

epoch: 500, lr = 0.006370, slopeBox = -0.000093, slopePose = -0.000614
epoch: 600, lr = 0.005710, slopeBox = -0.000089, slopePose = -0.000612
epoch: 700, lr = 0.005050, slopeBox = -0.000107, slopePose = -0.000769
epoch: 800, lr = 0.004390, slopeBox = -0.000100, slopePose = -0.000776
epoch: 900, lr = 0.003730, slopeBox = -0.000125, slopePose = -0.000857
epoch: 1000, lr = 0.003070, slopeBox = -0.000137, slopePose = -0.000967
epoch: 1100, lr = 0.002410, slopeBox = -0.000154, slopePose = -0.001047
epoch: 1200, lr = 0.001750, slopeBox = -0.000188, slopePose = -0.001283
epoch: 1300, lr = 0.001090, slopeBox = -0.000250, slopePose = -0.001761
epoch: 1400, lr = 0.000430, slopeBox = -0.000365, slopePose = -0.002572

都在lr = 0.0005的时候，学习率达到最大。

于是进行如下实验：

以训练2的结果作为pretrained输入，其他参数不变，设置lr0=0.001， lrf=0.1，训练500个epoch，观察两个东西。 

a. 刚开始训练的反冲达到多少 

b. 50~500epoch的每50个epoch的Loss下降的斜率

## 二、不同学习率下的resume训练的反冲和收敛

**训练3**

**lr0 = 0.00002    lrf = 0.5**

Loss反冲达到的峰值 0.679；500个epoch收敛在0.669；lr和斜率关系如下：

epoch: 50, lr = 0.000018, slopeBox = -0.000015, slopePose = -0.000341
epoch: 100, lr = 0.000017, slopeBox = -0.000012, slopePose = -0.000118
epoch: 150, lr = 0.000017, slopeBox = -0.000010, slopePose = -0.000030
epoch: 200, lr = 0.000016, slopeBox = 0.000002, slopePose = 0.000012
epoch: 250, lr = 0.000015, slopeBox = 0.000012, slopePose = 0.000080
epoch: 300, lr = 0.000013, slopeBox = 0.000009, slopePose = 0.000011
epoch: 350, lr = 0.000013, slopeBox = 0.000029, slopePose = 0.000169
epoch: 400, lr = 0.000012, slopeBox = 0.000005, slopePose = -0.000025
epoch: 450, lr = 0.000010, slopeBox = 0.000009, slopePose = 0.000118

**训练5**

**lr0 = 0.0005    lrf = 0.2**

Loss反冲达到的峰值 TODO；500个epoch收敛在TODO；lr和斜率关系如下：

TODO

**训练6**

**lr0 = 0.001    lrf = 0.1**

Loss反冲达到的峰值 0.724；500个epoch后收敛在 0.673；lr和斜率关系如下：

epoch: 50, lr = 0.000865, slopeBox = -0.000061, slopePose = -0.000720
epoch: 100, lr = 0.000775, slopeBox = -0.000094, slopePose = -0.000587
epoch: 150, lr = 0.000685, slopeBox = -0.000089, slopePose = -0.000616
epoch: 200, lr = 0.000595, slopeBox = -0.000096, slopePose = -0.000643
epoch: 250, lr = 0.000505, slopeBox = -0.000098, slopePose = -0.000607
epoch: 300, lr = 0.000415, slopeBox = -0.000094, slopePose = -0.000775
epoch: 350, lr = 0.000325, slopeBox = -0.000102, slopePose = -0.000715
epoch: 400, lr = 0.000235, slopeBox = -0.000136, slopePose = -0.001041
epoch: 450, lr = 0.000145, slopeBox = -0.000137, slopePose = -0.000978

**训练2（就是那个训练2）**

**lr0 = 0.01    lrf = 0.01**

Loss反冲达到的峰值 0.948；500个epoch收敛在0.840；lr和斜率关系如下：

epoch: 500, lr = 0.006370, slopeBox = -0.000093, slopePose = -0.000614
epoch: 600, lr = 0.005710, slopeBox = -0.000089, slopePose = -0.000612
epoch: 700, lr = 0.005050, slopeBox = -0.000107, slopePose = -0.000769
epoch: 800, lr = 0.004390, slopeBox = -0.000100, slopePose = -0.000776
epoch: 900, lr = 0.003730, slopeBox = -0.000125, slopePose = -0.000857
epoch: 1000, lr = 0.003070, slopeBox = -0.000137, slopePose = -0.000967
epoch: 1100, lr = 0.002410, slopeBox = -0.000154, slopePose = -0.001047
epoch: 1200, lr = 0.001750, slopeBox = -0.000188, slopePose = -0.001283
epoch: 1300, lr = 0.001090, slopeBox = -0.000250, slopePose = -0.001761
epoch: 1400, lr = 0.000430, slopeBox = -0.000365, slopePose = -0.002572

**训练7（160分辨率，32倍下采样的，网络不同且是scratch，但可以观察一下）**

**lr0 = 0.01    lrf = 0.005**

1000epoch收敛在0.614；lr和斜率关系如下：

epoch: 500, lr = 0.004528, slopeBox = -0.000225, slopePose = -0.001582
epoch: 600, lr = 0.003533, slopeBox = -0.000254, slopePose = -0.001587
epoch: 700, lr = 0.002537, slopeBox = -0.000316, slopePose = -0.001959
epoch: 800, lr = 0.001543, slopeBox = -0.000410, slopePose = -0.002652
epoch: 900, lr = 0.000548, slopeBox = -0.000637, slopePose = -0.004349



**本节的初步结论：似乎是支持学习率稍微小一点的，但目前还没有详细分析，准备先做章节三的实验**

## 三、控制变量研究测试学习率对confusion mat（拟合训练集的完成度）的影响

保证同一个网络结构、随机种子、训练集，仅仅改变初始学习率和最终学习率

**lr0 = 0.01    lrf = 0.005**

**lr0 = 0.01    lrf = 0.01**

TP = 46182；FP = 85；FN = 55

**lr0 = 0.005    lrf = 0.005**


