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

b. 200~500epoch的每100个epoch的Loss下降侠侣


