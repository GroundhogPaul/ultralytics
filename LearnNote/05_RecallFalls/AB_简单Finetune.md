# 一、起因和现象

详见文档 AA_RecalFalls

用yolo官方结构，无论是176x还是160x，recall都降了很多，手机上不能用了。

# 二、研究finetuing的影响

### finetune之前

训练文件夹：runs/LapaTrain/160x1000epoch_MileStone

train/box_loss = 0.615

train/cls_loss = 0.361

train/dfl_loss = 0.989

trainR 0.997（略，由于没有数据增强，所以不具有借鉴意义）

valR = 0.745    （来自AC_引入多分辨率验证集.md的多分辨率数据集）

### 学习率设置1：lr0 = 0.005,    lrf = 0.016（最终学习率 0.000085）

训练文件夹：runs/LapaTrain/160xFinetune_0.005_0.016

Loss反冲峰值 0.880，Loss最终收敛 0.602    最后100个epoch的斜率 ??

train/box_loss = 0.602

train/cls_loss = 0.352

train/dfl_loss = 0.98

valR = 0.565

### 学习率设置2：lr0 = 0.001,     lrf = 0.08（最终学习率 0.000085）

训练文件夹：runs/LapaTrain/160xFinetune_0.001_0.08

Loss反冲峰值 0.684，Loss最终收敛 0.597    最后100个epoch的斜率 ??

train/box_loss = 0.597

train/cls_loss = 0.347

train/dfl_loss = 0.98 

valR = 0.643

## 2.2 分析

越finetune，train集上表现变好了（loss都下降），而val集上表现却变差了（Recall下降），是非常明显的过拟合
