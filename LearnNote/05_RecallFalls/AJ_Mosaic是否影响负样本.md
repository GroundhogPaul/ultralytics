# 一、起因和现象

# 二、不同Scale

### 结构：官方yolo11-pose结构，160分辨率1000epoch，永远使用Mosaic

训练结果文件夹：.\runs\LapaTrain\160xNegativeWithMosaic

**使用的数据集**：

train:

- D:/users/xiaoyaopan/PxyAI/DataSet/Lapa-yolo11/images/train
- D:/users/xiaoyaopan/PxyAI/DataSet/Lapa_Plus-yolo11/FalsePositive/images/train
- D:/users/xiaoyaopan/PxyAI/DataSet/Lapa_Plus-yolo11/Mosaic256/images/train
- D:/users/xiaoyaopan/PxyAI/DataSet/Lapa_Plus-yolo11/Mosaic255_9/images/train
- D:/users/xiaoyaopan/PxyAI/DataSet/Lapa_Plus-yolo11/Mosaic256_16/images/train
- D:/users/xiaoyaopan/PxyAI/DataSet/Lapa_Plus-yolo11/FalseNegative250721/images/train
  val: D:/users/xiaoyaopan/PxyAI/DataSet/Lapa-yolo11/valTinyMultiResolution/images/val 

**final epoch**

根据results.csv中的val集loss，这个就是最佳epoch

Train_ box_loss = 0.63

Train_cls_loss = 0.35

Train_dfl_loss = 0.95

**正样本测试集**

Val_box_loss = 0.75

Val_cls_loss = 0.44

Val_dfl_loss = 0.84

Val_Precision = 0.993

Val_Recall = 0.776

Val_mAP50 = 0.886

| 脸宽占比 | Recall | Score |
| ---- | ------ | ----- |
| 0.05 | 0.03   | 0.7   |
| 0.06 | 0.05   | 0.8   |
| 0.07 | 0.26   | 0.73  |
| 0.08 | 0.58   | 0.76  |
| 0.09 | 0.74   | 0.79  |
| 0.10 | 0.89   | 0.81  |
| 0.11 | 0.87   | 0.83  |
| 0.12 | 0.92   | 0.84  |

**负样本测试集**

True Positive: 118273

False Positive: 59

### 结构：官方yolo11-pose结构，160分辨率1000epoch，不打开任何Mosaic

训练结果文件夹：.\runs\LapaTrain\160xNegativeWithNoMosaic

**使用的数据集**：

train:

- D:/users/xiaoyaopan/PxyAI/DataSet/Lapa-yolo11/images/train
- D:/users/xiaoyaopan/PxyAI/DataSet/Lapa_Plus-yolo11/FalsePositive/images/train
- D:/users/xiaoyaopan/PxyAI/DataSet/Lapa_Plus-yolo11/Mosaic256/images/train
- D:/users/xiaoyaopan/PxyAI/DataSet/Lapa_Plus-yolo11/Mosaic255_9/images/train
- D:/users/xiaoyaopan/PxyAI/DataSet/Lapa_Plus-yolo11/Mosaic256_16/images/train
- D:/users/xiaoyaopan/PxyAI/DataSet/Lapa_Plus-yolo11/FalseNegative250721/images/train
  val: D:/users/xiaoyaopan/PxyAI/DataSet/Lapa-yolo11/valTinyMultiResolution/images/val

**750 epoch**

根据results.csv中的val集的class loss比较低的epoch

Train_ box_loss = 0.64

Train_cls_loss = 0.34

Train_dfl_loss = 0.96

**正样本测试集**

Val_box_loss = 0.86

Val_cls_loss = 0.39

Val_dfl_loss = 0.84

Val_Precision = 0.995

Val_Recall = 0.838

Val_mAP50 = 0.919

| 脸宽占比 | Recall | Score |
| ---- | ------ | ----- |
| 0.05 | 0.03   | 0.59  |
| 0.06 | 0.34   | 0.58  |
| 0.07 | 0.55   | 0.65  |
| 0.08 | 0.82   | 0.72  |
| 0.09 | 0.92   | 0.76  |
| 0.10 | 0.89   | 0.8   |
| 0.11 | 0.89   | 0.82  |
| 0.12 | 0.95   | 0.81  |

**负样本测试集**

True Positive: 118273

False Positive: 49
