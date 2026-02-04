import utilBasicRunPose
from ultralytics.models import YOLO


if __name__ == '__main__':
 
    override_params = {
        'epochs': 1000,
        'batch': 64,
        'name': '160x_ds16_2.9G',
        'patience': 0,
        # 'amp': False,
        'seed':0,
        'save_period':10,
        'val': True
    }

    data = './01_BasicRunPoseSetting/DataSet-lapa-pose-plus2.yaml'
# ----- scratch ----- #
    base_cfg = './01_BasicRunPoseSetting/02_TrainLapaConf2.yaml'
    # model = YOLO('./ultralytics/cfg/models/11/yolo11n-pose.yaml')
    # model = YOLO('./01_BasicRunPoseSetting/Model-yolo11-pose_Pad16.yaml')
    model = YOLO('./01_BasicRunPoseSetting/Model-yolo11n-pose_16ds_3.6G.yaml', task='pose')
    # model = YOLO('./01_BasicRunPoseSetting/Model-yolo11n-pose_16ds_attention.yaml') # delete 32x ds but remain attention
# ----- resume ----- #
    # base_cfg = './01_BasicRunPoseSetting/02_TrainLapaConf_Resume.yaml'
    # model = YOLO('./runs/LapaTrain/128x_Relu/weights/best.pt')
# ----- finetune ----- #
    # base_cfg = './01_BasicRunPoseSetting/02_TrainLapaConf_Finetune.yaml'
    # model = YOLO('./runs/LapaTrain/160x1000epoch_MileStone/weights/best.pt')

# ----- ----- #
    model.train(cfg=base_cfg, data = data, **override_params)