import utilBasicRunPose
from ultralytics.models import YOLO


if __name__ == '__main__':
    # model = YOLO(model='./ultralytics/weights/yolo11n-pose.pt')
    # model.train(data="./ultralytics/cfg/datasets/coco-pose.yaml", epochs=3, imgsz=320)

    base_cfg = './01_BasicRunPose/02_TrainLapaConf.yaml'
 
    override_params = {
        'epochs': 1000,
        'batch': 64,
        # 'resume': True,
        'resume': False,
    }

    # model = YOLO('./ultralytics/cfg/models/11/yolo11n-pose.yaml')
    model = YOLO('./ultralytics/cfg/models/11/yolo11n-pose_16ds.yaml') # delete all the 32x
    # model = YOLO('./runs/LapaTrain/176x1000epoch_MileStone/weights/best.pt') # the 176x miletone 
    # model = YOLO('./runs/LapaTrain/176xToBeResumed/weights/best.pt')
    model.train(cfg=base_cfg, **override_params)