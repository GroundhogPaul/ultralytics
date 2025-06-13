import utilBasicRunPose
from ultralytics.models import YOLO


if __name__ == '__main__':
    # model = YOLO(model='./ultralytics/weights/yolo11n-pose.pt')
    # model.train(data="./ultralytics/cfg/datasets/coco-pose.yaml", epochs=3, imgsz=320)

    base_cfg = './01_BasicRunPose/02_TrainLapaConf.yaml'
 
    override_params = {
        # 'epochs': 30,
        # 'batch': 32
    }

    model = YOLO('./ultralytics/cfg/models/11/yolo11n-pose.yaml')
    model.train(cfg=base_cfg, **override_params)