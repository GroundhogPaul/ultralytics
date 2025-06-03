import utilBasicRunPose
from ultralytics.models import YOLO


if __name__ == '__main__':
    print("I am here.\n")
    # model = YOLO(model='./ultralytics/weights/yolo11n-pose.pt')
    # model.train(data="./ultralytics/cfg/datasets/coco-pose.yaml", epochs=3, imgsz=320)

    model = YOLO('./ultralytics/cfg/models/11/yolo11-pose.yaml')
    model.train(
            # data='ultralytics/cfg/datasets/coco8-pose.yaml',
            data='./ultralytics/cfg/datasets/lapa8-pose.yaml',
            # data="./ultralytics/cfg/datasets/lapa-pose.yaml",
                # cache=False,
                imgsz=320,
                epochs=5,
                # batch=8,
                # task='pose',
                close_mosaic=1,
                # mosaic = 0.0,
                pretrained = False,
                # device='0',
                # optimizer='SGD', # using SGD
                # project='runs/train-pose',
                # name='exp',
                )