import utilBasicRunPose
from ultralytics.models import YOLO


if __name__ == '__main__':
    # model = YOLO(model='./ultralytics/weights/yolo11n-pose.pt')
    # model.train(data="./ultralytics/cfg/datasets/coco-pose.yaml", epochs=3, imgsz=320)

    model = YOLO('./ultralytics/cfg/models/11/yolo11-pose.yaml')
    model.train(
                # data='ultralytics/cfg/datasets/coco8-pose.yaml',
                data='./ultralytics/cfg/datasets/lapa8-pose.yaml',
                # data="./ultralytics/cfg/datasets/lapa-pose.yaml",
                task='pose',
                pretrained = False,

                # cache=False,
                imgsz=128,
                translate=0.5,
                scale=0.5,
                degrees=45.0,
                flipud=0.5,
                # fliplr=0.5,
                mosaic = 0.0,
                close_mosaic=-1,

                single_cls=True,

                epochs=5,
                batch=16,
                # device = '0',
                workers = 1,

                # optimizer='SGD', # using SGD
                # project='runs/train-pose',
                # name='exp',
                )