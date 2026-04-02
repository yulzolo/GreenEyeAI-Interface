import warnings
from ultralytics import YOLO

warnings.filterwarnings('ignore')

def train_yolo11():
    model = YOLO('yolo11m.pt')  


    model.train(
        data='dataset.yaml',  
        epochs=250,  
        imgsz=640,  
        batch=12,  
        device='8',  
        workers=8,  
        patience=50,  
        save=True,  
        save_period=10,  
        val=True,  
        project='runs/train',  
        name='yolo11_custom',  
        exist_ok=True,  
        pretrained=True,  
        optimizer='auto',  
        lr0=0.01,  
        lrf=0.01,  
        momentum=0.937,  
        weight_decay=0.0005,  
        warmup_epochs=3,  
        warmup_momentum=0.8,  
        warmup_bias_lr=0.1,  
        box=7.5,  
        cls=0.5,  
        dfl=1.5,  
        hsv_h=0.015, 
        hsv_s=0.7,  
        hsv_v=0.4,  
        degrees=0.0, 
        translate=0.1, 
        scale=0.5,  
        shear=0.0,  
        flipud=0.0, 
        fliplr=0.5,  
        mosaic=1.0, 
        mixup=0.0, 
        copy_paste=0.0,  
        close_mosaic=15,  
        resume=False, 
    )

if __name__ == '__main__':
    train_yolo11()
