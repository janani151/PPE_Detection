from ultralytics import YOLO

# Load the YOLOv8 model (pretrained weights)
model = YOLO('yolov8n.pt')

# Start training on CPU with optimized settings
results = model.train(
    data='data.yaml',
    epochs=50,
    batch=4,  # Reduced batch size to avoid memory issues
    imgsz=640,
    project='runs/train',
    name='best',
    save=True,
    workers=0,  # Avoid multiprocessing issues
    device='cpu',  # Force CPU usage
    verbose=True  # Enable detailed logging
)
