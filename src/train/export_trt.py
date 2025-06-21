from ultralytics import YOLO
 
model = YOLO('models/best.pt')

# Export to TensorRT engine w/ dynamic batching 
model.export(
    format='engine',      # Export directly to TensorRT
    device=0,             # Use CUDA GPU
    imgsz=1280,            # Ensure matches training size
    dynamic=True,         # Enable dynamic batch size
    batch=4,             # Max batch size for dynamic TRT engine
    half=True             # FP16 for performance
)
 