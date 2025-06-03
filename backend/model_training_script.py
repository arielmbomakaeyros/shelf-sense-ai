from ultralytics import YOLO
import yaml
import os
import shutil
import json
import numpy as np
from sklearn.model_selection import KFold

# Ensure Google Drive is mounted (run once at notebook start)
# from google.colab import drive
# drive.mount('/content/drive')

# Function to convert NumPy types to Python types recursively
def convert_to_serializable(obj):
    if isinstance(obj, np.generic):  # NumPy scalars (e.g., np.float64)
        return obj.item()
    elif isinstance(obj, np.ndarray):  # NumPy arrays
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    return obj

# K-Fold Cross-Validation
data_yaml = '/content/drive/MyDrive/ShelfsenseAI.v1i.yolov11/data.yaml'
with open(data_yaml, 'r') as f:
    data_config = yaml.safe_load(f)

train_images = [os.path.join(data_config['train'], img) for img in os.listdir(data_config['train']) if img.endswith('.jpg')]
np.random.shuffle(train_images)

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=51)
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_images)):
    print(f"Training Fold {fold + 1}/{k}")
    fold_train_images = [train_images[i] for i in train_idx]
    fold_val_images = [train_images[i] for i in val_idx]
    fold_data = {
        'train': '/content/fold_train',
        'val': '/content/fold_val',
        'nc': data_config['nc'],
        'names': data_config['names']
    }

    # Clear previous fold data
    for path in ['/content/fold_train', '/content/fold_val']:
        if os.path.exists(path):
            shutil.rmtree(path)
    os.makedirs('/content/fold_train/images', exist_ok=True)
    os.makedirs('/content/fold_train/labels', exist_ok=True)
    os.makedirs('/content/fold_val/images', exist_ok=True)
    os.makedirs('/content/fold_val/labels', exist_ok=True)

    # Create symlinks for training images and labels
    for img in fold_train_images:
        os.symlink(img, f'/content/fold_train/images/{os.path.basename(img)}')
        label = img.replace('images', 'labels').replace('.jpg', '.txt')
        if os.path.exists(label):
            os.symlink(label, f'/content/fold_train/labels/{os.path.basename(label)}')
        else:
            print(f"Warning: Label file not found for {img}")

    # Create symlinks for validation images and labels
    for img in fold_val_images:
        os.symlink(img, f'/content/fold_val/images/{os.path.basename(img)}')
        label = img.replace('images', 'labels').replace('.jpg', '.txt')
        if os.path.exists(label):
            os.symlink(label, f'/content/fold_val/labels/{os.path.basename(label)}')
        else:
            print(f"Warning: Label file not found for {img}")

    with open('/content/fold_data.yaml', 'w') as f:
        yaml.dump(fold_data, f)

    # Train model
    model = YOLO('yolov8n.pt')  # Use yolov11n.pt if targeting YOLOv11
    results = model.train(
        data='/content/fold_data.yaml',
        epochs=30,
        imgsz=640,  # Reverted to 640 for better accuracy
        batch=16,
        device=0,
        patience=5,
        lr0=0.001,
        freeze=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        project='/content/drive/MyDrive/ShelfsenseAI.v1i.yolov11',
        name=f'fold_{fold + 1}',
        cache=True
    )

    # Validate
    metrics = model.val(data='/content/fold_data.yaml', split='val')
    fold_metrics.append({
        'mAP@0.5': metrics.box.map50,
        'precision': metrics.box.p,
        'recall': metrics.box.r
    })

# Save fold metrics
serializable_fold_metrics = convert_to_serializable(fold_metrics)
with open('/content/drive/MyDrive/ShelfsenseAI.v1i.yolov11/fold_metrics.json', 'w') as f:
    json.dump(serializable_fold_metrics, f, indent=2)

# Compute and print average metrics
avg_metrics = {
    'mAP@0.5': np.mean([m['mAP@0.5'] for m in fold_metrics]),
    'precision': np.mean([m['precision'] for m in fold_metrics]),
    'recall': np.mean([m['recall'] for m in fold_metrics])
}
print("Average Metrics:", convert_to_serializable(avg_metrics))

# Optional: Pseudo-Labeling (uncomment to enable)
"""
pseudo_image_dir = '/content/drive/MyDrive/shelfsense/open_images_v7/validation/data'
pseudo_label_dir = '/content/drive/MyDrive/ShelfsenseAI.v1i.yolov11/pseudo_labels'
os.makedirs(pseudo_label_dir, exist_ok=True)
train_pseudo_dir = '/content/drive/MyDrive/ShelfsenseAI.v1i.yolov11/train_pseudo/images'
os.makedirs(train_pseudo_dir, exist_ok=True)

model = YOLO('/content/drive/MyDrive/ShelfsenseAI.v1i.yolov11/fold_5/weights/best.pt')
for img_name in os.listdir(pseudo_image_dir):
    if img_name.endswith('.jpg'):
        img_path = os.path.join(pseudo_image_dir, img_name)
        results = model.predict(img_path, conf=0.5)
        with open(os.path.join(pseudo_label_dir, img_name.replace('.jpg', '.txt')), 'w') as f:
            for box in results[0].boxes:
                cls = int(box.cls)
                xywh = box.xywhn[0].tolist()
                f.write(f"{cls} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n")
        shutil.copy(img_path, os.path.join(train_pseudo_dir, img_name))

# Update data.yaml for pseudo-labels
with open(data_yaml, 'r') as f:
    data_config = yaml.safe_load(f)
data_config['train'] = [data_config['train'], train_pseudo_dir]
with open(data_yaml, 'w') as f:
    yaml.dump(data_config, f)
"""

# Final Training
model = YOLO('yolov8n.pt')  # Use yolov11n.pt if targeting YOLOv11
results = model.train(
    data='/content/drive/MyDrive/ShelfsenseAI.v1i.yolov11/data.yaml',
    epochs=50,
    imgsz=640,  # Reverted to 640
    batch=16,
    device=0,
    patience=10,
    lr0=0.001,
    freeze=10,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.2,
    project='/content/drive/MyDrive/ShelfsenseAI.v1i.yolov11',
    name='shelfsense_yolov8n_final',
    cache=True
)

# Validate on test set
metrics = model.val(data='/content/drive/MyDrive/ShelfsenseAI.v1i.yolov11/data.yaml', split='test')
print(f"Final Test mAP@0.5: {metrics.box.map50:.3f}")

# Export to ONNX
model.export(format='onnx', imgsz=640, dynamic=True, simplify=True)


















# from ultralytics import YOLO
# import yaml
# import os
# from sklearn.model_selection import KFold
# import numpy as np

# # Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# # K-Fold Cross-Validation
# data_yaml = '/content/drive/MyDrive/ShelfsenseAI.v1i.yolov11/data.yaml'
# with open(data_yaml, 'r') as f:
#     data_config = yaml.safe_load(f)

# train_images = [os.path.join(data_config['train'], img) for img in os.listdir(data_config['train']) if img.endswith('.jpg')]
# np.random.shuffle(train_images)

# k = 5
# kf = KFold(n_splits=k, shuffle=True, random_state=51)
# fold_metrics = []

# for fold, (train_idx, val_idx) in enumerate(kf.split(train_images)):
#     print(f"Training Fold {fold + 1}/{k}")
#     fold_train_images = [train_images[i] for i in train_idx]
#     fold_val_images = [train_images[i] for i in val_idx]
#     fold_data = {
#         'train': '/content/fold_train',
#         'val': '/content/fold_val',
#         'nc': data_config['nc'],
#         'names': data_config['names']
#     }

#     os.makedirs('/content/fold_train/images', exist_ok=True)
#     os.makedirs('/content/fold_train/labels', exist_ok=True)
#     os.makedirs('/content/fold_val/images', exist_ok=True)
#     os.makedirs('/content/fold_val/labels', exist_ok=True)

#     for img in fold_train_images:
#         os.symlink(img, f'/content/fold_train/images/{os.path.basename(img)}')
#         label = img.replace('images', 'labels').replace('.jpg', '.txt')
#         os.symlink(label, f'/content/fold_train/labels/{os.path.basename(label)}')
#     for img in fold_val_images:
#         os.symlink(img, f'/content/fold_val/images/{os.path.basename(img)}')
#         label = img.replace('images', 'labels').replace('.jpg', '.txt')
#         os.symlink(label, f'/content/fold_val/labels/{os.path.basename(label)}')

#     with open('/content/fold_data.yaml', 'w') as f:
#         yaml.dump(fold_data, f)

#     model = YOLO('yolov8n.pt')
#     results = model.train(
#         data='/content/fold_data.yaml',
#         epochs=30,
#         imgsz=640,
#         batch=16,
#         device=0,
#         patience=5,
#         lr0=0.001,
#         freeze=10,
#         hsv_h=0.015,
#         hsv_s=0.7,
#         hsv_v=0.4,
#         degrees=10,
#         translate=0.1,
#         scale=0.5,
#         shear=2.0,
#         flipud=0.5,
#         fliplr=0.5,
#         mosaic=1.0,
#         mixup=0.2,
#         project='/content/drive/MyDrive/ShelfsenseAI.v1i.yolov11',
#         name=f'fold_{fold + 1}',
#         cache=True
#     )

#     metrics = model.val(data='/content/fold_data.yaml', split='val')
#     fold_metrics.append({
#         'mAP@0.5': metrics.box.map50,
#         'precision': metrics.box.p,
#         'recall': metrics.box.r
#     })

# # Save fold metrics
# with open('/content/drive/MyDrive/ShelfsenseAI.v1i.yolov11/fold_metrics.json', 'w') as f:
#     json.dump(fold_metrics, f, indent=2)

# avg_metrics = {
#     'mAP@0.5': np.mean([m['mAP@0.5'] for m in fold_metrics]),
#     'precision': np.mean([m['precision'] for m in fold_metrics]),
#     'recall': np.mean([m['recall'] for m in fold_metrics])
# }
# print("Average Metrics:", avg_metrics)

# # Final training with pseudo-labels (if used)
# model = YOLO('yolov8n.pt')
# results = model.train(
#     data='/content/drive/MyDrive/ShelfsenseAI.v1i.yolov11/data.yaml',
#     epochs=50,
#     imgsz=224,
#     batch=16,
#     device=0,
#     patience=10,
#     lr0=0.001,
#     freeze=10,
#     hsv_h=0.015,
#     hsv_s=0.7,
#     hsv_v=0.4,
#     degrees=10,
#     translate=0.1,
#     scale=0.5,
#     shear=2.0,
#     flipud=0.5,
#     fliplr=0.5,
#     mosaic=1.0,
#     mixup=0.2,
#     project='/content/drive/MyDrive/ShelfsenseAI.v1i.yolov11',
#     name='shelfsense_yolov8n_final',
#     cache=True
# )

# # Validate on test set
# metrics = model.val(data='/content/drive/MyDrive/ShelfsenseAI.v1i.yolov11/data.yaml', split='test')
# print(f"Final Test mAP@0.5: {metrics.box.map50:.3f}")

# # Export to ONNX
# model.export(format='onnx', imgsz=224, dynamic=True, simplify=True)