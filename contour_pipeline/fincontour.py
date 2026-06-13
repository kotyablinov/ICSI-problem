"""
Пайплайн для одного видео от определения ооцита в кадре, до выделения конутра.

Реализация последовательного обнаружения ооцита в кадре, 
сегментации изображения с выделением ооцита и иглы, с последующим сохранением,
полученных изображений с наложенной маской. Сегментация с помощью UNet++
c изображениями приводимыми к формату 512х512
"""

import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
import os
from pathlib import Path
import segmentation_models_pytorch as smp

weights_cls = r"D:\diplom\weights\resnet18_detect.pth"
weights_seg = r"D:\diplom\weights\unetpp_oocyte.pth"
video_path = r"D:\diplom\dataset\200.avi"
save_dir = r"D:\diplom\images"

os.makedirs(save_dir, exist_ok=True)

# Классификация ооцита ResNet18
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

model_cls = models.resnet18(weights=None)
model_cls.fc = nn.Linear(model_cls.fc.in_features, 2)
model_cls.load_state_dict(torch.load(weights_cls, map_location="cpu"))
model_cls.eval()

def predict(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(frame_rgb).unsqueeze(0)
    with torch.no_grad():
        output = model_cls(input_tensor)
        prob = torch.softmax(output, dim=1)[0, 1].item()
    return prob > 0.5, prob

# Сегментация U-Net++
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_seg = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=3
).to(DEVICE)

model_seg.load_state_dict(torch.load(weights_seg, map_location=DEVICE))
model_seg.eval()

def segment_oocyte(frame, model, target_size=512):
    img_resized = cv2.resize(frame, (target_size, target_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype("float32") / 255.0
    img_tensor = torch.tensor(np.transpose(img_norm, (2, 0, 1))).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    orig_h, orig_w = frame.shape[:2]
    mask_resized = cv2.resize(pred_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask_resized

def overlay_mask(image, mask, alpha=0.4):
    """
    Маска накладывается поверх изображения с заданной прозрачностью.
    Только в пикселях, где mask != 0, происходит изменение.
    """
    result = image.copy()

    color_oocyte = (0, 255, 0)   # зелёный
    color_needle = (0, 0, 255)   # красный

    for class_id, color in [(1, color_oocyte), (2, color_needle)]:
        class_mask = (mask == class_id)
        if np.any(class_mask):
            colored_layer = np.zeros_like(image, dtype=np.uint8)
            colored_layer[class_mask] = color

            result[class_mask] = cv2.addWeighted(
                image[class_mask], 1 - alpha,
                colored_layer[class_mask], alpha,
                0
            )

    return result

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Не удалось открыть видео: {video_path}")

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    found, score = predict(frame)
    label = f"OOCYTE FOUND ({score:.2f})" if found else f"OOCYTE NOT FOUND ({score:.2f})"
    color = (0, 255, 0) if found else (0, 0, 255)

    if found:
        mask = segment_oocyte(frame, model_seg)
        frame = overlay_mask(frame, mask)

    cv2.putText(frame, label, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    filename = os.path.join(save_dir, f"{frame_id:03d}.png")
    cv2.imwrite(filename, frame)
    frame_id += 1

cap.release()
