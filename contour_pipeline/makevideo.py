"""
Объединение кадров с сегментацией в одно видео.

Скрипт для объединения обработанных изображений в видео,
на котором отображены и подсвечены ооцит и игла. Вспомогательная
визуальная функция, для демонстрации работы.
"""

import cv2
import os
from pathlib import Path

image_dir = Path(r"D:\diplom\images")
output_video = r"D:\diplom\result.avi"

# Получаем список PNG
images = sorted([img for img in image_dir.glob("*.png")])
if not images:
    raise ValueError("В папке нет PNG файлов")

# Читаем первую картинку, чтобы узнать размер кадра
frame = cv2.imread(str(images[0]))
h, w, _ = frame.shape

# Настраиваем видеозапись: 30 fps и кодек MJPG
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
video = cv2.VideoWriter(output_video, fourcc, 30, (w, h))

# Запись кадров
for img_path in images:
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"Не удалось прочитать {img_path}")
        continue
    video.write(frame)

video.release()
print(f"Видео сохранено: {output_video}")
