"""
Обнаружение и визуализация контура ооцита контурно-геометрическим способом.

Код покадрово обрабатывает видео, повышает контраст изображения,
выполняет бинаризацию и ищет наиболее крупный круглый объект. Найденный
контур ооцита преобразуется в полярные координаты, очищается от выбросов
и сглаживается с помощью преобразования Фурье.

На каждый кадр наносится сглаженный контур и центр ооцита. Если на текущем
кадре контур не найден, выполняется попытка использовать контур,
обнаруженный на предыдущем кадре. Обработанные кадры сохраняются
в отдельную директорию в формате PNG.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from numpy.fft import fft, ifft

video_path = Path("D:/diplom/dataset/1.avi")
out_dir = Path("D:/diplom/lines/1_out")
os.makedirs(out_dir, exist_ok=True)

prev_center = None
prev_contour = None

def preprocess_frame(gray):
    """
    Контрастирование, усиление средних тонов и размытие для подготовки к анализу.
    """
    alpha = 1.6
    beta = 10
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    gray_f = adjusted.astype(np.float32) / 255.0
    gamma = 0.8
    corrected = np.power(gray_f, gamma)
    adjusted = np.clip(corrected * 255, 0, 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(adjusted, (3, 3), 0)
    return blurred

def find_oocyte_contour(gray_img):
    """Поиск наиболее круглого объекта"""
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    best_contour = None
    best_score = 0

    for contour in contours:
        if len(contour) < 5:
            continue
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if area > 3000 and 0.6 < circularity <= 1.2:
            if circularity > best_score:
                best_score = circularity
                best_contour = contour

    if best_contour is None:
        for contour in contours:
            if len(contour) < 5:
                continue
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if area > 2000 and 0.4 < circularity <= 1.3:
                if circularity > best_score:
                    best_score = circularity
                    best_contour = contour

    return best_contour

def contour_to_polar(contour, center):
    """Преобразование в r(θ)"""
    cx, cy = center
    angles, radii = [], []
    for pt in contour:
        x, y = pt[0]
        dx, dy = x - cx, y - cy
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        angles.append(theta)
        radii.append(r)
    angles = np.array(angles)
    radii = np.array(radii)
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    sort_idx = np.argsort(angles)
    return angles[sort_idx], radii[sort_idx]

def fourier_smooth(radii, keep=15):
    """Фурье-сглаживание"""
    fft_vals = fft(radii)
    fft_vals[keep:-keep] = 0
    return np.real(ifft(fft_vals))

def filter_by_deviation(theta, r, r_smooth, threshold=10):
    """Удалить точки с сильным отклонением"""
    filtered_theta, filtered_r = [], []
    for t, r_val, rs in zip(theta, r, r_smooth):
        if abs(r_val - rs) < threshold:
            filtered_theta.append(t)
            filtered_r.append(r_val)
    return np.array(filtered_theta), np.array(filtered_r)

def draw_polar_contour(img, r_vals, theta, center, color=(0, 255, 0)):
    """Отрисовка сглаженного r(θ)"""
    cx, cy = center
    pts = []
    for r, t in zip(r_vals, theta):
        x = int(cx + r * np.cos(t))
        y = int(cy + r * np.sin(t))
        pts.append((x, y))
    for i in range(len(pts)):
        cv2.line(img, pts[i], pts[(i + 1) % len(pts)], color, 2)

def is_oocyte_still_present(frame_gray, center, theta, r, prev_center, max_shift=50):
    """Проверка — остался ли ооцит"""
    mask = np.zeros_like(frame_gray, dtype=np.uint8)
    cx, cy = center
    pts = [(int(cx + ri * np.cos(ti)), int(cy + ri * np.sin(ti))) for ri, ti in zip(r, theta)]
    pts = np.array(pts).reshape((-1, 1, 2))
    cv2.drawContours(mask, [pts], -1, 255, -1)

    mean_val = cv2.mean(frame_gray, mask=mask)[0]
    std_val = cv2.meanStdDev(frame_gray, mask=mask)[1][0][0]

    dx, dy = cx - prev_center[0], cy - prev_center[1]
    center_shift = np.sqrt(dx**2 + dy**2)

    return mean_val < 170 and std_val > 8 and center_shift < max_shift

def process_frame(frame):
    global prev_center, prev_contour
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed = preprocess_frame(gray)
    contour = find_oocyte_contour(processed)

    if contour is not None and len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), _, _ = ellipse
        center = (cx, cy)
        theta, r = contour_to_polar(contour, center)
        r_smooth = fourier_smooth(r, keep=15)
        theta_filtered, r_filtered = filter_by_deviation(theta, r, r_smooth, threshold=10)
        r_final = fourier_smooth(r_filtered, keep=15)

        draw_polar_contour(frame, r_final, theta_filtered, center)
        cv2.circle(frame, (int(cx), int(cy)), 4, (255, 0, 0), -1)
        cv2.putText(frame, "OOCYTE DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        prev_center = center
        prev_contour = (theta_filtered, r_final)

    elif prev_center is not None and prev_contour is not None:
        theta, r = prev_contour
        if is_oocyte_still_present(gray, prev_center, theta, r, prev_center):
            draw_polar_contour(frame, r, theta, prev_center, color=(0, 200, 200))
            cv2.circle(frame, (int(prev_center[0]), int(prev_center[1])), 4, (255, 200, 0), -1)
            cv2.putText(frame, "OOCYTE RESTORED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 200), 2)
        else:
            prev_center = None
            prev_contour = None
            cv2.putText(frame, "OOCYTE LOST", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "OOCYTE NOT FOUND", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return frame

cap = cv2.VideoCapture(str(video_path))
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    output = process_frame(frame)
    out_path = out_dir / f"{frame_idx:04d}.png"
    cv2.imwrite(str(out_path), output)
    frame_idx += 1

cap.release()
print(f"Готово! Обработано {frame_idx} кадров.")
