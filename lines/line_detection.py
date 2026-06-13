"""
Обнаружение ооцита и границ инъекционной иглы на кадрах видео.

Скрипт покадрово обрабатывает видео, нормализует яркость и контраст,
выделяет ооцит как крупный овальный объект в центральной части кадра
и ищет две почти горизонтальные границы иглы в правой части изображения.

Для поиска иглы применяются фильтр Габора, детектор границ Canny
и вероятностное преобразование Хафа. Найденный ооцит обозначается
эллипсом, а границы иглы — прямыми линиями. Обработанные кадры
сохраняются в отдельную директорию в формате PNG.
"""

import cv2
import numpy as np
import os
from math import atan2, degrees
from pathlib import Path

video_path = Path("D:/diplom/dataset/66.avi")
out_dir = Path("D:/diplom/lines/1_out")
save_every = 1
min_line_length = 50
angle_tol_deg = 15
right_margin = 0.85

prev_lines = None  # память для иглы между кадрами


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def preprocess(gray, target_mean=0.85, target_std=0.155):
    gray_f = gray.astype(np.float32) / 255.0
    mean_val = np.mean(gray_f)
    std_val = np.std(gray_f)

    alpha = np.clip(target_std / (std_val + 1e-6), 0.6, 2.0)
    beta = np.clip((target_mean - mean_val * alpha) * 255, -80, 80)

    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    mean_after = np.mean(adjusted.astype(np.float32) / 255.0)
    gamma = np.clip(1.0 + (target_mean - mean_after) * 1.5, 0.6, 1.4)
    img_gamma = np.power(adjusted.astype(np.float32) / 255.0, gamma)
    corrected = np.clip(img_gamma * 255, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(corrected, (5, 5), 0)
    return blurred


def detect_oocyte_contour(img, gray):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None
    h, w = img.shape[:2]
    best, best_score = None, -1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < h * w * 0.01:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0 or len(cnt) < 5:
            continue
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (ma, MA), angle = ellipse
        ratio = max(ma, MA) / min(ma, MA)
        if ratio > 2.5:
            continue
        if not (w * 0.2 < cx < w * 0.8 and h * 0.2 < cy < h * 0.8):
            continue
        circ = 4 * np.pi * area / (perim * perim)
        score = circ * np.sqrt(area)
        if score > best_score:
            best = ellipse
            best_score = score
    return (best is not None), best


def enhance_line_oriented(gray, theta_deg=0, ksize=31, sigma=4, lambd=10, gamma=0.4):
    """Gabor-фильтр для выделения вытянутых структур под углом theta_deg"""
    theta = np.deg2rad(theta_deg)
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
    return cv2.filter2D(gray, cv2.CV_8UC1, kernel)


def mean_intensity_along_line(gray, x1, y1, x2, y2):
    length = int(np.hypot(x2 - x1, y2 - y1))
    xs = np.linspace(x1, x2, length).astype(np.int32)
    ys = np.linspace(y1, y2, length).astype(np.int32)
    return np.mean(gray[ys, xs])


def detect_needle_segments(gray, image_width):
    """
    Улучшенный поиск иглы:
    - усиливает вытянутые структуры через Gabor
    - выделяет края (Canny)
    - фильтрует по яркости и углу
    - сохраняет прошлую позицию при потере
    """
    global prev_lines
    gabor = enhance_line_oriented(gray, theta_deg=0)
    edges = cv2.Canny(gabor, 30, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=40, maxLineGap=10)
    if lines is None:
        # fallback на предыдущие
        if prev_lines is not None:
            return True, prev_lines
        else:
            return False, []

    candidates = []
    for x1, y1, x2, y2 in [l[0] for l in lines]:
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length < min_line_length:
            continue
        angle = abs(degrees(atan2(dy, dx)))
        angle = angle if angle <= 90 else 180 - angle
        if angle > angle_tol_deg:
            continue

        # фильтр по яркости вдоль линии
        mean_val = mean_intensity_along_line(gray, x1, y1, x2, y2)
        if not (60 < mean_val < 190):
            continue

        # выбираем только правую часть изображения
        if x1 > image_width * right_margin or x2 > image_width * right_margin:
            candidates.append((x1, y1, x2, y2))

    if len(candidates) < 2:
        if prev_lines is not None:
            return True, prev_lines
        return False, []

    candidates.sort(key=lambda l: (l[1] + l[3]) / 2)
    top, bottom = candidates[0], candidates[-1]
    prev_lines = [top, bottom]
    return True, [top, bottom]


def shift_left_edge_by_variation(gray, x1, y1, x2, y2, win_size=15, diff_thresh=50):
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length < win_size * 2 + 1:
        return x1, y1, x2, y2
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    intensities, coords = [], []
    for i in range(length - win_size):
        xi = int(round(x1 + dx * i))
        yi = int(round(y1 + dy * i))
        if 0 <= xi < gray.shape[1] and 0 <= yi < gray.shape[0]:
            intensities.append(gray[yi, xi])
            coords.append((xi, yi))
        else:
            break
    intensities = np.array(intensities, dtype=np.float32)
    for i in range(win_size, len(intensities) - win_size):
        prev_win = intensities[i - win_size:i]
        curr_win = intensities[i:i + win_size]
        diff = abs(np.mean(curr_win) - np.mean(prev_win))
        if diff > diff_thresh:
            new_x1, new_y1 = coords[i]
            return new_x1, new_y1, x2, y2
    return x1, y1, x2, y2


# === Основной цикл ===
if not video_path.exists():
    raise FileNotFoundError(f"Видео не найдено: {video_path}")
ensure_dir(out_dir)

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError("Не удалось открыть видео")

idx, saved = 0, 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if idx % save_every != 0:
        idx += 1
        continue

    vis = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = preprocess(gray)

    # --- ооцит ---
    bin_img = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 35, 2)
    bin_inv = 255 - bin_img
    found_oocyte, ellipse = detect_oocyte_contour(bin_inv, gray)
    if found_oocyte:
        cv2.ellipse(vis, ellipse, (0, 255, 0), 2)

    # --- игла ---
    found_lines, segs = detect_needle_segments(g, gray.shape[1])
    if found_lines:
        adjusted_segs = []
        for (x1, y1, x2, y2) in segs:
            ax1, ay1, ax2, ay2 = shift_left_edge_by_variation(gray, x1, y1, x2, y2)
            adjusted_segs.append((ax1, ay1, ax2, ay2))
            cv2.line(vis, (ax1, ay1), (ax2, ay2), (0, 255, 255), 2, cv2.LINE_AA)

    if not found_oocyte and not found_lines:
        cv2.putText(vis, "No objects detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(vis,
                    f"Oocyte:{'OK' if found_oocyte else 'NA'} | Needle:{'OK' if found_lines else 'NA'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imwrite(str(out_dir / f"frame_{idx:06d}.png"), vis)
    saved += 1
    idx += 1

cap.release()
print(f"Готово: сохранено {saved} кадров в {out_dir}")
