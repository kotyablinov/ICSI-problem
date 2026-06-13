"""
Оценка качества алгоритма сегментации ооцита и иглы. На основе opencv.

Скрипт обрабатывает изображения из размеченного датасета, обнаруживает
ооцит как крупный овальный объект и иглу как две почти горизонтальные
линии. На основе найденных объектов формируется предсказанная маска,
где 0 обозначает фон, 1 — ооцит, 2 — иглу.

Предсказанные маски сравниваются с эталонными масками датасета.
По всем найденным парам изображение–маска рассчитываются средние
значения IoU, Dice и общей точности классификации пикселей.
"""

import cv2
import numpy as np
import os
from math import atan2, degrees
from pathlib import Path

# === ДАТАСЕТ ===
ROOT = Path("D:/diplom/dataset_segment")
IMG_DIR = ROOT / "images"
MASK_DIR = ROOT / "masks"

# Параметры для "толщины" иглы на предсказанной маске
NEEDLE_THICKNESS = 3

prev_lines = None  # память для иглы между кадрами (для датасета будем сбрасывать на каждом изображении)


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
    min_line_length = 50
    angle_tol_deg = 15
    right_margin = 0.85

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


# ====== ДОБАВЛЕНО (минимально): GT->labels, pred mask, метрики ======
def load_gt_label_mask(mask_path: Path):
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Не прочиталась маска: {mask_path}")

    # Цветная маска -> по двум крупнейшим ненулевым цветам: 1=ооцит, 2=игла
    if m.ndim == 3:
        flat = m.reshape(-1, 3)
        colors, counts = np.unique(flat, axis=0, return_counts=True)

        bg = np.array([0, 0, 0], dtype=colors.dtype)
        non_bg = [(colors[i], counts[i]) for i in range(len(colors)) if not np.array_equal(colors[i], bg)]
        if len(non_bg) == 0:
            return np.zeros(m.shape[:2], dtype=np.uint8)

        non_bg.sort(key=lambda x: x[1], reverse=True)
        oocyte_color = non_bg[0][0]
        needle_color = non_bg[1][0] if len(non_bg) > 1 else None

        gt = np.zeros(m.shape[:2], dtype=np.uint8)
        gt[(m[:, :, 0] == oocyte_color[0]) & (m[:, :, 1] == oocyte_color[1]) & (m[:, :, 2] == oocyte_color[2])] = 1
        if needle_color is not None:
            gt[(m[:, :, 0] == needle_color[0]) & (m[:, :, 1] == needle_color[1]) & (m[:, :, 2] == needle_color[2])] = 2
        return gt

    # Одноканальная: если уже 0/1/2 — оставляем; иначе нормализуем по двум крупнейшим ненулевым значениям
    vals, cnts = np.unique(m, return_counts=True)
    vals = vals.astype(np.int64)

    # если похоже на label-map 0..2
    if set(vals.tolist()).issubset({0, 1, 2}):
        return m.astype(np.uint8)

    nonzero = [(v, c) for v, c in zip(vals, cnts) if v != 0]
    gt = np.zeros_like(m, dtype=np.uint8)
    if len(nonzero) == 0:
        return gt

    nonzero.sort(key=lambda x: x[1], reverse=True)
    oocyte_val = nonzero[0][0]
    needle_val = nonzero[1][0] if len(nonzero) > 1 else None

    gt[m == oocyte_val] = 1
    if needle_val is not None:
        gt[m == needle_val] = 2
    return gt


def predict_label_mask_from_image(img_bgr):
    global prev_lines
    prev_lines = None  # для честной оценки на независимых картинках

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = preprocess(gray)

    # --- ооцит ---
    bin_img = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 35, 2)
    bin_inv = 255 - bin_img
    found_oocyte, ellipse = detect_oocyte_contour(bin_inv, gray)

    # --- игла ---
    found_lines, segs = detect_needle_segments(g, gray.shape[1])

    h, w = gray.shape
    pred = np.zeros((h, w), dtype=np.uint8)

    if found_oocyte:
        # fill эллипса значением 1
        cv2.ellipse(pred, ellipse, 1, thickness=-1)

    if found_lines:
        # иглу рисуем поверх ооцита значением 2
        for (x1, y1, x2, y2) in segs:
            ax1, ay1, ax2, ay2 = shift_left_edge_by_variation(gray, x1, y1, x2, y2)
            cv2.line(pred, (ax1, ay1), (ax2, ay2), 2, NEEDLE_THICKNESS, cv2.LINE_AA)

        # anti-alias может дать значения не {0,1,2} — приводим к {0,1,2}
        pred[pred > 2] = 2

    return pred


def mean_iou_dice_pixelacc(gt, pr, classes=(1, 2)):
    gt = gt.astype(np.int64)
    pr = pr.astype(np.int64)

    # Global pixel accuracy
    pixel_acc = float(np.mean(gt == pr))

    ious = []
    dices = []
    for c in classes:
        gt_c = (gt == c)
        pr_c = (pr == c)
        inter = np.logical_and(gt_c, pr_c).sum()
        union = np.logical_or(gt_c, pr_c).sum()
        denom = gt_c.sum() + pr_c.sum()

        iou = (inter / union) if union > 0 else 1.0
        dice = (2 * inter / denom) if denom > 0 else 1.0
        ious.append(float(iou))
        dices.append(float(dice))

    return float(np.mean(ious)), float(np.mean(dices)), pixel_acc


# ====== ОЦЕНКА ПО ДАТАСЕТУ: только средние значения ======
img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
img_paths = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in img_exts])

sum_iou = 0.0
sum_dice = 0.0
sum_acc = 0.0
n = 0

missing_masks = 0

for img_path in img_paths:
    mask_path = MASK_DIR / img_path.name
    if not mask_path.exists():
        cands = list(MASK_DIR.glob(img_path.stem + ".*"))
        if len(cands) == 0:
            missing_masks += 1
            continue
        mask_path = cands[0]

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        continue

    gt = load_gt_label_mask(mask_path)
    pr = predict_label_mask_from_image(img)

    if pr.shape != gt.shape:
        pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    miou, mdice, acc = mean_iou_dice_pixelacc(gt, pr, classes=(1, 2))

    sum_iou += miou
    sum_dice += mdice
    sum_acc += acc
    n += 1

if n == 0:
    print("Нет валидных пар image/mask для расчёта метрик. Проверьте соответствие имён файлов.")
    print(f"Изображений найдено: {len(img_paths)}, пропущено без масок: {missing_masks}")
else:
    print(f"Оценено пар image/mask: {n} (пропущено без масок: {missing_masks})")
    print(f"Mean IoU (oocyte+needle): {sum_iou / n:.4f}")
    print(f"Mean Dice (oocyte+needle): {sum_dice / n:.4f}")
    print(f"Pixel accuracy (global): {sum_acc / n:.4f}")