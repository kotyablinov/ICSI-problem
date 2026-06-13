"""
Подсчет метрик для контурного-геометрического метода через r(theta).

Для оценки качества метода на датасете для сегментации высчитывались 
метрики по оценке точности полученного контура ооцита и иглы.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from numpy.fft import fft, ifft
import pandas as pd

# ====== Пути датасета ======
ROOT = Path("D:/diplom/dataset_segment")
IMG_DIR = ROOT / "images"
MASK_DIR = ROOT / "masks"

# (опционально) куда сохранять таблицу метрик/визуализации
OUT_DIR = Path("D:/diplom/metrics_math_seg")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ====== Ваши функции (без изменений) ======
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

# ====== Минимальные ДОБАВЛЕНИЯ: предсказание маски, GT-маска, метрики ======
def predict_oocyte_mask_from_bgr(img_bgr, keep=15, dev_thr=10):
    """
    Делает бинарную маску (0/1) ооцита по вашему контуру+сглаживанию.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    processed = preprocess_frame(gray)
    contour = find_oocyte_contour(processed)

    h, w = gray.shape
    pred = np.zeros((h, w), dtype=np.uint8)

    if contour is None or len(contour) < 5:
        return pred

    (cx, cy), _, _ = cv2.fitEllipse(contour)
    center = (cx, cy)

    theta, r = contour_to_polar(contour, center)
    if len(r) < 10:
        return pred

    r_smooth = fourier_smooth(r, keep=keep)
    theta_f, r_f = filter_by_deviation(theta, r, r_smooth, threshold=dev_thr)
    if len(r_f) < 10:
        return pred

    r_final = fourier_smooth(r_f, keep=keep)

    pts = np.stack([cx + r_final * np.cos(theta_f), cy + r_final * np.sin(theta_f)], axis=1)
    pts = np.rint(pts).astype(np.int32)
    pts = pts.reshape((-1, 1, 2))

    # fillPoly сам обрежет точки по границам изображения
    cv2.fillPoly(pred, [pts], 1)
    return pred

def load_gt_oocyte_mask(mask_path: Path):
    """
    Возвращает бинарную GT-маску (0/1) ооцита.
    Авто-логика:
      - если маска RGB и цвета разные: считает [0,0,0] фоном, а среди остальных цветов берет самый большой по площади как "ооцит"
      - если маска grayscale/label: среди значений >0 берет самое частое как "ооцит"
    """
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Не прочиталась маска: {mask_path}")

    # RGB/цветная разметка
    if m.ndim == 3:
        # если по сути grayscale в 3 каналах
        if np.array_equal(m[:, :, 0], m[:, :, 1]) and np.array_equal(m[:, :, 1], m[:, :, 2]):
            m = m[:, :, 0]
        else:
            flat = m.reshape(-1, 3)
            colors, counts = np.unique(flat, axis=0, return_counts=True)

            # фон = черный, если есть
            bg = np.array([0, 0, 0], dtype=colors.dtype)
            non_bg_idx = [i for i, c in enumerate(colors) if not np.array_equal(c, bg)]
            if len(non_bg_idx) == 0:
                return np.zeros(m.shape[:2], dtype=np.uint8)

            # берём самый большой класс среди не-фона (обычно ооцит > игла)
            best_i = non_bg_idx[int(np.argmax(counts[non_bg_idx]))]
            best_color = colors[best_i]
            gt = (m[:, :, 0] == best_color[0]) & (m[:, :, 1] == best_color[1]) & (m[:, :, 2] == best_color[2])
            return gt.astype(np.uint8)

    # grayscale / label map
    m = m.astype(np.int64)
    vals, cnts = np.unique(m, return_counts=True)

    # исключаем фон 0
    nonzero = vals[vals != 0]
    if len(nonzero) == 0:
        return np.zeros(m.shape, dtype=np.uint8)

    # берём самый частый ненулевой label как "ооцит" (обычно ооцит больше иглы)
    nz_cnts = cnts[vals != 0]
    best_val = nonzero[int(np.argmax(nz_cnts))]
    gt = (m == best_val)
    return gt.astype(np.uint8)

def seg_metrics_binary(gt01: np.ndarray, pr01: np.ndarray):
    """
    gt01, pr01: uint8 {0,1}
    IoU, Dice, PixelAcc
    """
    gt = gt01.astype(bool)
    pr = pr01.astype(bool)

    inter = np.logical_and(gt, pr).sum()
    union = np.logical_or(gt, pr).sum()
    gt_sum = gt.sum()
    pr_sum = pr.sum()

    iou = (inter / union) if union > 0 else 1.0
    dice = (2 * inter / (gt_sum + pr_sum)) if (gt_sum + pr_sum) > 0 else 1.0
    acc = (gt == pr).mean()

    return float(iou), float(dice), float(acc)

# ====== Прогон по датасету ======
img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
img_paths = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in img_exts])

rows = []
missing_masks = 0

for img_path in img_paths:
    # базовое сопоставление: одинаковое имя файла
    mask_path = MASK_DIR / img_path.name
    if not mask_path.exists():
        # fallback: по stem
        candidates = list(MASK_DIR.glob(img_path.stem + ".*"))
        if len(candidates) > 0:
            mask_path = candidates[0]
        else:
            missing_masks += 1
            continue

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        continue

    gt = load_gt_oocyte_mask(mask_path)
    pr = predict_oocyte_mask_from_bgr(img, keep=15, dev_thr=10)

    # если размеры вдруг разные — приводим pred к размеру gt (или наоборот)
    if pr.shape != gt.shape:
        pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    iou, dice, acc = seg_metrics_binary(gt, pr)

    rows.append({
        "image": img_path.name,
        "IoU": iou,
        "Dice": dice,
        "PixelAcc": acc,
        "gt_area": int(gt.sum()),
        "pred_area": int(pr.sum()),
    })

df = pd.DataFrame(rows)

print(f"Изображений найдено: {len(img_paths)}")
print(f"Пары image/mask обработано: {len(df)}")
print(f"Без масок пропущено: {missing_masks}")

if len(df) > 0:
    summary = df[["IoU", "Dice", "PixelAcc"]].agg(["mean", "std", "min", "max"])

    # сохранение
    df.to_csv(OUT_DIR / "metrics_math_vs_gt.csv", index=False)
    print(f"CSV сохранён: {OUT_DIR / 'metrics_math_vs_gt.csv'}")
else:
    print("Нет данных для метрик — проверьте соответствие имён файлов в images/ и masks/.")


if len(df) > 0:
    # 1) Сводка в консоль
    print("\n=== SUMMARY (console) ===")
    print(df[["IoU", "Dice", "PixelAcc"]].agg(["mean", "std", "min", "max"]).to_string())

    # 2) Худшие/лучшие примеры
    print("\n=== WORST 10 by IoU ===")
    print(df.sort_values("IoU").head(10)[["image", "IoU", "Dice", "PixelAcc", "gt_area", "pred_area"]].to_string(index=False))

    print("\n=== BEST 10 by IoU ===")
    print(df.sort_values("IoU", ascending=False).head(10)[["image", "IoU", "Dice", "PixelAcc", "gt_area", "pred_area"]].to_string(index=False))

    # 3) (опционально) Метрики по каждому изображению в консоль
    PRINT_PER_IMAGE = False
    if PRINT_PER_IMAGE:
        print("\n=== PER-IMAGE METRICS ===")
        print(df[["image", "IoU", "Dice", "PixelAcc", "gt_area", "pred_area"]].to_string(index=False))