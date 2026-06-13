"""
Анализ структурных изменений ооцита до и после введения иглы.

Скрипт обрабатывает видео, обнаруживает ооцит с помощью ResNet-18,
сегментирует ооцит и иглу с помощью U-Net++, определяет момент нахождения
иглы внутри ооцита и выбирает стабильные кадры до введения и после
извлечения иглы.

Выбранные кадры совмещаются по центру ооцита. Во внутренней общей области
вычисляются метрики различия яркости, текстуры, энтропии и резкости.
Результаты вместе с известным исходом оплодотворения сохраняются
в файл structure_changes.csv.
"""

import cv2
import csv
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
from pathlib import Path
import segmentation_models_pytorch as smp

# ---------------- НАСТРОЙКИ ----------------

weights_cls = r"D:\diplom\weights\resnet18_detect.pth"
weights_seg = r"D:\diplom\weights\unetpp768_oocyte.pth"

dataset_dir = Path(r"D:\diplom\dataset")
videos_info_csv = Path(r"D:\diplom\dataset\videos_info.csv")
output_csv = Path(r"D:\diplom\structure_changes.csv")

inner_margin_px = 8

RADIUS_WEIGHT = 400
TIME_WEIGHT = 0.3
NEEDLE_WEIGHT = 3
NEEDLE_SAFE_DIST = 25

EPS = 1e-6

# ---------------- ЗАГРУЗКА ТАБЛИЦЫ С РЕЗУЛЬТАТАМИ ----------------

def load_video_results(csv_path):
    result_map = {}

    if not csv_path.exists():
        print(f"Предупреждение: не найден файл {csv_path}")
        return result_map

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,")
            delimiter = dialect.delimiter
        except Exception:
            delimiter = ";"

        reader = csv.DictReader(f, delimiter=delimiter)

        # пытаемся определить названия колонок
        fieldnames = [name.strip() for name in reader.fieldnames] if reader.fieldnames else []

        num_col = None
        fert_col = None

        for col in fieldnames:
            low = col.strip().lower()
            if low in ["№", "no", "num", "номер"]:
                num_col = col
            if low == "оплодотворение":
                fert_col = col

        if num_col is None:
            raise ValueError("В videos_info.csv не найден столбец с номером видео (№ / номер).")

        if fert_col is None:
            raise ValueError("В videos_info.csv не найден столбец 'оплодотворение'.")

        for row in reader:
            raw_id = str(row[num_col]).strip()
            raw_result = str(row[fert_col]).strip()

            if raw_id == "":
                continue

            # сохраняем и как строку, и как число без ведущих нулей
            result_map[raw_id] = raw_result
            try:
                result_map[str(int(float(raw_id)))] = raw_result
            except Exception:
                pass

    return result_map


video_result_map = load_video_results(videos_info_csv)

# ---------------- КЛАССИФИКАЦИЯ ООЦИТА ----------------

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


# ---------------- СЕГМЕНТАЦИЯ U-NET++ ----------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_seg = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=3
).to(DEVICE)

model_seg.load_state_dict(torch.load(weights_seg, map_location=DEVICE))
model_seg.eval()

def segment_oocyte(frame, model, target_size=768):
    img_resized = cv2.resize(frame, (target_size, target_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype("float32") / 255.0
    img_tensor = torch.tensor(np.transpose(img_norm, (2, 0, 1))).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    orig_h, orig_w = frame.shape[:2]
    mask_resized = cv2.resize(
        pred_mask.astype(np.uint8),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST
    )
    return mask_resized


# ---------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------------

def classify_needle_position(
        mask,
        contact_dilate=3,
        close_kernel_size=41,
        min_contact_pixels=10
    ):

    oocyte = (mask == 1).astype(np.uint8)
    needle = (mask == 2).astype(np.uint8)

    if oocyte.sum() == 0 or needle.sum() == 0:
        return "NEEDLE OUT"

    kernel_contact = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (contact_dilate, contact_dilate)
    )
    needle_dilated = cv2.dilate(needle, kernel_contact, iterations=1)
    contact = cv2.bitwise_and(needle_dilated, oocyte)
    contact_pixels = int(contact.sum())

    if contact_pixels < min_contact_pixels:
        return "NEEDLE OUT"

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
    )
    oocyte_filled = cv2.morphologyEx(oocyte, cv2.MORPH_CLOSE, kernel_close)
    oocyte_filled = cv2.morphologyEx(oocyte_filled, cv2.MORPH_CLOSE, kernel_contact)

    ys, xs = np.where(needle > 0)
    if len(xs) == 0:
        return "NEEDLE OUT"

    coords = np.column_stack((xs, ys))

    tip_idx = np.argmin(coords[:, 0])
    tip_x, tip_y = coords[tip_idx]

    h, w = mask.shape[:2]
    if 0 <= tip_x < w and 0 <= tip_y < h and oocyte_filled[tip_y, tip_x] > 0:
        return "NEEDLE IN"
    else:
        return "BOUNDARY"


def get_oocyte_features(mask):
    oocyte = (mask == 1).astype(np.uint8)
    cnts, _ = cv2.findContours(oocyte, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    pts = cnt[:, 0, :]
    dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)

    radius_mean = np.mean(dists)
    radius_std = np.std(dists)
    radius_std_norm = radius_std / (radius_mean + EPS)

    return {
        "cx": cx,
        "cy": cy,
        "radius_std_norm": radius_std_norm
    }


def get_needle_distance(mask):
    oocyte = (mask == 1).astype(np.uint8)
    needle = (mask == 2).astype(np.uint8)

    if oocyte.sum() == 0 or needle.sum() == 0:
        return None

    dist_map = cv2.distanceTransform((1 - oocyte).astype(np.uint8), cv2.DIST_L2, 5)

    ys, xs = np.where(needle > 0)
    if len(xs) == 0:
        return None

    return float(np.min(dist_map[ys, xs]))


def get_frame_by_id(video_path, frame_id):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Не удалось прочитать кадр {frame_id} из {video_path}")

    return frame


def get_centroid(mask_bin):
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def shift_image_and_mask(image, mask, dx, dy):
    h, w = mask.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    shifted_img = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    shifted_mask = cv2.warpAffine(
        mask.astype(np.uint8), M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return shifted_img, shifted_mask


def build_union_bbox(mask1, mask2, pad=15):
    union_mask = ((mask1 > 0) | (mask2 > 0)).astype(np.uint8)
    ys, xs = np.where(union_mask > 0)

    if len(xs) == 0:
        return None

    x1 = max(0, xs.min() - pad)
    y1 = max(0, ys.min() - pad)
    x2 = min(union_mask.shape[1], xs.max() + pad + 1)
    y2 = min(union_mask.shape[0], ys.max() + pad + 1)

    return x1, y1, x2, y2


def normalize_inside_mask(gray, mask):
    vals = gray[mask > 0].astype(np.float32)
    if len(vals) == 0:
        return gray.copy()

    mean = vals.mean()
    std = vals.std()

    out = gray.astype(np.float32).copy()
    out[mask > 0] = (out[mask > 0] - mean) / (std + EPS)

    vals2 = out[mask > 0]
    min_v = vals2.min()
    max_v = vals2.max()

    out[mask > 0] = 255.0 * (out[mask > 0] - min_v) / (max_v - min_v + EPS)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def masked_entropy(gray, mask):
    vals = gray[mask > 0]
    if len(vals) == 0:
        return None

    hist = cv2.calcHist([vals], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + EPS)
    p = p[p > 0]

    return float(-np.sum(p * np.log2(p)))


def laplacian_variance(gray, mask):
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    vals = lap[mask > 0]
    if len(vals) == 0:
        return None
    return float(np.var(vals))


def masked_mean(gray, mask):
    vals = gray[mask > 0]
    if len(vals) == 0:
        return None
    return float(np.mean(vals))


def masked_std(gray, mask):
    vals = gray[mask > 0]
    if len(vals) == 0:
        return None
    return float(np.std(vals))


def relative_delta(before, after, eps=EPS):
    if before is None or after is None:
        return None
    return float((after - before) / (abs(before) + eps))


def compare_oocyte_regions(frame1, mask1, frame2, mask2, inner_margin_px=8):
    oocyte1 = (mask1 == 1).astype(np.uint8)
    oocyte2 = (mask2 == 1).astype(np.uint8)

    if oocyte1.sum() == 0 or oocyte2.sum() == 0:
        return None

    c1 = get_centroid(oocyte1)
    c2 = get_centroid(oocyte2)
    if c1 is None or c2 is None:
        return None

    dx = int(round(c1[0] - c2[0]))
    dy = int(round(c1[1] - c2[1]))

    frame2_aligned, mask2_aligned = shift_image_and_mask(frame2, mask2, dx, dy)
    oocyte2_aligned = (mask2_aligned == 1).astype(np.uint8)

    bbox = build_union_bbox(oocyte1, oocyte2_aligned, pad=20)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox

    crop1 = frame1[y1:y2, x1:x2].copy()
    crop2 = frame2_aligned[y1:y2, x1:x2].copy()
    m1 = oocyte1[y1:y2, x1:x2].copy()
    m2 = oocyte2_aligned[y1:y2, x1:x2].copy()

    common_mask = ((m1 > 0) & (m2 > 0)).astype(np.uint8)
    if common_mask.sum() == 0:
        return None

    if inner_margin_px > 0:
        ksize = 2 * inner_margin_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        inner_mask = cv2.erode(common_mask, kernel, iterations=1)
    else:
        inner_mask = common_mask.copy()

    if inner_mask.sum() == 0:
        return None

    gray1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)

    gray1 = normalize_inside_mask(gray1, inner_mask)
    gray2 = normalize_inside_mask(gray2, inner_mask)

    diff = cv2.absdiff(gray1, gray2)
    diff_vals = diff[inner_mask > 0]

    mean_abs_diff = float(np.mean(diff_vals))
    max_abs_diff = float(np.max(diff_vals))
    changed_ratio = float(np.mean(diff_vals > 20))

    mean1 = masked_mean(gray1, inner_mask)
    mean2 = masked_mean(gray2, inner_mask)

    std1 = masked_std(gray1, inner_mask)
    std2 = masked_std(gray2, inner_mask)

    ent1 = masked_entropy(gray1, inner_mask)
    ent2 = masked_entropy(gray2, inner_mask)

    lap1 = laplacian_variance(gray1, inner_mask)
    lap2 = laplacian_variance(gray2, inner_mask)

    return {
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,
        "changed_ratio": changed_ratio,
        "mean_intensity_rel_delta": relative_delta(mean1, mean2),
        "std_intensity_rel_delta": relative_delta(std1, std2),
        "entropy_rel_delta": relative_delta(ent1, ent2),
        "lap_var_rel_delta": relative_delta(lap1, lap2),
        "shift_dx": dx,
        "shift_dy": dy,
        "inner_area_pixels": int(inner_mask.sum())
    }


# ---------------- ОБРАБОТКА ОДНОГО ВИДЕО ----------------

def process_video(video_path, inner_margin_px=8):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    needle_states = []
    features = []
    needle_distances = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        found, _ = predict(frame)

        if found:
            mask = segment_oocyte(frame, model_seg)
            needle_state = classify_needle_position(mask)
            feat = get_oocyte_features(mask)
            needle_dist = get_needle_distance(mask)
        else:
            needle_state = "NEEDLE OUT"
            feat = None
            needle_dist = None

        needle_states.append(needle_state)
        features.append(feat)
        needle_distances.append(needle_dist)

    cap.release()

    in_frames = [i for i, state in enumerate(needle_states) if state == "NEEDLE IN"]

    if len(in_frames) == 0:
        return {
            "status": "NO_NEEDLE_IN",
            "best_start_frame": None,
            "best_end_frame": None
        }

    first_in_frame = in_frames[0]
    last_in_frame = in_frames[-1]

    motions = [None] * len(features)
    for i in range(1, len(features)):
        if features[i] is None or features[i - 1] is None:
            continue

        motions[i] = np.hypot(
            features[i]["cx"] - features[i - 1]["cx"],
            features[i]["cy"] - features[i - 1]["cy"]
        )

    best_start_frame = None
    best_end_frame = None

    best_score = float("inf")
    for i in range(max(0, first_in_frame - 40), first_in_frame):
        f = features[i]
        m = motions[i]
        nd = needle_distances[i]

        if f is None or m is None:
            continue

        if nd is not None and nd < 15:
            continue

        needle_penalty = 0 if nd is None else max(0, NEEDLE_SAFE_DIST - nd)

        score = (
            m
            + RADIUS_WEIGHT * f["radius_std_norm"]
            + TIME_WEIGHT * (first_in_frame - i)
            + NEEDLE_WEIGHT * needle_penalty
        )

        if score < best_score:
            best_score = score
            best_start_frame = i

    best_score = float("inf")
    for i in range(last_in_frame + 1, len(features)):
        f = features[i]
        m = motions[i]

        if f is None or m is None:
            continue

        score = m + RADIUS_WEIGHT * f["radius_std_norm"]

        if score < best_score:
            best_score = score
            best_end_frame = i

    if best_start_frame is None:
        candidates_before = [
            i for i, state in enumerate(needle_states[:first_in_frame])
            if state in ["NEEDLE OUT", "BOUNDARY"]
        ]

        if candidates_before:
            if len(candidates_before) > 5:
                min_pos = 5
            else:
                min_pos = len(candidates_before)
            best_start_frame = candidates_before[-min_pos]
        else:
            best_start_frame = 0

    if best_end_frame is None:
        candidates_after = [
            i for i, state in enumerate(needle_states[last_in_frame + 1:], start=last_in_frame + 1)
            if state in ["NEEDLE OUT", "BOUNDARY"]
        ]

        if candidates_after:
            if len(candidates_after) > 5:
                max_pos = 4
            else:
                max_pos = len(candidates_after) - 1
            best_end_frame = candidates_after[max_pos]
        else:
            best_end_frame = len(features) - 1

    frame_before = get_frame_by_id(video_path, best_start_frame)
    frame_after = get_frame_by_id(video_path, best_end_frame)

    mask_before = segment_oocyte(frame_before, model_seg)
    mask_after = segment_oocyte(frame_after, model_seg)

    comparison_metrics = compare_oocyte_regions(
        frame_before, mask_before,
        frame_after, mask_after,
        inner_margin_px=inner_margin_px
    )

    if comparison_metrics is None:
        return {
            "status": "COMPARE_FAILED",
            "best_start_frame": best_start_frame,
            "best_end_frame": best_end_frame
        }

    result = {
        "status": "OK",
        "best_start_frame": best_start_frame,
        "best_end_frame": best_end_frame,
        "first_in_frame": first_in_frame,
        "last_in_frame": last_in_frame
    }
    result.update(comparison_metrics)

    return result


# ---------------- ПАКЕТНАЯ ОБРАБОТКА ВСЕХ ВИДЕО ----------------
video_paths = sorted(dataset_dir.glob("*.avi"), key=lambda p: int(p.stem))
rows = []

for video_path in video_paths:
    video_id = video_path.stem

    result_value = video_result_map.get(video_id)
    if result_value is None:
        try:
            result_value = video_result_map.get(str(int(float(video_id))))
        except Exception:
            result_value = None

    print(f"Обработка: {video_path.name}")

    try:
        metrics = process_video(video_path, inner_margin_px=inner_margin_px)

        row = {
            "video_id": video_id,
            "result": result_value,
            "mean_abs_diff": metrics.get("mean_abs_diff"),
            "max_abs_diff": metrics.get("max_abs_diff"),
            "changed_ratio": metrics.get("changed_ratio"),
            "mean_intensity_rel_delta": metrics.get("mean_intensity_rel_delta"),
            "std_intensity_rel_delta": metrics.get("std_intensity_rel_delta"),
            "entropy_rel_delta": metrics.get("entropy_rel_delta"),
            "lap_var_rel_delta": metrics.get("lap_var_rel_delta")
        }

    except Exception:
        row = {
            "video_id": video_id,
            "result": result_value,
            "mean_abs_diff": None,
            "max_abs_diff": None,
            "changed_ratio": None,
            "mean_intensity_rel_delta": None,
            "std_intensity_rel_delta": None,
            "entropy_rel_delta": None,
            "lap_var_rel_delta": None
        }

    rows.append(row)

fieldnames = [
    "video_id",
    "result",
    "mean_abs_diff",
    "max_abs_diff",
    "changed_ratio",
    "mean_intensity_rel_delta",
    "std_intensity_rel_delta",
    "entropy_rel_delta",
    "lap_var_rel_delta"
]

with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\nГотово.")
print(f"CSV сохранен: {output_csv}")