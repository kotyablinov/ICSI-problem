"""
Обрезание изображений по контуру ооцита. Предварительный этап для обучения на изображениях.

Чтобы снизить влияние ошибки из-за объектов помимо ооцита изображение кадрировалось и обрезалось.
Для этого изображение:
1. сегментировалось
2. обрезалось 
3. в новый файл перезаписывался путь к обрезанному изображению
В итоге получался файл, где для каждого видео хранились пути до кадра до и после, 
кадрированный по ооциту
"""

import csv
from pathlib import Path

import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp


CSV_PATH = r"D:\diplom\dataset_before_after\before_after_pairs_clean.csv"
WEIGHTS_SEG = r"D:\diplom\weights\unetpp768_oocyte.pth"
OUTPUT_ROOT = r"D:\diplom\dataset_before_after\cropped_pairs"

TARGET_SIZE = 768
TARGET_CROP_SIZE = 512
MIN_OOCYTE_AREA = 500
SAVE_DEBUG = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_seg = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=3
).to(DEVICE)

model_seg.load_state_dict(torch.load(WEIGHTS_SEG, map_location=DEVICE))
model_seg.eval()


def segment_frame(frame):
    orig_h, orig_w = frame.shape[:2]

    img_resized = cv2.resize(frame, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(np.transpose(img_norm, (2, 0, 1))).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model_seg(img_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask


def get_reference_point(mask):
    needle_points = np.column_stack(np.where(mask == 2))  # y, x

    if len(needle_points) > 0:
        y_mean, x_mean = needle_points.mean(axis=0)
        return float(x_mean), float(y_mean)

    h, w = mask.shape[:2]
    return w / 2.0, h / 2.0


def select_main_oocyte(mask, min_area=500):
    oocyte_bin = (mask == 1).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        oocyte_bin,
        connectivity=8
    )

    if num_labels <= 1:
        return None

    ref_x, ref_y = get_reference_point(mask)
    candidates = []

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        cx, cy = centroids[label_id]
        dist2 = (cx - ref_x) ** 2 + (cy - ref_y) ** 2

        candidates.append({
            "label_id": label_id,
            "area": area,
            "dist2": dist2,
            "center": (float(cx), float(cy)),
            "bbox": (
                int(stats[label_id, cv2.CC_STAT_LEFT]),
                int(stats[label_id, cv2.CC_STAT_TOP]),
                int(stats[label_id, cv2.CC_STAT_LEFT] + stats[label_id, cv2.CC_STAT_WIDTH]),
                int(stats[label_id, cv2.CC_STAT_TOP] + stats[label_id, cv2.CC_STAT_HEIGHT]),
            )
        })

    large_candidates = [c for c in candidates if c["area"] >= min_area]
    pool = large_candidates if len(large_candidates) > 0 else candidates

    best = min(pool, key=lambda x: x["dist2"])
    return best


def crop_centered_square_or_keep(image, center_xy, target_crop_size=512):
    img_h, img_w = image.shape[:2]

    # если кадр уже маленький и квадратный — оставляем как есть
    if img_h <= target_crop_size and img_w <= target_crop_size and img_h == img_w:
        return image.copy(), (0, 0, img_w, img_h), img_w, "kept_original_square"

    # иначе режем квадрат максимально возможного размера, но не больше target_crop_size
    crop_size = min(target_crop_size, img_h, img_w)

    cx, cy = center_xy
    cx = float(np.clip(cx, 0, img_w - 1))
    cy = float(np.clip(cy, 0, img_h - 1))

    half = crop_size // 2

    left = int(round(cx)) - half
    top = int(round(cy)) - half

    left = max(0, min(left, img_w - crop_size))
    top = max(0, min(top, img_h - crop_size))

    right = left + crop_size
    bottom = top + crop_size

    cropped = image[top:bottom, left:right]

    crop_box = (left, top, right, bottom)

    if crop_size == target_crop_size:
        crop_mode = "cropped_512"
    else:
        crop_mode = f"cropped_max_square_{crop_size}"

    return cropped, crop_box, crop_size, crop_mode


def overlay_mask(image, mask, alpha=0.4):
    result = image.copy()

    colors = {
        1: (0, 255, 0),
        2: (0, 0, 255)
    }

    for class_id, color in colors.items():
        class_mask = (mask == class_id)
        if np.any(class_mask):
            colored = np.zeros_like(image, dtype=np.uint8)
            colored[class_mask] = color
            result[class_mask] = cv2.addWeighted(
                image[class_mask], 1 - alpha,
                colored[class_mask], alpha,
                0
            )

    return result


def process_image(image):
    mask = segment_frame(image)
    obj = select_main_oocyte(mask, MIN_OOCYTE_AREA)

    if obj is None:
        center_xy = (image.shape[1] / 2.0, image.shape[0] / 2.0)
        center_mode = "fallback_center"
    else:
        center_xy = obj["center"]
        center_mode = "oocyte_center"

    cropped, crop_box, crop_size, crop_mode = crop_centered_square_or_keep(
        image,
        center_xy,
        TARGET_CROP_SIZE
    )

    info = {
        "mask": mask,
        "obj": obj,
        "center_xy": center_xy,
        "center_mode": center_mode,
        "crop_box": crop_box,
        "crop_size": crop_size,
        "crop_mode": crop_mode
    }

    return cropped, info


output_root = Path(OUTPUT_ROOT)
before_out_dir = output_root / "before"
after_out_dir = output_root / "after"
debug_out_dir = output_root / "debug"
out_csv_path = output_root / "before_after_pairs_cropped.csv"

before_out_dir.mkdir(parents=True, exist_ok=True)
after_out_dir.mkdir(parents=True, exist_ok=True)

if SAVE_DEBUG:
    debug_out_dir.mkdir(parents=True, exist_ok=True)


rows_in = []

with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    original_fieldnames = list(reader.fieldnames) if reader.fieldnames is not None else []

    for row in reader:
        rows_in.append(row)

out_fieldnames = original_fieldnames.copy()
if "crop_status" not in out_fieldnames:
    out_fieldnames.append("crop_status")


rows_out = []
total = 0
errors = 0

for idx, row in enumerate(rows_in, start=1):
    total += 1

    before_path = row.get("before_path", "")
    after_path = row.get("after_path", "")

    before_img = cv2.imread(before_path)
    after_img = cv2.imread(after_path)

    if before_img is None:
        print(f"[{idx}] Ошибка чтения before: {before_path}")
        new_row = row.copy()
        new_row["crop_status"] = "read_error_before"
        rows_out.append(new_row)
        errors += 1
        continue

    if after_img is None:
        print(f"[{idx}] Ошибка чтения after: {after_path}")
        new_row = row.copy()
        new_row["crop_status"] = "read_error_after"
        rows_out.append(new_row)
        errors += 1
        continue

    try:
        cropped_before, before_info = process_image(before_img)
        cropped_after, after_info = process_image(after_img)

        before_name = Path(before_path).stem + "_crop.png"
        after_name = Path(after_path).stem + "_crop.png"

        save_before_path = before_out_dir / before_name
        save_after_path = after_out_dir / after_name

        ok_before = cv2.imwrite(str(save_before_path), cropped_before)
        ok_after = cv2.imwrite(str(save_after_path), cropped_after)

        if not ok_before:
            raise RuntimeError(f"Не удалось сохранить: {save_before_path}")

        if not ok_after:
            raise RuntimeError(f"Не удалось сохранить: {save_after_path}")

        new_row = row.copy()
        new_row["before_path"] = str(save_before_path)
        new_row["after_path"] = str(save_after_path)
        new_row["crop_status"] = (
            f"before_{before_info['center_mode']}_{before_info['crop_mode']};"
            f"after_{after_info['center_mode']}_{after_info['crop_mode']}"
        )
        rows_out.append(new_row)

        if SAVE_DEBUG:
            dbg_before = overlay_mask(before_img, before_info["mask"])
            dbg_after = overlay_mask(after_img, after_info["mask"])

            if before_info["obj"] is not None:
                x1, y1, x2, y2 = before_info["obj"]["bbox"]
                cx, cy = before_info["obj"]["center"]
                cv2.rectangle(dbg_before, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.circle(dbg_before, (int(round(cx)), int(round(cy))), 4, (255, 0, 255), -1)

            if after_info["obj"] is not None:
                x1, y1, x2, y2 = after_info["obj"]["bbox"]
                cx, cy = after_info["obj"]["center"]
                cv2.rectangle(dbg_after, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.circle(dbg_after, (int(round(cx)), int(round(cy))), 4, (255, 0, 255), -1)

            x1, y1, x2, y2 = before_info["crop_box"]
            cv2.rectangle(dbg_before, (x1, y1), (x2, y2), (255, 0, 0), 2)

            x1, y1, x2, y2 = after_info["crop_box"]
            cv2.rectangle(dbg_after, (x1, y1), (x2, y2), (255, 0, 0), 2)

            debug_before_path = debug_out_dir / (Path(before_path).stem + "_debug.png")
            debug_after_path = debug_out_dir / (Path(after_path).stem + "_debug.png")

            cv2.imwrite(str(debug_before_path), dbg_before)
            cv2.imwrite(str(debug_after_path), dbg_after)

        if idx % 50 == 0:
            print(f"Обработано: {idx}")

    except Exception as e:
        print(f"[{idx}] Ошибка обработки")
        print(f"before: {before_path}")
        print(f"after:  {after_path}")
        print(f"error:  {e}")

        new_row = row.copy()
        new_row["crop_status"] = "processing_error"
        rows_out.append(new_row)
        errors += 1


with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=out_fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)


print()
print("Готово")