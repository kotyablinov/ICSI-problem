"""
Формирование пар кадров до и после введения иглы в ооцит.

Скрипт обрабатывает видео, определяет положение иглы относительно
ооцита с помощью моделей классификации и сегментации, выбирает наиболее
стабильные кадры до введения и после извлечения иглы, сохраняет изображения
в каталоги before/after и формирует таблицу before_after_pairs.csv.

Результат оплодотворения для каждого видео загружается из videos_info.csv.
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

output_dir = Path(r"D:\diplom\dataset_before_after")
before_dir = output_dir / "before"
after_dir = output_dir / "after"
output_csv = output_dir / "before_after_pairs.csv"

RADIUS_WEIGHT = 400
TIME_WEIGHT = 0.3
NEEDLE_WEIGHT = 3
NEEDLE_SAFE_DIST = 25

EPS = 1e-6

before_dir.mkdir(parents=True, exist_ok=True)
after_dir.mkdir(parents=True, exist_ok=True)

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


def save_frame(frame, path):
    ok = cv2.imwrite(str(path), frame)
    if not ok:
        raise RuntimeError(f"Не удалось сохранить изображение: {path}")


# ---------------- ОБРАБОТКА ОДНОГО ВИДЕО ----------------

def process_video(video_path):
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

    return {
        "status": "OK",
        "best_start_frame": best_start_frame,
        "best_end_frame": best_end_frame,
        "first_in_frame": first_in_frame,
        "last_in_frame": last_in_frame
    }


def save_before_after_frames(video_path, start_frame, end_frame):
    frame_before = get_frame_by_id(video_path, start_frame)
    frame_after = get_frame_by_id(video_path, end_frame)

    before_name = f"{video_path.stem}_frame_{start_frame:06d}_before.png"
    after_name = f"{video_path.stem}_frame_{end_frame:06d}_after.png"

    before_path = before_dir / before_name
    after_path = after_dir / after_name

    save_frame(frame_before, before_path)
    save_frame(frame_after, after_path)

    return before_path, after_path


# ---------------- ПОЛУЧЕНИЕ СПИСКА ВИДЕО ----------------

def sort_key(path_obj):
    stem = path_obj.stem
    try:
        return (0, int(stem))
    except Exception:
        return (1, stem.lower())


video_paths = sorted(dataset_dir.glob("*.avi"), key=sort_key)

# ---------------- ПАКЕТНАЯ ОБРАБОТКА ВСЕХ ВИДЕО ----------------

rows = []

for video_path in video_paths:
    video_id = video_path.stem
    video_file = video_path.name

    result_value = video_result_map.get(video_id)
    if result_value is None:
        try:
            result_value = video_result_map.get(str(int(float(video_id))))
        except Exception:
            result_value = None

    print(f"Обработка: {video_file}")

    before_path_str = ""
    after_path_str = ""

    try:
        info = process_video(video_path)

        if info["status"] == "OK":
            before_path, after_path = save_before_after_frames(
                video_path,
                info["best_start_frame"],
                info["best_end_frame"]
            )
            before_path_str = str(before_path)
            after_path_str = str(after_path)
            print(
                f"  Сохранено: before={info['best_start_frame']}, "
                f"after={info['best_end_frame']}"
            )
        else:
            print(f"  Пропуск: {info['status']}")

    except Exception as e:
        print(f"  Ошибка: {e}")

    row = {
        "video_file": video_file,
        "before_path": before_path_str,
        "after_path": after_path_str,
        "result": result_value if result_value is not None else ""
    }

    rows.append(row)

fieldnames = [
    "video_file",
    "before_path",
    "after_path",
    "result"
]

with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\nГотово.")
print(f"Кадры сохранены в: {output_dir}")
print(f"CSV сохранен: {output_csv}")