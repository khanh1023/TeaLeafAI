
from pathlib import Path
import random
import shutil
import json
import re

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "DuLieuGoc"
DEST = Path(__file__).resolve().parent / "dataset"

# Tỷ lệ chia dữ liệu: 70% train, 15% val, 15% test
RATIOS = (0.70, 0.15, 0.15)

SEED = 42

EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# ==========================================
# 2. KHAI BÁO 8 LỚP
# ==========================================

CLASSES = [
    "algal_leaf",
    "anthracnose",
    "bird_eye_spot",
    "brown_blight",
    "gray_light",
    "healthy",
    "red_leaf_spot",
    "white_spot"
]

# Tên thư mục nguồn -> tên lớp chuẩn
CLASS_MAP = {
    "algal leaf": "algal_leaf",
    "algal_leaf": "algal_leaf",

    "anthracnose": "anthracnose",

    "bird eye spot": "bird_eye_spot",
    "bird_eye_spot": "bird_eye_spot",

    "brown blight": "brown_blight",
    "brown_blight": "brown_blight",

    "gray light": "gray_light",
    "gray_light": "gray_light",
    "grey light": "gray_light",
    "grey_light": "gray_light",

    "healthy": "healthy",

    "red leaf spot": "red_leaf_spot",
    "red_leaf_spot": "red_leaf_spot",

    "white spot": "white_spot",
    "white_spot": "white_spot"
}


# ==========================================
# 3. CHUẨN HÓA TÊN THƯ MỤC
# ==========================================

def normalize_name(name):
    name = name.strip().lower()
    name = name.replace("-", " ")
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    return name


# ==========================================
# 4. THU THẬP ẢNH TỪ 8 THƯ MỤC
# ==========================================

def collect_images():

    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu gốc: {SOURCE}"
        )

    collected = {
        cls: [] for cls in CLASSES
    }

    for folder in SOURCE.iterdir():

        if not folder.is_dir():
            continue

        key = normalize_name(folder.name)

        class_name = CLASS_MAP.get(key)

        if class_name is None:
            print(
                f"Bỏ qua thư mục không thuộc 8 lớp: {folder.name}"
            )
            continue

        images = [
            p for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in EXTENSIONS
        ]

        collected[class_name].extend(images)

    return collected


# ==========================================
# 5. HIỂN THỊ SỐ LƯỢNG ẢNH
# ==========================================

def show_image_summary(collected):

    print("\n========== SỐ ẢNH TÌM THẤY ==========")

    for cls in CLASSES:
        print(f"{cls}: {len(collected[cls])}")

    print("=====================================")

    missing = [
        cls for cls in CLASSES
        if len(collected[cls]) == 0
    ]

    if missing:
        print("\nCẢNH BÁO: Các lớp chưa có ảnh:")

        for cls in missing:
            print("-", cls)

        raise RuntimeError(
            "Bạn cần bổ sung ảnh cho đủ 8 lớp trước khi chia dữ liệu."
        )

    too_few = [
        cls for cls in CLASSES
        if len(collected[cls]) < 3
    ]

    if too_few:
        raise RuntimeError(
            "Mỗi lớp cần ít nhất 3 ảnh để chia thành train, val và test."
        )


# ==========================================
# 6. CHIA DỮ LIỆU 70/15/15
# ==========================================

def split_dataset(collected):

    answer = input(
        "\nGõ YES để tạo lại training/dataset "
        "(dữ liệu dataset cũ sẽ bị xóa): "
    )

    if answer.strip().upper() != "YES":
        print("Đã hủy chia dữ liệu.")
        return

    # Chỉ xóa thư mục dataset đã tạo trước đó
    if DEST.exists():
        shutil.rmtree(DEST)

    rng = random.Random(SEED)

    # Tạo thư mục train, val, test cho 8 lớp
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            (DEST / split / cls).mkdir(
                parents=True,
                exist_ok=True
            )

    summary = {}

    for cls, paths in collected.items():

        paths = list(paths)
        rng.shuffle(paths)

        n = len(paths)

        n_train = int(n * RATIOS[0])
        n_val = int(n * RATIOS[1])

        chunks = {
            "train": paths[:n_train],
            "val": paths[n_train:n_train + n_val],
            "test": paths[n_train + n_val:]
        }

        summary[cls] = {
            "total": n,
            "train": len(chunks["train"]),
            "val": len(chunks["val"]),
            "test": len(chunks["test"])
        }

        # Sao chép ảnh vào từng tập
        for split, chunk in chunks.items():

            target = DEST / split / cls

            for i, src in enumerate(chunk):

                out = target / src.name

                # Tránh trùng tên ảnh
                if out.exists():
                    out = target / f"{src.stem}_{i}{src.suffix.lower()}"

                shutil.copy2(src, out)

    # Lưu thông tin chia dữ liệu
    summary_path = DEST / "split_summary.json"

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    # Lưu danh sách 8 lớp
    classes_path = DEST / "classes.json"

    classes_path.write_text(
        json.dumps(
            CLASSES,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n========== HOÀN TẤT ==========")
    print("Đã chia dữ liệu theo tỷ lệ 70/15/15.")

    print("\nChi tiết:")
    print(json.dumps(
        summary,
        ensure_ascii=False,
        indent=2
    ))

    print(f"\nThư mục kết quả: {DEST}")

    print("\nLưu ý:")
    print("- Ảnh trong DuLieuGoc không bị xóa.")
    print("- Chỉ thư mục training/dataset được tạo lại.")
    print("- Cần huấn luyện lại mô hình sau khi chia dữ liệu.")


# ==========================================
# 7. CHƯƠNG TRÌNH CHÍNH
# ==========================================

def main():

    print("BẮT ĐẦU CHUẨN BỊ DỮ LIỆU 8 LỚP")

    collected = collect_images()

    show_image_summary(collected)

    split_dataset(collected)


if __name__ == "__main__":
    main()
