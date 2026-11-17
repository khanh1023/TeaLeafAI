
from pathlib import Path
import json
import sys

import tensorflow as tf
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


def configure_utf8_console():
    """Ensure console output can display Vietnamese text on Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


configure_utf8_console()

# ==========================================
# 1. CẤU HÌNH
# ==========================================

HERE = Path(__file__).resolve().parent

DATA = HERE / "dataset"

MODEL_DIR = (
    HERE.parent
    / "backend"
    / "models"
)

MODEL_PATH = MODEL_DIR / "tea_leaf_mobilenetv2.keras"

CLASSES_PATH = MODEL_DIR / "classes.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

EXPECTED_CLASSES = [
    "algal_leaf",
    "anthracnose",
    "bird_eye_spot",
    "brown_blight",
    "gray_light",
    "healthy",
    "red_leaf_spot",
    "white_spot"
]


# ==========================================
# 2. KIỂM TRA VÀ ĐỌC DANH SÁCH LỚP
# ==========================================

def load_classes():

    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file classes.json: {CLASSES_PATH}"
        )

    with open(
        CLASSES_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        classes = json.load(f)

    if classes != EXPECTED_CLASSES:
        raise ValueError(
            "Danh sách lớp trong classes.json "
            "không khớp với danh sách lớp trong train.py."
        )

    print("\n========== DANH SÁCH 8 LỚP ==========")

    for i, cls in enumerate(classes):
        print(f"{i + 1}. {cls}")

    return classes


# ==========================================
# 3. KIỂM TRA DỮ LIỆU TEST
# ==========================================

def check_test_dataset(test_dir, classes):

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục test: {test_dir}"
        )

    print("\n========== KIỂM TRA DỮ LIỆU TEST ==========")

    for cls in classes:

        class_dir = test_dir / cls

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Thiếu thư mục lớp: {class_dir}"
            )

        image_count = sum(
            1 for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {
                ".jpg", ".jpeg", ".png",
                ".bmp", ".webp"
            }
        )

        if image_count == 0:
            raise ValueError(
                f"Không có ảnh trong lớp {cls}."
            )

        print(f"{cls}: {image_count} ảnh")


# ==========================================
# 4. ĐỌC DỮ LIỆU TEST
# ==========================================

def load_test_data(test_dir, classes):

    print("\nĐang đọc dữ liệu test...")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        class_names=classes,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
        seed=SEED
    )

    return test_ds


# ==========================================
# 5. TẢI MÔ HÌNH
# ==========================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình: {MODEL_PATH}"
        )

    print("\nĐang tải mô hình MobileNetV2...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Đã tải mô hình thành công.")

    return model


# ==========================================
# 6. DỰ ĐOÁN TRÊN TẬP TEST
# ==========================================

def predict_test(model, test_ds):

    print("\nĐang dự đoán trên tập test...")

    y_true = np.concatenate([
        labels.numpy()
        for images, labels in test_ds
    ])

    predictions = model.predict(
        test_ds,
        verbose=1
    )

    y_pred = np.argmax(
        predictions,
        axis=1
    )

    return y_true, y_pred


# ==========================================
# 7. TÍNH CÁC CHỈ SỐ ĐÁNH GIÁ
# ==========================================

def evaluate_metrics(y_true, y_pred, classes):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(classes)))
    )

    print("\n========== KẾT QUẢ ĐÁNH GIÁ ==========")

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nCLASSIFICATION REPORT:")
    print(report_text)

    return accuracy, report_dict, cm


# ==========================================
# 8. VẼ CONFUSION MATRIX
# ==========================================

def save_confusion_matrix(cm, classes):

    print("\nĐang tạo Confusion Matrix...")

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.5,
        cbar=True
    )

    plt.xlabel(
        "Nhãn dự đoán",
        fontsize=12
    )

    plt.ylabel(
        "Nhãn thực tế",
        fontsize=12
    )

    plt.title(
        "Confusion Matrix - Tea Leaf AI 8 Classes",
        fontsize=14
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.yticks(
        rotation=0
    )

    # Điều chỉnh bố cục biểu đồ
    plt.tight_layout()

    output_image = HERE / "confusion_matrix.png"

    # Lưu biểu đồ
    plt.savefig(
        output_image,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Đã lưu Confusion Matrix: {output_image}")

    return output_image


# ==========================================
# 9. LƯU KẾT QUẢ ĐÁNH GIÁ
# ==========================================

def save_evaluation(
    classes,
    accuracy,
    report_dict,
    cm
):

    result = {
        "model": "MobileNetV2",

        "num_classes": len(classes),

        "classes": classes,

        "accuracy": float(accuracy),

        "classification_report": report_dict,

        "confusion_matrix": cm.tolist()
    }

    output_json = HERE / "evaluation.json"

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Đã lưu kết quả JSON: {output_json}")

    return output_json


# ==========================================
# 10. CHƯƠNG TRÌNH CHÍNH
# ==========================================

def main():

    print("==========================================")
    print("       TEA LEAF AI - MODEL EVALUATION")
    print("       MOBILENETV2 - 8 CLASSES")
    print("==========================================")

    # Đọc danh sách lớp
    classes = load_classes()

    # Kiểm tra dữ liệu
    test_dir = DATA / "test"

    check_test_dataset(
        test_dir,
        classes
    )

    # Đọc dữ liệu test
    test_ds = load_test_data(
        test_dir,
        classes
    )

    # Tải mô hình
    model = load_model()

    # Kiểm tra số đầu ra
    if model.output_shape[-1] != len(classes):
        raise ValueError(
            "Số đầu ra mô hình không khớp số lớp."
        )

    # Dự đoán
    y_true, y_pred = predict_test(
        model,
        test_ds
    )

    # Tính chỉ số đánh giá
    accuracy, report_dict, cm = evaluate_metrics(
        y_true,
        y_pred,
        classes
    )

    # Lưu Confusion Matrix
    save_confusion_matrix(
        cm,
        classes
    )

    # Lưu kết quả JSON
    save_evaluation(
        classes,
        accuracy,
        report_dict,
        cm
    )

    print("\n==========================================")
    print("       ĐÁNH GIÁ HOÀN TẤT!")
    print("==========================================")

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nCác file kết quả:")
    print("- confusion_matrix.png")
    print("- evaluation.json")


# ==========================================
# 11. CHẠY CHƯƠNG TRÌNH
# ==========================================

if __name__ == "__main__":
    main()
