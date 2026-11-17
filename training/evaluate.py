
from pathlib import Path
import json

import tensorflow as tf
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. CẤU HÌNH
# ==========================================

HERE = Path(__file__).resolve().parent

DATA = HERE / "dataset"
MODEL_PATH = (
    HERE.parent
    / "backend"
    / "models"
    / "tea_leaf_mobilenetv2.keras"
)

CLASSES_PATH = (
    HERE.parent
    / "backend"
    / "models"
    / "classes.json"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# ==========================================
# 2. ĐỌC DANH SÁCH 8 LỚP
# ==========================================

def load_classes():

    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {CLASSES_PATH}"
        )

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    print("\nDanh sách lớp:")

    for i, cls in enumerate(classes):
        print(f"{i + 1}. {cls}")

    return classes


# ==========================================
# 3. ĐÁNH GIÁ MÔ HÌNH
# ==========================================

def main():

    print("====================================")
    print("     ĐÁNH GIÁ MÔ HÌNH TEA LEAF AI")
    print("====================================")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình: {MODEL_PATH}"
        )

    test_dir = DATA / "test"

    if not test_dir.exists():
        raise FileNotFoundError(
            "Không tìm thấy tập dữ liệu test. "
            "Hãy chạy prepare_dataset.py trước."
        )

    classes = load_classes()

    # Đọc dữ liệu test theo đúng thứ tự classes.json
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        class_names=classes,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    # Tải mô hình đã huấn luyện
    print("\nĐang tải mô hình...")

    model = tf.keras.models.load_model(MODEL_PATH)

    # Dự đoán
    print("\nĐang đánh giá trên tập test...")

    y_true = np.concatenate([
        labels.numpy()
        for images, labels in test_ds
    ])

    predictions = model.predict(test_ds)

    y_pred = np.argmax(predictions, axis=1)

    # Tính độ chính xác
    accuracy = accuracy_score(y_true, y_pred)

    print("\n========== KẾT QUẢ ==========")

    print(f"Độ chính xác Accuracy: {accuracy * 100:.2f}%")

    # Báo cáo theo từng lớp
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0
    )

    print("\nBÁO CÁO PHÂN LOẠI:")
    print(report)

    # Ma trận nhầm lẫn
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(classes)))
    )

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=classes,
        yticklabels=classes
    )

    plt.xlabel("Nhãn dự đoán")
    plt.ylabel("Nhãn thực tế")
    plt.title("Confusion Matrix - Tea Leaf AI 8 Classes")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    # Lưu ảnh ma trận nhầm lẫn
    output_image = HERE / "confusion_matrix.png"

    plt.savefig(output_image, dpi=300)

    plt.show()

    # Lưu kết quả đánh giá
    result = {
        "classes": classes,
        "accuracy": float(accuracy),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(classes))),
            target_names=classes,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": cm.tolist()
    }

    output_json = HERE / "evaluation.json"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n========== ĐÃ HOÀN TẤT ==========")
    print(f"Ma trận nhầm lẫn: {output_image}")
    print(f"Kết quả đánh giá: {output_json}")


if __name__ == "__main__":
    main()
