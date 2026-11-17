
from pathlib import Path
import json

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# ==========================================
# 1. CẤU HÌNH
# ==========================================

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset"
OUT = HERE.parent / "backend" / "models"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

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


# ==========================================
# 2. KIỂM TRA DỮ LIỆU
# ==========================================

def check_dataset():

    print("\n========== KIỂM TRA DỮ LIỆU ==========")

    for split in ("train", "val", "test"):

        split_path = DATA / split

        if not split_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục: {split_path}\n"
                "Hãy chạy python prepare_dataset.py trước."
            )

        for cls in CLASSES:

            class_path = split_path / cls

            if not class_path.exists():
                raise RuntimeError(
                    f"Thiếu thư mục lớp {cls} trong {split}."
                )

            images = [
                p for p in class_path.iterdir()
                if p.is_file()
                and p.suffix.lower() in {
                    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
                }
            ]

            if len(images) == 0:
                raise RuntimeError(
                    f"Lớp {cls} trong {split} đang trống."
                )

            print(f"{split}/{cls}: {len(images)} ảnh")

    print("\nDữ liệu đã được kiểm tra.")


# ==========================================
# 3. ĐỌC DỮ LIỆU
# ==========================================

def load_split(name, shuffle):

    return tf.keras.utils.image_dataset_from_directory(
        DATA / name,
        class_names=CLASSES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=shuffle,
        seed=SEED
    )


# ==========================================
# 4. XÂY DỰNG MÔ HÌNH MOBILENETV2
# ==========================================

def build_model():

    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    # Đóng băng các lớp MobileNetV2 ban đầu
    base.trainable = False

    inputs = layers.Input(shape=(*IMG_SIZE, 3))

    # Chuẩn hóa pixel từ 0-255 về -1 đến 1
    x = layers.Rescaling(
        1.0 / 127.5,
        offset=-1
    )(inputs)

    x = base(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.25)(x)

    # Đầu ra có 8 neuron tương ứng 8 lớp
    outputs = layers.Dense(
        len(CLASSES),
        activation="softmax"
    )(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==========================================
# 5. HUẤN LUYỆN MÔ HÌNH
# ==========================================

def main():

    print("======================================")
    print("      TEA LEAF AI - 8 LỚP")
    print("      MÔ HÌNH MOBILENETV2")
    print("======================================")

    # Kiểm tra dữ liệu trước khi train
    check_dataset()

    OUT.mkdir(parents=True, exist_ok=True)

    # Đọc dữ liệu
    train_ds = load_split(
        "train", True
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = load_split(
        "val", False
    ).prefetch(tf.data.AUTOTUNE)

    # Xây dựng mô hình
    model = build_model()

    model.summary()

    # Đường dẫn lưu mô hình
    best_path = OUT / "tea_leaf_mobilenetv2.keras"

    # Các callback hỗ trợ huấn luyện
    callbacks = [

        ModelCheckpoint(
            filepath=best_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Bắt đầu huấn luyện
    print("\nBẮT ĐẦU HUẤN LUYỆN MOBILENETV2...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Lưu danh sách lớp
    classes_path = OUT / "classes.json"

    classes_path.write_text(
        json.dumps(
            CLASSES,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n========== HOÀN TẤT ==========")

    print(f"Mô hình được lưu tại: {best_path}")

    print(f"Danh sách lớp được lưu tại: {classes_path}")

    print("\n8 lớp phân loại:")

    for i, cls in enumerate(CLASSES):
        print(f"{i + 1}. {cls}")

    print("\nTiếp theo chạy lệnh: python evaluate.py")


if __name__ == "__main__":
    main()
