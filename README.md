# TeaLeafAI dataset v2 — archive parts

6,840 split images: 4,788 train / 1,026 validation / 1,026 test. Four additional healthy reserve images are excluded from training/evaluation.

Run `python training/download_dataset_v2.py` from the main code checkout. The script downloads these parts, verifies each SHA-256, assembles the ZIP and verifies every image.

These files are byte ranges of one ZIP, not independent ZIP archives. Join part01, part02 and part03 in that order.

Full archive SHA-256: `acc711fbc56ca597207fa76047d520f9f387efbf3a42b527bc15e8348fbccfd0`.

Public additions retain CC BY 4.0 with author/DOI attribution; user-provided originals retain their original terms. Images were EXIF-corrected, converted to RGB JPEG and bounded to 512 px. Sources, changes and attribution are included in the archive README and at https://github.com/khanh1023/TeaLeafAI/blob/main/docs/DATASET_V2.md .

This tag is data only; existing website models have not been retrained on this version.

TeaLeafAI-dataset-v2-20261125.zip.part01 — 90000000 bytes — SHA-256 0d92acc541c0ee97d4b0f22b1316aec110e65b5d41360a6713ff2710f78742ae
TeaLeafAI-dataset-v2-20261125.zip.part02 — 90000000 bytes — SHA-256 d92a8abaffd0695566c870ed75b78490db30e88bad3a6089b580f10086ed2c90
TeaLeafAI-dataset-v2-20261125.zip.part03 — 75012357 bytes — SHA-256 dee71c3de2ed575c9df55049ebc795d443b250fb107b407f379113d84bcd0a96
