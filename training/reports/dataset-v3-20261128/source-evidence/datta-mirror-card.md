---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype:
      class_label:
        names:
          '0': algal_spot
          '1': brown_blight
          '2': gray_blight
          '3': healthy
          '4': helopeltis
          '5': red_spot
  splits:
  - name: train
    num_bytes: 30321493
    num_examples: 5867
  download_size: 31879641
  dataset_size: 30321493
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
license: cc-by-nc-4.0
task_categories:
- image-classification
size_categories:
- 1K<n<10K
---

# Tea Leaf Disease Classification

A dataset for disease classification of tea leaves. The dataset contains 5,867 images across 6 classes: algal_spot, brown_blight, gray_blight, healthy, helopeltis, red_spot.  
Images per class:
- algal_spot: 1,000
- brown_blight: 867
- gray_blight: 1,000
- healthy: 1,000
- helopeltis: 1,000
- red_spot: 1,000

This dataset is indexed on https://project-agml.github.io/ as part of the AgML python library.

## Citation

```bibtex
@article{BALASUNDARAM2025103784,
title = {Tea leaf disease detection using segment anything model and deep convolutional neural networks},
journal = {Results in Engineering},
volume = {25},
pages = {103784},
year = {2025},
issn = {2590-1230},
doi = {https://doi.org/10.1016/j.rineng.2024.103784},
url = {https://www.sciencedirect.com/science/article/pii/S2590123024020279},
author = {Ananthakrishnan Balasundaram and Prem Sundaresan and Aryan Bhavsar and Mishti Mattu and Muthu Subash Kavitha and Ayesha Shaik}
}
```

https://www.kaggle.com/datasets/saikatdatta1994/tea-leaf-disease