# 🫁 Pneumonia Detection from Chest X-Rays

Binary image classification (NORMAL vs PNEUMONIA) using a CNN built from scratch with TensorFlow/Keras.

## Results
| Metric | Score |
|---|---|
| Test Accuracy | 91% |
| Pneumonia Recall | 98% |
| ROC-AUC | 0.9574 |

Only **8 out of 390** pneumonia cases were missed on the test set.

## Dataset
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — 5,216 pediatric chest X-rays (Kaggle).

## Pipeline
1. EDA & class distribution analysis
2. 80/20 train/validation split (official val folder ignored — only 16 images)
3. Data augmentation (rotation, zoom, translation)
4. Class weights to handle imbalance (~74% pneumonia)
5. CNN from scratch: 4 Conv blocks + BatchNorm + GlobalAveragePooling
6. Training with EarlyStopping + ReduceLROnPlateau callbacks
7. Evaluation: classification report, ROC-AUC, confusion matrix

## Tech Stack
- Python, TensorFlow/Keras, scikit-learn
- NumPy, Matplotlib, Plotly

## Limitation
This model was trained on **pediatric** X-rays (ages 1–5). It will not generalize to adult chest X-rays.

## Run it
Open in Google Colab and add your Kaggle API credentials to download the dataset.

## 👨‍💻 Author

**Alaa Mekawi**
