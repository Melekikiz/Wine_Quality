# Wine Quality Prediction

This project focuses on predicting the quality of red wines using machine learning models. The dataset used is the Wine Quality dataset from the UCI Machine Learning Repository.

---

## Project Overview

- Data preprocessing, feature engineering, and classification were performed on the dataset.
- Models trained include Logistic Regression, Decision Tree, Random Forest, and Random Forest with SMOTE for handling class imbalance.
- Model performances were evaluated using Accuracy, Confusion Matrix, Classification Report, and ROC-AUC metrics.
- The best performance was achieved with the Random Forest and its optimized version.
- Trained models are saved in `.pkl` format for future use.

---

## File Contents

- `wine_quality.py`: Main Python script for training and evaluating models.
- `winequality-red.csv`: Red wine dataset.
- `models/`: Folder containing the saved model files.
- `requirements.txt`: Project dependencies.

---

## Installation and Usage

1. Clone or download the repository:

```bash
git clone https://github.com/Melekikiz/Wine_Quality.git
cd Wine_Quality



Performance Summary:
Model	Accuracy	AUC
Logistic Regression	0.89	0.88
Decision Tree	0.90	0.81
Random Forest	0.94	0.95
Random Forest + SMOTE	0.91	0.89



Contact
For any questions or feedback:

Melek Ikiz
Email: mellikiz.33@gmail.com
GitHub: https://github.com/Melekikiz


This project is developed for educational purposes.
