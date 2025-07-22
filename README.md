# 🍷 Wine Quality Prediction

This project is a machine learning application that predicts the quality of red and white wines based on physicochemical features using classification models.

---

## 📌 Project Goals

- Analyze wine datasets and explore feature relationships.
- Build multiple machine learning models to predict wine quality.
- Evaluate and compare model performance.
- Save trained models for future predictions.

---

## 📂 Dataset Information

The dataset contains physicochemical and quality information about wine samples.

- **Red Wine Dataset**: `winequality-red.csv`  
- **White Wine Dataset**: `winequality-white.csv`  
- **Source**: UCI Machine Learning Repository  
  ➡️ https://archive.ics.uci.edu/ml/datasets/Wine+Quality

Each row in the dataset represents a wine sample with the following features:

- `fixed acidity`
- `volatile acidity`
- `citric acid`
- `residual sugar`
- `chlorides`
- `free sulfur dioxide`
- `total sulfur dioxide`
- `density`
- `pH`
- `sulphates`
- `alcohol`
- `quality` (target variable)

---

## 🧠 Machine Learning Models

Trained models are saved under the `models/` directory.

| Model                       | File Name                      |
|----------------------------|---------------------------------|
| Logistic Regression         | `logistic_regression_wine.pkl` |
| Decision Tree               | `decision_tree_wine.pkl`       |
| Random Forest               | `random_forest_wine.pkl`       |
| Random Forest + SMOTE       | `rf_smote_model.pkl`           |

---

## 🛠️ Installation & Requirements

Install required libraries using:

```bash
pip install -r requirements.txt


Project Structure:

Wine_Quality/
│
├── models/                     # Trained ML models (.pkl files)
│   ├── decision_tree_wine.pkl
│   ├── logistic_regression_wine.pkl
│   ├── random_forest_wine.pkl
│   └── rf_smote_model.pkl
│
├── wine_quality.py            # Main Python script
├── winequality-red.csv        # Red wine dataset
├── winequality-white.csv      # White wine dataset
├── winequality.names          # Feature descriptions
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation



📈 Model Evaluation
The models are evaluated using common classification metrics:

Accuracy

Precision

Recall

F1 Score

ROC-AUC

You can further visualize the performance using ROC curves or confusion matrices.

👩‍💻 Author
Melek Ikiz
🔗 GitHub: @Melekikiz
📧 Email: mellikiz.33@gmail.com
