import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

df=pd.read_csv("winequality-red.csv", sep=";")
print(df.head())
print(df.info())
print(df.describe())

#Missing Values
print("Missing values per column:")
print(df.isnull().sum())

#Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatap")
plt.show()

#Binary quality label: 1 if quality>=7 (good) else 0 (bad)
df['quality_label']= (df['quality']>=7).astype(int)
print("Value counts for quality Label:")
print(df["quality_label"].value_counts(normalize=True))

#Define features and target variable
X=df.drop(['quality', 'quality_label'], axis=1)
y=df["quality_label"]

#Split data into train and test sets (%80 train, %20 test)
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

#Standardize features
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#Train logistic regression model
lr_model=LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

#Predict on test set
y_pred_lr=lr_model.predict(X_test_scaled)

#Evuluate logistig regression performance
print("Logistig Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test,y_pred_lr))

#Feature importance from logistic regression coefficients
importance=pd.DataFrame({
    'Feature':X.columns,
    'Coefficient':lr_model.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

#Plot feature importance for logistic regression
plt.figure(figsize=(8,5))
sns.barplot(x='Coefficient', y='Feature', data=importance, palette='coolwarm')
plt.title("Feature Importance (Logistic Regression)")
plt.show()

#Train decision tree model
dt_model=DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)
y_pred_dt=dt_model.predict(X_test_scaled)

#Train random forest model
rf_model=RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf=rf_model.predict(X_test_scaled)


#SMOTE
smote=SMOTE(random_state=42)

X_resampled, y_resampled=smote.fit_resample(X_train, y_train)

smote_scaler=StandardScaler()
X_resampled_scaled=smote_scaler.fit_transform(X_resampled)
X_test_rescaled=smote_scaler.transform(X_test)

rf_smote_model=RandomForestClassifier(random_state=42)
rf_smote_model.fit(X_resampled_scaled, y_resampled)
y_pred_smote=rf_smote_model.predict(X_test_rescaled)

#Calculate ROC AUC scored for all models
lr_probs=lr_model.predict_proba(X_test_scaled)[:, 1]
dt_probs=dt_model.predict_proba(X_test_scaled)[:, 1]
rf_probs=rf_model.predict_proba(X_test_scaled)[:, 1]
smote_probs = rf_smote_model.predict_proba(X_test_scaled)[:, 1]

lr_auc=roc_auc_score(y_test, lr_probs)
dt_auc=roc_auc_score(y_test, dt_probs)
rf_auc=roc_auc_score(y_test, rf_probs)
smote_auc = roc_auc_score(y_test, smote_probs)


#Compute ROC curves
lr_fpr, lr_tpr, _=roc_curve(y_test, lr_probs)
dt_fpr, dt_tpr, _=roc_curve(y_test, dt_probs)
rf_fpr, rf_tpr, _=roc_curve(y_test, rf_probs)
smote_fpr, smote_tpr, _ = roc_curve(y_test, smote_probs)

#Plot ROC curves for all models

plt.figure(figsize=(8,6))
plt.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC={lr_auc:.2f})")
plt.plot(dt_fpr, dt_tpr, label=f"Decision Tree (AUC= {dt_auc:.2f})")
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC ={rf_auc:.2f})")
plt.plot(smote_fpr, smote_tpr, label=f"RF + SMOTE (AUC = {smote_auc:.2f})")
plt.plot([0,1],[0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()



#GridSearchCV ile RF Optimizasyonu
param_grid={
    'n_estimators':[50,100],
    'max_depth':[10,20],
    'min_samples_split':[2,5],
    'min_samples_leaf':[1,2]

}

grid_search=GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    n_jobs=1,
    verbose=2
)
grid_search.fit(X_train_scaled,y_train)
print("Best parameters found:", grid_search.best_params_)
print("Best csoss_validation score:", grid_search.best_score_)

best_rf=grid_search.best_estimator_
y_pred_best_rf=best_rf.predict(X_test_scaled)

print("Optimized Random Forest Accuracy:", accuracy_score(y_test, y_pred_best_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_best_rf))

#Perform
results=pd.DataFrame({
    "Model":["Logistic Regression", "Decision Tree", "Random Forest", "RF + SMOTE"],
    "Accuracy":[
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test,y_pred_rf),
        accuracy_score(y_test, y_pred_smote)
    ],
    "AUC":[
        lr_auc,
        dt_auc,
        rf_auc,
        smote_auc
    ]
})

print("\nModel Performance Comparison:")
print(results.sort_values(by="Accuracy", ascending=False))



#save model with joblib and os
os.makedirs("models", exist_ok=True)
joblib.dump(lr_model, "models/logistic_regression_wine.pkl")
joblib.dump(dt_model, "models/decision_tree_wine.pkl")
joblib.dump(rf_model,"models/random_forest_wine.pkl")
joblib.dump(rf_smote_model,"models/rf_smote_model.pkl")
print("Model saved successfully.")

loaded_model=joblib.load("models/rf_smote_model.pkl")
print("Loaded model test set accuracy:", accuracy_score(y_test, loaded_model.predict(X_test_rescaled)))
