import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load data
file_path = '/workspaces/Classifier-for-Process-Yield-Prediction/signal-data.csv'  # adjust if needed
data = pd.read_csv(file_path)

print("First 5 rows:\n", data.head())
print("\nData description:\n", data.describe())
print("\nData info:\n")
print(data.info())

# Handle missing values
print("\nMissing values before filling:\n", data.isnull().sum())
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = data.select_dtypes(include=['object', 'category']).columns

for col in numerical_cols:
    data[col].fillna(data[col].mean(), inplace=True)

for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

print("\nMissing values after filling:\n", data.isnull().sum())

# Drop time if exists
if 'Time' in data.columns:
    data.drop(['Time'], axis=1, inplace=True)

print("\nData after cleaning:\n", data.head())

# ========== Visualization ==========
unique_vals = data['Pass/Fail'].unique()
targets = [data[data['Pass/Fail'] == val] for val in unique_vals]

# Histograms for sensor measurements
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for i, ax in enumerate(axes.flat):
    for target in targets:
        sns.histplot(target[str(i+1)], kde=True, ax=ax)
    ax.set_title(f'Sensor {i+1} Measurements by Pass/Fail')
fig.legend(labels=['Pass', 'Fail'])
plt.tight_layout()
plt.savefig('sensor_histograms.png')
plt.close()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(data.select_dtypes(include=['number']).corr(), annot=True, cmap="YlGnBu")
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

# Distribution plot
plt.figure()
plt.hist(data['Pass/Fail'])
plt.xlabel('Pass/Fail')
plt.ylabel('Frequency')
plt.title('Distribution of Pass/Fail')
plt.savefig('pass_fail_distribution_hist.png')
plt.close()

# Bar chart
plt.figure()
data['Pass/Fail'].value_counts().plot(kind='bar')
plt.xlabel('Pass/Fail')
plt.ylabel('Count')
plt.title('Count of Pass/Fail')
plt.savefig('pass_fail_bar_chart.png')
plt.close()

# Box plot
plt.figure()
sns.boxplot(x='Pass/Fail', y='1', data=data)
plt.title('Boxplot of Sensor 1 by Pass/Fail')
plt.savefig('boxplot_sensor1.png')
plt.close()

# Violin plot
plt.figure()
sns.violinplot(x='Pass/Fail', y='1', data=data)
plt.title('Violin plot of Sensor 1 by Pass/Fail')
plt.savefig('violinplot_sensor1.png')
plt.close()

# ========== Modeling ==========
X = data.drop('Pass/Fail', axis=1)
y = data['Pass/Fail']

# Handle imbalance with SMOTE
print("\nClass counts before SMOTE:\n", y.value_counts())
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print("\nClass counts after SMOTE:\n", y_res.value_counts())

# Split & scale
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ========== Random Forest ==========
rf = RandomForestClassifier(random_state=42)
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
rf_grid = GridSearchCV(rf, rf_params, cv=5)
rf_grid.fit(X_train, y_train)
print("\nBest Random Forest params:", rf_grid.best_params_)
print(classification_report(y_test, rf_grid.predict(X_test)))

# ========== Support Vector Machine ==========
svm = SVC(random_state=42)
svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svm_grid = GridSearchCV(svm, svm_params, cv=5)
svm_grid.fit(X_train, y_train)
print("\nBest SVM params:", svm_grid.best_params_)
print(classification_report(y_test, svm_grid.predict(X_test)))

# ========== Naive Bayes ==========
nb = GaussianNB()
nb.fit(X_train, y_train)
print("\nNaive Bayes Classification Report:\n", classification_report(y_test, nb.predict(X_test)))

# ========== Accuracy comparison ==========
models = ['Random Forest', 'SVM', 'Naive Bayes']
accuracy = [rf_grid.score(X_test, y_test), svm_grid.score(X_test, y_test), nb.score(X_test, y_test)]
for model, acc in zip(models, accuracy):
    print(f"{model}: Accuracy = {acc:.4f}")

# ========== Save best model ==========
best_model = rf_grid
joblib.dump(best_model, 'best_model.pkl')
print("\nBest model saved as 'best_model.pkl'.")
