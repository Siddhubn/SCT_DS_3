import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree

data_path = './bank-full.csv'
data = pd.read_csv(data_path, delimiter=';')

print("Dataset Info:")
print(data.info())
print("\nFirst 5 Rows:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Data Analysis
print("\nSummary Statistics:")
print(data.describe())

# Plotting the distribution of the target variable
y_dist = data['y'].value_counts()
sns.barplot(x=y_dist.index, y=y_dist.values)
plt.title('Distribution of Target Variable')
plt.xlabel('Target Variable (y)')
plt.ylabel('Count')
plt.show()

# Correlation Heatmap (numeric features only)
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = data[numeric_columns].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap (Numeric Features Only)')
plt.show()

# Pairplot for selected numerical features
numerical_features = ['age', 'balance', 'duration', 'campaign']
sns.pairplot(data[numerical_features])
plt.show()

# Boxplot to analyze categorical features with respect to the target
categorical_features = ['job', 'marital', 'education']
for feature in categorical_features:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=data[feature], y=data['age'], hue=data['y'])
    plt.title(f'{feature} vs Age by Target')
    plt.xlabel(feature)
    plt.ylabel('Age')
    plt.legend(title='Target')
    plt.xticks(rotation=45)
    plt.show()

label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data.drop(columns=['y'])
y = data['y']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(20, 10))
tree.plot_tree(classifier, feature_names=data.columns[:-1], class_names=label_encoders['y'].classes_, filled=True, fontsize=10)
plt.show()

import joblib
joblib.dump(classifier, 'decision_tree_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
