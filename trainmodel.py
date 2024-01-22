from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import gc


# Load data
chunk = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', chunksize=1000000, index_col=0)
df = pd.concat(chunk)

# Drop rows with NaN values
df.dropna(inplace=True)

# Assuming 'Label' is the target column
Y = df[' Label']
X = df.drop(" Label", axis=1)

# Handle non-numeric columns
non_numeric_cols = X.select_dtypes(exclude=np.number).columns.tolist()

label_encoders = {}
for col in non_numeric_cols:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col])

# Replace infinite values with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale your data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=50, max_features="log2", random_state=0, n_jobs=-1)

# Fit the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print(df[' Label'].unique())


# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred, pos_label='DDoS')

cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='DDoS')

recall = recall_score(y_test, y_pred, pos_label='DDoS')

# Print metrics
print("Accuracy:", accuracy)
print("F1 Score:", f1score)
print("Confusion Matrix:\n", cm)
print("Precision:", precision)
print("Recall:", recall)

misclassified_samples = X_test[y_test != y_pred]
mc=misclassified_samples.shape[0]
print("Misclassified :",mc)

metrics_dict = {
    "RF(n_estimators=50, max_features='log2')": {
        "accuracy": accuracy,
        "f1_score": f1score,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm
    }
}

# Assuming you have initialized your empty dictionaries
f1_dict = {}
precision_dict = {}
recall_dict = {}
accuracy_dict = {}

# Then update them with your metrics
f1_dict["RF(n_estimators=50, max_features='log2')"] = f1score
precision_dict["RF(n_estimators=50, max_features='log2')"] = precision
recall_dict["RF(n_estimators=50, max_features='log2')"] = recall
accuracy_dict["RF(n_estimators=50, max_features='log2')"] = accuracy


print(classification_report(y_test, y_pred, digits=5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "DDoS"], yticklabels=["Benign", "DDoS"])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('RF Confusion Matrix (max_features=log2)')
plt.show()

keys = f1_dict, precision_dict, recall_dict, accuracy_dict
metrics = ['F1_Score', 'Precision', 'Recall', 'Accuracy']
data = pd.DataFrame(keys)
data.index = metrics
print(data)

result = data.plot(kind='bar', rot=0, figsize=(20, 12), cmap='Set2');
plt.title('Balanced Dataset')
result.legend(bbox_to_anchor=(1, 1.02), loc='upper left');
plt.show()

