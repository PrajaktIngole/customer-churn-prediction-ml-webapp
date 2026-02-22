import pandas as pd
import mysql.connector
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="churn_db"
)

# Load data into pandas DataFrame
df = pd.read_sql("SELECT * FROM customers", conn)

# Close connection
conn.close()

print("Data Loaded Successfully")
print(df.head())
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

print("\nChurn Distribution:")
print(df['Churn'].value_counts())

# Set style
sns.set(style="whitegrid")

# Churn Count Plot
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title("Customer Churn Distribution")
plt.show()


plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.show()


# Convert target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print(df['Churn'].head())

df = df.drop('customerID', axis=1)

df = pd.get_dummies(df, drop_first=True)

print("After Encoding Shape:", df.shape)



X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

print("Model Trained Successfully")


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Save model
pickle.dump(model, open("churn_model.pkl", "wb"))

# Save column structure
pickle.dump(X.columns, open("model_columns.pkl", "wb"))