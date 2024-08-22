import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv('student_performance.csv')

# Handle missing values
num_features = ['Study Hours per Week', 'Attendance Rate', 'Previous Grades']
cat_features = ['Participation in Extracurricular Activities', 'Parent Education Level']

# Impute numerical features with mean
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='mean')
data[num_features] = num_imputer.fit_transform(data[num_features])

# Impute categorical features with the most frequent value
cat_imputer = SimpleImputer(strategy='most_frequent')
data[cat_features] = cat_imputer.fit_transform(data[cat_features])

# Encode categorical features
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Encode the target variable
target_encoder = LabelEncoder()
data['Passed'] = target_encoder.fit_transform(data['Passed'])

# Features and target
features = num_features + cat_features
target = 'Passed'

X = data[features]
y = data[target]

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Improved Model Accuracy: {accuracy:.2f}')

# Save the model, label encoders, and target encoder
joblib.dump(model, 'improved_student_performance_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')

# Print label encoder classes for debugging
for col, le in label_encoders.items():
    print(f"{col} classes: {le.classes_}")

print("Target encoder classes:", target_encoder.classes_)
