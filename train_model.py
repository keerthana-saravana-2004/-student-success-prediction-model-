# train_model.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample dataset
data = pd.DataFrame({
    'Department': ['CSE', 'EEE', 'ME', 'CSE', 'ME', 'EEE'],
    'GPALevel': ['High', 'Low', 'Medium', 'Medium', 'High', 'Low'],
    'ProjectSubmitted': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
    'Approved': [1, 0, 1, 0, 1, 0]
})

X = data[['Department', 'GPALevel', 'ProjectSubmitted']]
y = data['Approved']

# Preprocessing pipeline
categorical_features = ['Department', 'GPALevel', 'ProjectSubmitted']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[('cat', categorical_transformer, categorical_features)]
)

# Complete pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train and save the model
pipeline.fit(X, y)
joblib.dump(pipeline, 'model_pipeline.pkl')

print("✅ Model trained and saved as model_pipeline.pkl")
