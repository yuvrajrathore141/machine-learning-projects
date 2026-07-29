import joblib
from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

print("Fetching dataset...")
adult = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
data = adult.frame.dropna().copy()
data["target"] = data["class"].astype(str).str.contains(">50K").astype(int)
data = data.drop_duplicates()

features = data.drop(columns=["class", "target"])
target = data.target

X_train, X_test, y_train, y_test = train_test_split(features, target, stratify=target, random_state=42)

print("Training model...")
prep = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median")), features.select_dtypes("number").columns),
    (make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), features.select_dtypes(exclude="number").columns)
)

model = make_pipeline(prep, LogisticRegression(max_iter=500))
model.fit(X_train, y_train)

model_path = Path(__file__).parent / "model.joblib"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
