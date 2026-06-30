print("TRAIN.PY STARTED")
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("Churn.Csv")

# customerID prediction me use nahi hoga
df = df.drop("customerID", axis=1)

# TotalCharges ko numeric banao
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Missing values hatao
df.dropna(inplace=True)

# Features & target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})

# categorical columns
cat_cols = X.select_dtypes(include="object").columns

# preprocessing
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
], remainder="passthrough")

# pipeline model
model = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000))
])

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# train
model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)

# Report
print(classification_report(y_test, y_pred))

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.savefig("confusion_matrix.png")
plt.show()

# save model
joblib.dump(model, "model.pkl")

print("Model trained & saved successfully!")