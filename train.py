import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load Data ───────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv.xls")
print(f"Dataset shape: {df.shape}")

# ── 2. Clean Data ──────────────────────────────────────────────────────────────
print("\nCleaning data...")

# Drop customerID
df.drop("customerID", axis=1, inplace=True)

# Fix TotalCharges column (has spaces)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Encode target variable
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Encode all categorical columns
le = LabelEncoder()
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("Data cleaned successfully!")

# ── 3. Split Data ──────────────────────────────────────────────────────────────
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ── 4. Train & Compare Models ──────────────────────────────────────────────────
print("\nTraining models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(eval_metric="logloss", random_state=42)
}

best_acc   = 0
best_model = None

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc   = accuracy_score(y_test, preds)
    print(f"  {name}: {acc:.4f}")

    if acc > best_acc:
        best_acc   = acc
        best_model = model
        best_name  = name

# ── 5. Detailed Report on Best Model ──────────────────────────────────────────
print(f"\nBest Model: {best_name} ({best_acc:.4f})")
print("\nClassification Report:")
preds = best_model.predict(X_test_scaled)
print(classification_report(y_test, preds, target_names=["Stayed", "Churned"]))

# ── 6. Save Everything ─────────────────────────────────────────────────────────
print("Saving model, scaler, and feature names...")
joblib.dump(best_model,       "churn_model.pkl")
joblib.dump(scaler,           "churn_scaler.pkl")
joblib.dump(list(X.columns),  "feature_names.pkl")

print("\n✅ Done! Files saved:")
print("  - churn_model.pkl")
print("  - churn_scaler.pkl")
print("  - feature_names.pkl")