
# price_model_comparison.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# === Step 1: Load and filter data ===
PRODUCT_NAME = "Men's Street Footwear"
DATA_FILE = "Adidas US Sales.csv"

print(f"🔍 Loading data for product: {PRODUCT_NAME}")
df = pd.read_csv(DATA_FILE)
df['invoice_date'] = pd.to_datetime(df['invoice_date'])
df = df[df['product'] == PRODUCT_NAME]

# === Step 2: Create features ===
df['year'] = df['invoice_date'].dt.year
df['month'] = df['invoice_date'].dt.month
df['day'] = df['invoice_date'].dt.day
df['dayofweek'] = df['invoice_date'].dt.dayofweek
df['weekofyear'] = df['invoice_date'].dt.isocalendar().week
df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

features = ['units_sold', 'total_sales', 'operating_profit',
            'year', 'month', 'day', 'dayofweek', 'weekofyear', 'is_weekend']
target = 'price_per_unit'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 3: Train and evaluate models ===
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=150, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "R² Score": round(r2, 4)
    })
    print(f"{name}: MAE = {mae:.2f}, R² = {r2:.4f}")

# === Step 4: Save results ===
results_df = pd.DataFrame(results)
results_df.to_csv("price_model_validation_results.csv", index=False)
print("✅ Validation results saved to price_model_validation_results.csv")
