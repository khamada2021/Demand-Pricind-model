import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

# === Step 3: Hyperparameter Tuning for Random Forest ===
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_
print("✅ Best Random Forest Parameters:", grid_search.best_params_)

# === Step 4: Cross-validation for Gradient Boosting ===
gbr_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)
gbr_cv_scores = cross_val_score(gbr_model, X, y, cv=5, scoring='r2')
print(f"✅ Gradient Boosting Cross-Validated R² Avg: {gbr_cv_scores.mean():.4f}")

# === Step 5: Train and compare models ===
models = {
    "Linear Regression (scaled)": Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ]),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Best Random Forest Regressor": best_rf_model,
    "Gradient Boosting Regressor": gbr_model
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

# === Step 6: Save results ===
results_df = pd.DataFrame(results)
results_df.to_csv("model_evaluation_results.csv", index=False)
print("✅ Results saved to model_evaluation_results.csv")

# === Step 7: Final Prediction using Best Model ===
best_model = gbr_model
best_model.fit(X, y)
df['predicted_price_per_unit'] = best_model.predict(X)

df[['invoice_date', 'product', 'price_per_unit', 'predicted_price_per_unit']].to_csv("price_predictions.csv", index=False)
print("✅ Price predictions saved to price_predictions.csv")


# === Step 8: Plot Actual vs Predicted as Scatter Plot ===
plt.figure(figsize=(10, 6))
plt.scatter(df['invoice_date'], df['price_per_unit'], label='Actual Price', alpha=0.6)
plt.scatter(df['invoice_date'], df['predicted_price_per_unit'], label='Predicted Price', alpha=0.6)
plt.title(f"Actual vs Predicted Price per Unit for {PRODUCT_NAME}")
plt.xlabel('Date')
plt.ylabel('Price per Unit')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

