import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load and Prepare Weekly Data
df = pd.read_csv("Adidas US Sales.csv")
df['invoice_date'] = pd.to_datetime(df['invoice_date'])
df = df[df['product'] == "Men's Street Footwear"]

# Aggregate weekly
weekly = df.groupby(pd.Grouper(key='invoice_date', freq='W')).agg({
    'units_sold': 'sum',
    'price_per_unit': 'mean',
    'total_sales': 'sum',
    'operating_profit': 'mean'
}).reset_index()

# Feature engineering
weekly['year'] = weekly['invoice_date'].dt.year
weekly['month'] = weekly['invoice_date'].dt.month
weekly['weekofyear'] = weekly['invoice_date'].dt.isocalendar().week
weekly['dayofweek'] = weekly['invoice_date'].dt.dayofweek
weekly['lag_1'] = weekly['units_sold'].shift(1)
weekly['lag_2'] = weekly['units_sold'].shift(2)
weekly['rolling_mean_3'] = weekly['units_sold'].rolling(3).mean()
weekly['rolling_std_3'] = weekly['units_sold'].rolling(3).std()
weekly.dropna(inplace=True)

# Step 2: Create Features & Target
features = ['year', 'month', 'weekofyear', 'dayofweek', 'price_per_unit',
            'total_sales', 'operating_profit', 'lag_1', 'lag_2',
            'rolling_mean_3', 'rolling_std_3']
target = 'units_sold'

X = weekly[features]
y = weekly[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train and Compare Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42)
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

# Show model performance
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)

# Step 4: Final Demand Prediction using Best Model (Random Forest)
best_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
best_model.fit(X, y)

# Forecast next week (example)
last_row = weekly.iloc[-1:].copy()
next_week = last_row.copy()
next_week['invoice_date'] = next_week['invoice_date'] + pd.Timedelta(weeks=1)
next_week['lag_1'] = last_row['units_sold'].values
next_week['lag_2'] = last_row['lag_1'].values
next_week['rolling_mean_3'] = weekly['units_sold'].tail(3).mean()
next_week['rolling_std_3'] = weekly['units_sold'].tail(3).std()
next_week['year'] = next_week['invoice_date'].dt.year
next_week['month'] = next_week['invoice_date'].dt.month
next_week['weekofyear'] = next_week['invoice_date'].dt.isocalendar().week
next_week['dayofweek'] = next_week['invoice_date'].dt.dayofweek

X_future = next_week[features]
future_prediction = best_model.predict(X_future)[0]

print(f"\n📈 Predicted units sold for next week ({next_week['invoice_date'].dt.date.values[0]}): {int(future_prediction)} units")

# Optional: Plot predictions
y_pred_full = best_model.predict(X)
plt.figure(figsize=(12, 5))
plt.plot(weekly['invoice_date'], y, label='Actual')
plt.plot(weekly['invoice_date'], y_pred_full, label='Predicted')
plt.title("Actual vs Predicted Units Sold")
plt.xlabel("Week")
plt.ylabel("Units Sold")
plt.legend()
plt.tight_layout()
plt.show()
