# === Import Required Libraries ===
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# === Step 1: Load and Filter the Data ===
PRODUCT_NAME = "Men's Street Footwear"
DATA_FILE = "Adidas US Sales.csv"

print(f"🔍 Loading data for product: {PRODUCT_NAME}")

# Load the CSV and convert date column to datetime
df = pd.read_csv(DATA_FILE)
df['invoice_date'] = pd.to_datetime(df['invoice_date'])

# Filter only for the specified product
df = df[df['product'] == PRODUCT_NAME]

# === Step 2: Aggregate Weekly and Engineer Features ===

# Group data by week and aggregate values
weekly = df.groupby(pd.Grouper(key='invoice_date', freq='W')).agg({
    'units_sold': 'sum',
    'price_per_unit': 'mean',
    'total_sales': 'sum',
    'operating_profit': 'mean'
}).reset_index()

# Extract time-based features
weekly['year'] = weekly['invoice_date'].dt.year
weekly['month'] = weekly['invoice_date'].dt.month
weekly['weekofyear'] = weekly['invoice_date'].dt.isocalendar().week
weekly['dayofweek'] = weekly['invoice_date'].dt.dayofweek

# Create lag features and rolling window statistics
weekly['lag_1'] = weekly['units_sold'].shift(1)
weekly['lag_2'] = weekly['units_sold'].shift(2)
weekly['rolling_mean_3'] = weekly['units_sold'].rolling(3).mean()
weekly['rolling_std_3'] = weekly['units_sold'].rolling(3).std()

# Drop any rows with missing values caused by lag/rolling operations
weekly.dropna(inplace=True)

# === Step 3: Define Features and Train the Model ===

# Select features and target variable
features = [
    'year', 'month', 'weekofyear', 'dayofweek',
    'price_per_unit', 'total_sales', 'operating_profit',
    'lag_1', 'lag_2', 'rolling_mean_3', 'rolling_std_3'
]
target = 'units_sold'

X = weekly[features]
y = weekly[target]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate performance
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📈 Results for: {PRODUCT_NAME}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# === Step 4: Save Model and Feature List ===

# Save the trained model
joblib.dump(model, "demand_prediction_rf.pkl")

# Save the list of features used
with open("demand_model_features.txt", "w") as f:
    f.write("\n".join(features))

# === Step 5: Predict Full Dataset and Save Results ===

# Predict for all weekly data
weekly['predicted_units_sold'] = model.predict(X)
weekly['product'] = PRODUCT_NAME

# Save the predictions to CSV
weekly[['invoice_date', 'product', 'units_sold', 'predicted_units_sold']].to_csv("demand_predictions.csv", index=False)
print("✅ Predictions saved to demand_predictions.csv")

# === Step 6: Plot Actual vs Predicted Units Sold ===

plt.figure(figsize=(10, 6))
plt.plot(weekly['invoice_date'], weekly['units_sold'], label='Actual Demand', color='blue', linewidth=2)
plt.plot(weekly['invoice_date'], weekly['predicted_units_sold'], label='Predicted Demand', color='red', linestyle='--', linewidth=2)
plt.title(f"Actual vs Predicted Demand for {PRODUCT_NAME}")
plt.xlabel('Date')
plt.ylabel('Units Sold')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


