import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1. Data Loading
file_path = "C:/Users/yuchia/Desktop/作業/Python/期末專題/California House Price/1553768847-housing.csv"
data = pd.read_csv(file_path)

# 2. Data Cleaning and Preprocessing
# Handle missing values (only for numeric columns)
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Remove outliers based on interquartile range (IQR)
Q1 = data[numeric_columns].quantile(0.25)
Q3 = data[numeric_columns].quantile(0.75)
IQR = Q3 - Q1
filter = ~((data[numeric_columns] < (Q1 - 1.5 * IQR)) | (data[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)
data = data[filter]

# Convert categorical column `ocean_proximity` to numerical
data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)

# Feature selection (correlation threshold)
correlation_matrix = data.corr()
target_corr = correlation_matrix["median_house_value"].drop("median_house_value")
selected_features = target_corr[abs(target_corr) > 0.1].index.tolist()
features = data[selected_features]
target = data["median_house_value"]

# EDA - Exploratory Data Analysis
# Basic statistics
max_price = round(target.max(), 2)
min_price = round(target.min(), 2)
mean_price = round(target.mean(), 2)
median_price = round(target.median(), 2)
price_variance = round(target.var(), 2)
price_std = round(target.std(), 2)

print(f"Highest Price: {max_price}")
print(f"Lowest Price: {min_price}")
print(f"Average Price: {mean_price}")
print(f"Median Price: {median_price}")
print(f"Variance of Price: {price_variance}")
print(f"Standard Deviation of Price: {price_std}")

# House price distribution plot
plt.hist(target, bins=50, color='blue', edgecolor='black', alpha=0.7)
plt.title("Distribution of House Prices")
plt.xlabel("House Price (in thousands)")
plt.ylabel("Frequency")
plt.show()

# Average house price by average rooms (RM)
data["RM"] = data["total_rooms"] / data["households"]
data["RM_rounded"] = data["RM"].round(0).astype(int)
grouped_data = data.groupby("RM_rounded")["median_house_value"].mean()
grouped_data = grouped_data.round(2)
grouped_data.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Average House Price by Number of Rooms (Rounded)")
plt.xlabel("Number of Rooms (Rounded)")
plt.ylabel("Average House Price")
plt.xticks(rotation=0, fontsize=10)
plt.gcf().set_size_inches(15, 5)
plt.show()

# 3. Data Standardization
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 4. Data Splitting
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# 5. Model Training
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Bagging": BaggingRegressor(random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = round(mean_squared_error(y_test, y_pred), 2)
    rmse = round(np.sqrt(mse), 2)
    r2 = round(r2_score(y_test, y_pred), 2)
    results[name] = {"MSE": mse, "RMSE": rmse, "R2": r2}

# 6. Evaluation
for model_name, metrics in results.items():
    print("-")
    print(f"Model: {model_name}")
    print(f"Mean Squared Error (MSE): {metrics['MSE']:.2f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f}")
    print(f"R-Squared (R2): {metrics['R2']:.2f}")