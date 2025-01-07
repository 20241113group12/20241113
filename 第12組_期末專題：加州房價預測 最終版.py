import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from scipy.stats import boxcox

# 1. 資料加載
file_path = "C:/Users/yuchia/Desktop/作業/Python/期末專題/California House Price/1553768847-housing.csv"
data = pd.read_csv(file_path)

# 輸出前十筆資料
print("前十筆資料：")
print(data.head(10))

# 計算房價的基本統計量
print("\n房價的基本統計量：")
print(f"最高房價: {data['median_house_value'].max()}")
print(f"最低房價: {data['median_house_value'].min()}")
print(f"平均房價: {data['median_house_value'].mean():,.2f}")
print(f"中位數房價: {data['median_house_value'].median()}")
print(f"房價變異數: {data['median_house_value'].var():,.2f}")
print(f"房價標準差: {data['median_house_value'].std():,.2f}")
print()

# 原始房價分布
plt.figure(figsize=(12, 6))
plt.hist(data["median_house_value"], bins=25, color="skyblue", alpha=0.7, edgecolor='black')
plt.title("Original House Price Distribution")
plt.xlabel("House Price")
plt.ylabel("Frequency")
plt.show()

# 平均房價按平均房間數（RM）劃分
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

# 2. 資料清理與預處理
data.dropna(inplace=True)  # 處理缺失值
data = pd.get_dummies(data, columns=["ocean_proximity"], drop_first=True)  # 處理分類變數

# 特徵與目標變數分割
features = data.drop("median_house_value", axis=1)
target = data["median_house_value"]

# 處理目標變量的偏態
if (target <= 0).any():
    # 使用 Yeo-Johnson 變換
    transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    target_transformed = transformer.fit_transform(target.values.reshape(-1, 1)).flatten()
    print("使用 Yeo-Johnson 變換處理目標變量的偏態")
else:
    # 使用 Box-Cox 變換
    target_transformed, lambda_value = boxcox(target)
    print(f"使用 Box-Cox 變換處理目標變量的偏態, Lambda 值: {lambda_value:.4f}")

# 替換目標變量為經過變換的數據
target = target_transformed

# 經過偏態處理後的房價分布
plt.figure(figsize=(12, 6))
plt.hist(target_transformed, bins=25, color="orange", alpha=0.7, edgecolor='black')
plt.title("Transformed House Price Distribution")
plt.xlabel("House Price (Transformed)")
plt.ylabel("Frequency")
plt.show()

# 3. 資料標準化
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# 4. 使用 Lasso 進行嵌入式特徵選擇
lasso = LassoCV(cv=5, random_state=42, alphas=np.logspace(-3, 0, 50))
lasso.fit(features_scaled, target)

# 獲取選擇的重要特徵
selected_features_idx = np.where(lasso.coef_ != 0)[0]  # 非零係數的索引
selected_features = features.columns[selected_features_idx]

print(f"Lasso 選擇的重要特徵數量: {len(selected_features)}")
print("重要特徵如下：")
print(selected_features)

# 使用選擇後的特徵
features_selected = features_scaled[:, selected_features_idx]

# 5. 資料分割
X_train, X_test, y_train, y_test = train_test_split(features_selected, target, test_size=0.2, random_state=42)

# 6. 模型訓練與比較
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
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results[name] = {
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

# 輸出結果
for model_name, metrics in results.items():
    mse_formatted = f"{metrics['MSE']:.2f}"
    rmse_formatted = f"{metrics['RMSE']:.2f}"
    r2_formatted = f"{metrics['R2']:.2f}"
    print(f"{model_name}: MSE = {mse_formatted}, RMSE = {rmse_formatted}, R² = {r2_formatted}")

# 7. 視覺化選擇的特徵重要性
importance = np.abs(lasso.coef_[selected_features_idx])
plt.figure(figsize=(12, 6))
plt.barh(selected_features, importance, color="skyblue")
plt.title("Feature Importance (Lasso Regression)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.gca().invert_yaxis()
plt.show()
