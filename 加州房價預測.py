import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 加載資料集
file_path = "C:/Users/yuchia/Desktop/Python/期末專題/California House Price/1553768847-housing.csv"
data = pd.read_csv(file_path)

# 資料預處理：刪除缺失值
data = data.dropna()

# 將 ocean_proximity 類別型欄位轉換為數值
data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)

# 顯示最後十筆資料
print(data.tail(10))

# 分析數據：計算統計數據
max_price = round(data['median_house_value'].max(), 2)
min_price = round(data['median_house_value'].min(), 2)
mean_price = round(data['median_house_value'].mean(), 2)
median_price = round(data['median_house_value'].median(), 2)
var_price = round(data['median_house_value'].var(), 2)
std_price = round(data['median_house_value'].std(), 2)

print(f"Max house price: {max_price}")
print(f"Min house price: {min_price}")
print(f"Mean house price: {mean_price}")
print(f"Median house price: {median_price}")
print(f"House price variance: {var_price}")
print(f"House price standard deviation: {std_price}")

# 房價分布直方圖
plt.hist(data['median_house_value'], bins=range(0, int(max_price) + 100000, 100000), color='skyblue', edgecolor='black')
plt.xlabel('House Price')
plt.ylabel('Frequency')
plt.title('House Price Distribution')
plt.show()

# 四捨五入 RM 值，並分析平均房價
data['RM'] = data['total_rooms'] / data['households']
data['RM'] = data['RM'].round()
grouped = data.groupby('RM')['median_house_value'].mean()
grouped.plot(kind='bar', color='orange')
plt.xlabel('RM')
plt.ylabel('Average House Price')
plt.title('Average House Price by RM')
plt.show()

# 特徵標準化
features = data.drop('median_house_value', axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, data['median_house_value'], test_size=0.2, random_state=42)

# 模型訓練與評估
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 評估指標
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - Mean Squared Error (MSE): {mse:.2f}, R-squared (R2): {r2:.2f}")

# Logistic Regression（分類問題處理示例）
# 將 Price 分區間進行分類
data['Price_category'] = pd.cut(data['median_house_value'], bins=5, labels=False)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, data['Price_category'], test_size=0.2, random_state=42)
logistic_reg = LogisticRegression(max_iter=1000)
logistic_reg.fit(X_train, y_train)
y_pred_class = logistic_reg.predict(X_test)

# 分類評估指標
cm = confusion_matrix(y_test, y_pred_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.show()
