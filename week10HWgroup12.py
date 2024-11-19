import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 修改為本地的正確路徑
# train_path = "C:/Users/Administrator/Desktop/archive/train.csv"
# test_path = "C:/Users/Administrator/Desktop/archive/test.csv"

train_path = "C:/Users/yuchia/Downloads/archive/train.csv"
test_path = "C:/Users/yuchia/Downloads/archive/test.csv"

# 讀取資料，指定分隔符號為 ";"
train_data = pd.read_csv(train_path, sep=";")
test_data = pd.read_csv(test_path, sep=";")

# 檢查缺失值
print("Train Data Missing Values:")
print(train_data.isnull().sum())
print("\nTest Data Missing Values:")
print(test_data.isnull().sum())

# 特徵分析並繪製長條圖
features = ["age", "job", "marital", "education", "loan"]
for feature in features:
    plt.figure(figsize=(10, 6))
    train_data.groupby(feature)["y"].value_counts().unstack().plot(kind="bar", stacked=True)
    plt.title(f"{feature.capitalize()} vs Subscription (y)")
    plt.ylabel("Count")
    plt.xlabel(feature.capitalize())
    plt.legend(["No", "Yes"], title="Subscription")
    plt.show()

# 資料預處理
# 將 "y" 和 "loan" 轉換為數值型
train_data["y"] = train_data["y"].apply(lambda x: 1 if x == "yes" else 0)
train_data["loan"] = train_data["loan"].apply(lambda x: 1 if x == "yes" else 0)

# 特徵選擇
X = train_data[["age", "balance", "loan"]]
y = train_data["y"]

# 分割資料集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練羅吉斯回歸模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 測試模型
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"Validation Accuracy: {accuracy:.2f}")

# 測試集預測
test_data["loan"] = test_data["loan"].apply(lambda x: 1 if x == "yes" else 0)
X_test = test_data[["age", "balance", "loan"]]
test_predictions = model.predict(X_test)

# 輸出結果
test_data["y_pred"] = test_predictions
# output_path = "C:/Users/Administrator/Desktop/archive/test_predictions.csv"
output_path = "C:/Users/yuchia/Desktop/archive/test_predictions.csv"
test_data.to_csv(output_path, index=False)
print(f"Predictions saved to: {output_path}")