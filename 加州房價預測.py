import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 1. 資料讀取與檢查
file_path = "C:/Users/yuchia/Desktop/Python/期末專題/California House Price/1553768847-housing.csv"
data = pd.read_csv(file_path)

# 2. 處理缺失值
# 只對數值型欄位進行均值填補
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())


# 將 ocean_proximity 類別型欄位轉換為數值
data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)

# 提取目標變量與特徵
target = data["median_house_value"]  # 假設此欄為房價
features = data.drop("median_house_value", axis=1)

# 3. 數據標準化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 4. 輸出最後十筆資料
print("最後十筆資料：")
print(data.tail(10))

# 5. 基本統計信息
max_price = target.max()
min_price = target.min()
mean_price = target.mean()
median_price = target.median()
price_variance = target.var()
price_std = target.std()

print(f"最高房價: {max_price}")
print(f"最低房價: {min_price}")
print(f"平均房價: {mean_price}")
print(f"中位數房價: {median_price}")
print(f"房價變異數: {price_variance}")
print(f"房價標準差: {price_std}")

# 6. Plotting the distribution of house prices
plt.hist(target, bins=range(0, int(max_price), 50000), alpha=0.7, color='blue', edgecolor='black')
plt.title("Distribution of House Prices")
plt.xlabel("House Price (in thousands of dollars)")
plt.ylabel("Frequency")
plt.show()

# 7. Analysis and visualization of the number of rooms
if 'RM' in data.columns:  # Assuming 'RM' represents the number of rooms
    data["RM_rounded"] = data["RM"].round(0).astype(int)
    grouped_data = data.groupby("RM_rounded")["median_house_value"].mean()
    grouped_data.plot(kind="bar", color="skyblue", title="Average House Price by Number of Rooms")
    plt.xlabel("Number of Rooms")
    plt.ylabel("Average House Price")
    plt.show()

# 8. 模型訓練：線性回歸
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("線性回歸模型訓練完成")

# 9. 模型訓練：分類（羅吉斯回歸與決策樹）
# 將房價轉換為分類標籤（低價：0，高價：1，依據中位數）
threshold = target.median()
y_train_class = (y_train > threshold).astype(int)
y_test_class = (y_test > threshold).astype(int)

clf_log = LogisticRegression()
clf_log.fit(X_train, y_train_class)

clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_train, y_train_class)


# 10. 性能評估
for model, name in zip([clf_log, clf_dt], ["Logistic Regression", "Decision Tree"]):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test_class, y_pred)
    prec = precision_score(y_test_class, y_pred)
    recall = recall_score(y_test_class, y_pred)
    f1 = f1_score(y_test_class, y_pred)
    print(f"模型: {name}")
    print(f"準確率 (Accuracy): {acc:.2f}")
    print(f"精確率 (Precision): {prec:.2f}")
    print(f"召回率 (Recall): {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    # Plot confusion matrix with English title
    cm = confusion_matrix(y_test_class, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Low", "High"], yticklabels=["Low", "High"])
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# 11. 測試函數
# 黑盒測試
def black_box_test(model, X_sample):
    try:
        prediction = model.predict(X_sample)
        print(f"黑盒測試成功，預測結果: {prediction}")
    except Exception as e:
        print(f"黑盒測試失敗: {e}")

# 白盒測試
def white_box_test(model, X_sample, expected_shape):
    try:
        assert X_sample.shape == expected_shape, "輸入數據形狀不匹配"
        prediction = model.predict(X_sample)
        print(f"白盒測試成功，預測結果: {prediction}")
    except AssertionError as ae:
        print(f"白盒測試失敗: {ae}")
    except Exception as e:
        print(f"白盒測試失敗: {e}")

# 測試用例
black_box_test(clf_log, X_test[:5])
white_box_test(clf_log, X_test[:5], (5, X_test.shape[1]))
black_box_test(clf_dt, X_test[:5])
white_box_test(clf_dt, X_test[:5], (5, X_test.shape[1]))