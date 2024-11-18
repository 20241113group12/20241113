import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 讀取資料，嘗試指定編碼與處理表頭
file_path = r"C:/Users/yuchia/Downloads/test.csv"

# 讀取資料，嘗試指定編碼與處理表頭
data = pd.read_csv(file_path, encoding='utf-8')

# 統計特徵中是否認購定期存款(y)的個數
features = ['age', 'job', 'marital', 'education', 'loan']
for feature in features:
    plt.figure(figsize=(10, 5))
    data.groupby([feature, 'y']).size().unstack().plot(kind='bar', stacked=True)
    plt.title(f'Count of Subscription by {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()

# 保留age、balance、loan三個特徵來訓練羅吉斯回歸的模型

# 選擇特徵和目標變數
X = data[['age', 'balance', 'loan']]
y = data['y'].apply(lambda x: 1 if x == 'yes' else 0)

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 訓練羅吉斯回歸模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 預測與評估模型
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.2f}')