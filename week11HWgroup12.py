import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# 讀取檔案
train_path = "C:/Users/yuchia/Downloads/titanic/train.csv"
test_path = "C:/Users/yuchia/Downloads/titanic/test.csv"
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 印出前10筆資料
print("First 10 rows of training data:")
print(train_df.head(10))
print()

# 處理缺失值
train_df = train_df.copy()
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())

# 類別特徵轉換為數值
train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked'], drop_first=True)

# 特徵標準化
scaler = StandardScaler()
features_to_scale = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
train_df[features_to_scale] = scaler.fit_transform(train_df[features_to_scale])

# 分離特徵和目標
X = train_df.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
y = train_df['Survived']

# 划分訓練集與驗證集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練羅吉斯回歸模型
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_val)

# 訓練決策樹模型
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_val)

# 模型性能評估
print("Logistic Regression Model Performance:")
print(f"Accuracy: {accuracy_score(y_val, y_pred_log_reg):.2f}")
print(f"Precision: {precision_score(y_val, y_pred_log_reg):.2f}")
print(f"Recall: {recall_score(y_val, y_pred_log_reg):.2f}")
print(f"F1-score: {f1_score(y_val, y_pred_log_reg):.2f}")

print("\nDecision Tree Model Performance:")
print(f"Accuracy: {accuracy_score(y_val, y_pred_tree):.2f}")
print(f"Precision: {precision_score(y_val, y_pred_tree):.2f}")
print(f"Recall: {recall_score(y_val, y_pred_tree):.2f}")
print(f"F1-score: {f1_score(y_val, y_pred_tree):.2f}")

# 混淆矩陣
cm_log_reg = confusion_matrix(y_val, y_pred_log_reg)
cm_tree = confusion_matrix(y_val, y_pred_tree)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
disp_log_reg = ConfusionMatrixDisplay(confusion_matrix=cm_log_reg)
disp_tree = ConfusionMatrixDisplay(confusion_matrix=cm_tree)
disp_log_reg.plot(ax=axes[0], colorbar=False)
axes[0].set_title("Logistic Regression Confusion Matrix")
disp_tree.plot(ax=axes[1], colorbar=False)
axes[1].set_title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.show()

# 測試集預處理
test_df = test_df.copy()
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
test_df = pd.get_dummies(test_df, columns=['Sex', 'Embarked'], drop_first=True)
test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])
X_test = test_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

# 預測測試集
y_test_pred_log_reg = log_reg.predict(X_test)
y_test_pred_tree = tree_clf.predict(X_test)

# 輸出測試集預測結果
submission_log_reg = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': y_test_pred_log_reg
})
submission_tree = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': y_test_pred_tree
})

# 比較羅吉斯回歸和決策樹
print("\nModel Performance Comparison:")
# Accuracy
print("· Accuracy Comparison")
print(f"  Logistic Regression: {accuracy_score(y_val, y_pred_log_reg):.2f}")
print(f"  Decision Tree: {accuracy_score(y_val, y_pred_tree):.2f}")
# Precision
print("· Precision Comparison")
print(f"  Logistic Regression: {precision_score(y_val, y_pred_log_reg):.2f}")
print(f"  Decision Tree: {precision_score(y_val, y_pred_tree):.2f}")
# Recall
print("· Recall Comparison")
print(f"  Logistic Regression: {recall_score(y_val, y_pred_log_reg):.2f}")
print(f"  Decision Tree: {recall_score(y_val, y_pred_tree):.2f}")
# F1 Score
print("· F1 Score Comparison")
print(f"  Logistic Regression: {f1_score(y_val, y_pred_log_reg):.2f}")
print(f"  Decision Tree: {f1_score(y_val, y_pred_tree):.2f}")