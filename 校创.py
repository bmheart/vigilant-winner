import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# 步骤1：导入必要的库
file_path = "C:\\Users\\eea\\Desktop\\肾透析校创\\测试代码用2.xlsx"

# 步骤2：加载数据集
df = pd.read_excel(file_path)

# 步骤3：数据预处理
# 将所有列的非数值型值替换为 NaN
df = df.apply(pd.to_numeric, errors='coerce')

# 检查并处理完全为空的列
for column in df.columns:
    if df[column].isnull().all():
        # 这里选择删除列，也可以选择填充一个默认值，例如：df[column].fillna(0, inplace=True)
        df.drop(column, axis=1, inplace=True)

# 创建一个Imputer对象
imputer = SimpleImputer(strategy='mean')  # 可根据需要选择不同的填充策略

# 对数据集中的数值型列进行缺失值填充
df[df.select_dtypes(include=['float64', 'int64']).columns] = imputer.fit_transform(df.select_dtypes(include=['float64', 'int64']))

# 步骤4：特征选择
X = df.drop(["是否发生透析中低血压"], axis=1)

# 步骤5：目标变量
y = df["是否发生透析中低血压"]

# 步骤6：划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步骤7：特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 步骤8：训练模型
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 步骤9：模型评估
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
Accuracy: 0.8959775799538411
Confusion Matrix:
 [[3642  278]
 [ 353 1793]]
Classification Report:
               precision    recall  f1-score   support

         0.0       0.91      0.93      0.92      3920
         1.0       0.87      0.84      0.85      2146

    accuracy                           0.90      6066
   macro avg       0.89      0.88      0.89      6066
weighted avg       0.90      0.90      0.90      6066


相关可视化
[24]
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay, ConfusionMatrixDisplay

# 设置中文显示字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues, values_format='d')
disp.ax_.set_title('混淆矩阵')
plt.show()

# ROC 曲线和 AUC 可视化
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
roc_auc = auc(fpr, tpr)
roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='示例模型')
roc_disp.plot()
roc_disp.ax_.set_title('ROC 曲线')
plt.show()

# 特征重要性可视化（仅适用于支持的模型）
if hasattr(model, 'coef_'):
    feature_importance = model.coef_[0]
    features = X.columns
    plt.figure(figsize=(10, 6))
    plt.barh(features, feature_importance)
    plt.xlabel('特征重要性')
    plt.title('特征重要性图')
    plt.show()
else:
    print("模型不支持特征重要性可视化。")

二、朴素贝叶斯预测
建模代码
[25]
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB  # Import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# 步骤1：导入必要的库
file_path = "C:\\Users\\lenovo\\Desktop\\肾透析校创\\测试代码用2.xlsx"

# 步骤2：加载数据集
df = pd.read_excel(file_path)

# 步骤3：数据预处理
# 将所有列的非数值型值替换为 NaN
df = df.apply(pd.to_numeric, errors='coerce')

# 检查并处理完全为空的列
for column in df.columns:
    if df[column].isnull().all():
        df.drop(column, axis=1, inplace=True)

# 创建一个Imputer对象
imputer = SimpleImputer(strategy='mean')

# 对数据集中的数值型列进行缺失值填充
df[df.select_dtypes(include=['float64', 'int64']).columns] = imputer.fit_transform(df.select_dtypes(include=['float64', 'int64']))

# 步骤4：特征选择
X = df.drop(["是否发生透析中低血压"], axis=1)

# 步骤5：目标变量
y = df["是否发生透析中低血压"]

# 步骤6：划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步骤7：特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 步骤8：训练模型，使用朴素贝叶斯
model = GaussianNB()  # 使用 GaussianNB
model.fit(X_train_scaled, y_train)

# 步骤9：模型评估
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
Accuracy: 0.7283217936036928
Confusion Matrix:
 [[3060  860]
 [ 788 1358]]
Classification Report:
               precision    recall  f1-score   support

         0.0       0.80      0.78      0.79      3920
         1.0       0.61      0.63      0.62      2146

    accuracy                           0.73      6066
   macro avg       0.70      0.71      0.71      6066
weighted avg       0.73      0.73      0.73      6066


相关可视化（特征重要性不再适用）
[26]
# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.show()

# ROC 曲线和 AUC 可视化
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
roc_auc = auc(fpr, tpr)
roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='示例模型')
roc_disp.plot()
plt.title('ROC 曲线')
plt.show()

三、支持向量机预测
建模代码
[27]
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 步骤1：导入必要的库
file_path = "C:\\Users\\eea\\Desktop\\肾透析校创\\测试代码用2.xlsx"

# 步骤2：加载数据集
df = pd.read_excel(file_path)

# 步骤3：数据预处理
# 将所有列的非数值型值替换为 NaN
df = df.apply(pd.to_numeric, errors='coerce')

# 检查并处理完全为空的列
for column in df.columns:
    if df[column].isnull().all():
        df.drop(column, axis=1, inplace=True)

# 创建一个Imputer对象
imputer = SimpleImputer(strategy='mean')

# 对数据集中的数值型列进行缺失值填充
df[df.select_dtypes(include=['float64', 'int64']).columns] = imputer.fit_transform(df.select_dtypes(include=['float64', 'int64']))

# 步骤4：特征选择
X = df.drop(["是否发生透析中低血压"], axis=1)

# 步骤5：目标变量
y = df["是否发生透析中低血压"]

# 步骤6：划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步骤7：特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 步骤8：训练模型，使用支持向量机 (SVM)
model = SVC(probability=True)  # probability=True 可以得到概率估计，用于绘制 ROC 曲线
model.fit(X_train_scaled, y_train)

# 步骤9：模型评估

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
Accuracy: 0.9294427959116386
Confusion Matrix:
 [[3766  154]
 [ 274 1872]]
Classification Report:
               precision    recall  f1-score   support

         0.0       0.93      0.96      0.95      3920
         1.0       0.92      0.87      0.90      2146

    accuracy                           0.93      6066
   macro avg       0.93      0.92      0.92      6066
weighted avg       0.93      0.93      0.93      6066


相关可视化
[28]
# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.show()

# ROC 曲线和 AUC 可视化
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
roc_auc = auc(fpr, tpr)
roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='示例模型')
roc_disp.plot()
plt.title('ROC 曲线')
plt.show()

四、KNN预测
代码建模
[30]
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 步骤1：导入必要的库
file_path = "C:\\Users\\eea\\Desktop\\肾透析校创\\测试代码用2.xlsx"

# 步骤2：加载数据集
df = pd.read_excel(file_path)

# 步骤3：数据预处理
# 将所有列的非数值型值替换为 NaN
df = df.apply(pd.to_numeric, errors='coerce')

# 检查并处理完全为空的列
for column in df.columns:
    if df[column].isnull().all():
        df.drop(column, axis=1, inplace=True)

# 创建一个Imputer对象
imputer = SimpleImputer(strategy='mean')

# 对数据集中的数值型列进行缺失值填充
df[df.select_dtypes(include=['float64', 'int64']).columns] = imputer.fit_transform(df.select_dtypes(include=['float64', 'int64']))

# 步骤4：特征选择
X = df.drop(["是否发生透析中低血压"], axis=1)

# 步骤5：目标变量
y = df["是否发生透析中低血压"]

# 步骤6：划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步骤7：特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 步骤8：训练模型，使用 KNN
model = KNeighborsClassifier()
model.fit(X_train_scaled, y_train)

# 步骤9：模型评估
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
Accuracy: 0.8308605341246291
Confusion Matrix:
 [[3692  228]
 [ 798 1348]]
Classification Report:
               precision    recall  f1-score   support

         0.0       0.82      0.94      0.88      3920
         1.0       0.86      0.63      0.72      2146

    accuracy                           0.83      6066
   macro avg       0.84      0.78      0.80      6066
weighted avg       0.83      0.83      0.82      6066


相关可视化
[31]
# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.show()

# ROC 曲线和 AUC 可视化
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
roc_auc = auc(fpr, tpr)
roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='示例模型')
roc_disp.plot()
plt.title('ROC 曲线')
plt.show()

五、随机森林预测
代码建模
[32]
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 步骤1：导入必要的库
file_path = "C:\\Users\\eea\\Desktop\\肾透析校创\\测试代码用2.xlsx"

# 步骤2：加载数据集
df = pd.read_excel(file_path)

# 步骤3：数据预处理
# 将所有列的非数值型值替换为 NaN
df = df.apply(pd.to_numeric, errors='coerce')

# 检查并处理完全为空的列
for column in df.columns:
    if df[column].isnull().all():
        df.drop(column, axis=1, inplace=True)

# 创建一个Imputer对象
imputer = SimpleImputer(strategy='mean')

# 对数据集中的数值型列进行缺失值填充
df[df.select_dtypes(include=['float64', 'int64']).columns] = imputer.fit_transform(df.select_dtypes(include=['float64', 'int64']))

# 步骤4：特征选择
X = df.drop(["是否发生透析中低血压"], axis=1)

# 步骤5：目标变量
y = df["是否发生透析中低血压"]

# 步骤6：划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步骤7：特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 步骤8：训练模型，使用随机森林
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# 步骤9：模型评估
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
Accuracy: 0.9393339927464557
Confusion Matrix:
 [[3809  111]
 [ 257 1889]]
Classification Report:
               precision    recall  f1-score   support

         0.0       0.94      0.97      0.95      3920
         1.0       0.94      0.88      0.91      2146

    accuracy                           0.94      6066
   macro avg       0.94      0.93      0.93      6066
weighted avg       0.94      0.94      0.94      6066


相关可视化
[33]
# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.show()

# ROC 曲线和 AUC 可视化
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
roc_auc = auc(fpr, tpr)
roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='示例模型')
roc_disp.plot()
plt.title('ROC 曲线')
plt.show()

[34]
# 特征重要性可视化
feature_importance = model.feature_importances_
features = X.columns

# 排序并绘制条形图
sorted_idx = feature_importance.argsort()[::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), feature_importance[sorted_idx], align="center")
plt.xticks(range(X.shape[1]), features[sorted_idx], rotation=45)
plt.xlabel('特征')
plt.ylabel('重要性')
plt.title('随机森林特征重要性')
plt.show()

[36]
pip install pydotplus
Defaulting to user installation because normal site-packages is not writeable
Collecting pydotplus
  Downloading pydotplus-2.0.2.tar.gz (278 kB)
     ---------------------------------------- 0.0/278.7 kB ? eta -:--:--
     - -------------------------------------- 10.2/278.7 kB ? eta -:--:--
     - -------------------------------------- 10.2/278.7 kB ? eta -:--:--
     ---- -------------------------------- 30.7/278.7 kB 262.6 kB/s eta 0:00:01
     -------- ---------------------------- 61.4/278.7 kB 409.6 kB/s eta 0:00:01
     ------------ ------------------------ 92.2/278.7 kB 476.3 kB/s eta 0:00:01
     --------------- -------------------- 122.9/278.7 kB 514.3 kB/s eta 0:00:01
     --------------------- -------------- 163.8/278.7 kB 544.7 kB/s eta 0:00:01
     -------------------------- --------- 204.8/278.7 kB 655.1 kB/s eta 0:00:01
     ------------------------------- ---- 245.8/278.7 kB 628.1 kB/s eta 0:00:01
     ------------------------------------ 278.7/278.7 kB 659.5 kB/s eta 0:00:00
  Preparing metadata (setup.py): started
  Preparing metadata (setup.py): finished with status 'done'
Requirement already satisfied: pyparsing>=2.0.1 in d:\anaconda\lib\site-packages (from pydotplus) (3.0.9)
Building wheels for collected packages: pydotplus
  Building wheel for pydotplus (setup.py): started
  Building wheel for pydotplus (setup.py): finished with status 'done'
  Created wheel for pydotplus: filename=pydotplus-2.0.2-py3-none-any.whl size=24578 sha256=033d11fc6ce8a8b2cf319f98f8bef294e3782a1d136a0004b2bdab07fd9e5b00
  Stored in directory: c:\users\lenovo\appdata\local\pip\cache\wheels\bd\ce\e8\ff9d9c699514922f57caa22fbd55b0a32761114b4c4acc9e03
Successfully built pydotplus
Installing collected packages: pydotplus
Successfully installed pydotplus-2.0.2
Note: you may need to restart the kernel to use updated packages.

[39]
from sklearn.tree import plot_tree
plt.figure(figsize=(15, 10))
plot_tree(tree_model, feature_names=X.columns, class_names=['0', '1'], filled=True, rounded=True)
plt.title('Decision Tree Visualization')
plt.show()

六、ANN预测
代码建模
[41]
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 步骤2：加载数据集
df = pd.read_excel(file_path)

# 步骤3：数据预处理
# 将所有列的非数值型值替换为 NaN
df = df.apply(pd.to_numeric, errors='coerce')

# 检查并处理完全为空的列
for column in df.columns:
    if df[column].isnull().all():
        df.drop(column, axis=1, inplace=True)

# 创建一个Imputer对象
imputer = SimpleImputer(strategy='mean')

# 对数据集中的数值型列进行缺失值填充
df[df.select_dtypes(include=['float64', 'int64']).columns] = imputer.fit_transform(df.select_dtypes(include=['float64', 'int64']))

# 步骤4：特征选择
X = df.drop(["是否发生透析中低血压"], axis=1)

# 步骤5：目标变量
y = df["是否发生透析中低血压"]

# 步骤6：划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步骤7：特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 步骤8：创建神经网络模型
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_train_scaled.shape[1]))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 步骤9：训练模型
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))

# 步骤10：模型评估
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
Epoch 1/10
759/759 [==============================] - 1s 1ms/step - loss: 0.2892 - accuracy: 0.8770 - val_loss: 0.2040 - val_accuracy: 0.9141
Epoch 2/10
759/759 [==============================] - 1s 1ms/step - loss: 0.1846 - accuracy: 0.9206 - val_loss: 0.1679 - val_accuracy: 0.9294
Epoch 3/10
759/759 [==============================] - 1s 1ms/step - loss: 0.1532 - accuracy: 0.9352 - val_loss: 0.1468 - val_accuracy: 0.9385
Epoch 4/10
759/759 [==============================] - 1s 2ms/step - loss: 0.1328 - accuracy: 0.9427 - val_loss: 0.1358 - val_accuracy: 0.9420
Epoch 5/10
759/759 [==============================] - 2s 2ms/step - loss: 0.1223 - accuracy: 0.9486 - val_loss: 0.1193 - val_accuracy: 0.9484
Epoch 6/10
759/759 [==============================] - 2s 2ms/step - loss: 0.1114 - accuracy: 0.9528 - val_loss: 0.1133 - val_accuracy: 0.9515
Epoch 7/10
759/759 [==============================] - 2s 2ms/step - loss: 0.1060 - accuracy: 0.9548 - val_loss: 0.1223 - val_accuracy: 0.9469
Epoch 8/10
759/759 [==============================] - 1s 2ms/step - loss: 0.0999 - accuracy: 0.9588 - val_loss: 0.1067 - val_accuracy: 0.9527
Epoch 9/10
759/759 [==============================] - 1s 1ms/step - loss: 0.0951 - accuracy: 0.9605 - val_loss: 0.1093 - val_accuracy: 0.9529
Epoch 10/10
759/759 [==============================] - 1s 1ms/step - loss: 0.0911 - accuracy: 0.9627 - val_loss: 0.1000 - val_accuracy: 0.9588
190/190 [==============================] - 0s 725us/step
Accuracy: 0.9587866798549292
Confusion Matrix:
 [[3810  110]
 [ 140 2006]]
Classification Report:
               precision    recall  f1-score   support

         0.0       0.96      0.97      0.97      3920
         1.0       0.95      0.93      0.94      2146

    accuracy                           0.96      6066
   macro avg       0.96      0.95      0.95      6066
weighted avg       0.96      0.96      0.96      6066


相关可视化（批次增大）
[44]
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import History

# 步骤1：导入必要的库
file_path = "C:\\Users\\eea\\Desktop\\肾透析校创\\测试代码用2.xlsx"

# 步骤2：加载数据集
df = pd.read_excel(file_path)

# 步骤3：数据预处理
# 将所有列的非数值型值替换为 NaN
df = df.apply(pd.to_numeric, errors='coerce')

# 检查并处理完全为空的列
for column in df.columns:
    if df[column].isnull().all():
        df.drop(column, axis=1, inplace=True)

# 创建一个Imputer对象
imputer = SimpleImputer(strategy='mean')
# 对数据集中的数值型列进行缺失值填充
df[df.select_dtypes(include=['float64', 'int64']).columns] = imputer.fit_transform(df.select_dtypes(include=['float64', 'int64']))

# 步骤4：特征选择
X = df.drop(["是否发生透析中低血压"], axis=1)

# 步骤5：目标变量
y = df["是否发生透析中低血压"]

# 步骤6：划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步骤7：特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 步骤8：建立神经网络模型
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练历史记录
history = History()

# 步骤9：训练模型
model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_data=(X_test_scaled, y_test), callbacks=[history])

#训练曲线可视化
plot_training_history(history)

# 步骤10：模型评估
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
Epoch 1/20
759/759 [==============================] - 1s 1ms/step - loss: 0.2882 - accuracy: 0.8712 - val_loss: 0.2029 - val_accuracy: 0.9123
Epoch 2/20
759/759 [==============================] - 1s 1ms/step - loss: 0.1917 - accuracy: 0.9179 - val_loss: 0.1767 - val_accuracy: 0.9255
Epoch 3/20
759/759 [==============================] - 1s 1ms/step - loss: 0.1637 - accuracy: 0.9299 - val_loss: 0.1518 - val_accuracy: 0.9334
Epoch 4/20
759/759 [==============================] - 1s 1ms/step - loss: 0.1468 - accuracy: 0.9361 - val_loss: 0.1448 - val_accuracy: 0.9400
Epoch 5/20
759/759 [==============================] - 1s 1ms/step - loss: 0.1327 - accuracy: 0.9439 - val_loss: 0.1322 - val_accuracy: 0.9428
Epoch 6/20
759/759 [==============================] - 1s 1ms/step - loss: 0.1218 - accuracy: 0.9487 - val_loss: 0.1271 - val_accuracy: 0.9436
Epoch 7/20
759/759 [==============================] - 1s 1ms/step - loss: 0.1141 - accuracy: 0.9514 - val_loss: 0.1211 - val_accuracy: 0.9474
Epoch 8/20
759/759 [==============================] - 1s 1ms/step - loss: 0.1074 - accuracy: 0.9551 - val_loss: 0.1115 - val_accuracy: 0.9509
Epoch 9/20
759/759 [==============================] - 1s 1ms/step - loss: 0.1017 - accuracy: 0.9565 - val_loss: 0.1144 - val_accuracy: 0.9522
Epoch 10/20
759/759 [==============================] - 1s 1ms/step - loss: 0.0968 - accuracy: 0.9597 - val_loss: 0.1084 - val_accuracy: 0.9535
Epoch 11/20
759/759 [==============================] - 1s 2ms/step - loss: 0.0913 - accuracy: 0.9613 - val_loss: 0.1057 - val_accuracy: 0.9552
Epoch 12/20
759/759 [==============================] - 2s 2ms/step - loss: 0.0872 - accuracy: 0.9635 - val_loss: 0.1146 - val_accuracy: 0.9515
Epoch 13/20
759/759 [==============================] - 2s 2ms/step - loss: 0.0837 - accuracy: 0.9661 - val_loss: 0.1053 - val_accuracy: 0.9566
Epoch 14/20
759/759 [==============================] - 2s 2ms/step - loss: 0.0801 - accuracy: 0.9670 - val_loss: 0.0960 - val_accuracy: 0.9593
Epoch 15/20
759/759 [==============================] - 1s 2ms/step - loss: 0.0752 - accuracy: 0.9692 - val_loss: 0.0972 - val_accuracy: 0.9594
Epoch 16/20
759/759 [==============================] - 2s 2ms/step - loss: 0.0755 - accuracy: 0.9686 - val_loss: 0.0909 - val_accuracy: 0.9651
Epoch 17/20
759/759 [==============================] - 2s 2ms/step - loss: 0.0704 - accuracy: 0.9710 - val_loss: 0.0949 - val_accuracy: 0.9618
Epoch 18/20
759/759 [==============================] - 2s 2ms/step - loss: 0.0684 - accuracy: 0.9716 - val_loss: 0.0927 - val_accuracy: 0.9624
Epoch 19/20
759/759 [==============================] - 1s 1ms/step - loss: 0.0652 - accuracy: 0.9731 - val_loss: 0.0886 - val_accuracy: 0.9659
Epoch 20/20
759/759 [==============================] - 1s 1ms/step - loss: 0.0619 - accuracy: 0.9743 - val_loss: 0.0884 - val_accuracy: 0.9654

190/190 [==============================] - 0s 749us/step
Accuracy: 0.9653808110781404
Confusion Matrix:
 [[3839   81]
 [ 129 2017]]
Classification Report:
               precision    recall  f1-score   support

         0.0       0.97      0.98      0.97      3920
         1.0       0.96      0.94      0.95      2146

    accuracy                           0.97      6066
   macro avg       0.96      0.96      0.96      6066
weighted avg       0.97      0.97      0.97      6066



七、RNN预测
代码建模及相关可视化
[47]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import History

# 步骤1：导入必要的库
file_path = "C:\\Users\\eea\\Desktop\\肾透析校创\\测试代码用2.xlsx"

# 步骤2：加载数据集
df = pd.read_excel(file_path)

# 步骤3：数据预处理
# 将所有列的非数值型值替换为 NaN
df = df.apply(pd.to_numeric, errors='coerce')

# 检查并处理完全为空的列
for column in df.columns:
    if df[column].isnull().all():
        df.drop(column, axis=1, inplace=True)

# 创建一个Imputer对象
imputer = SimpleImputer(strategy='mean')

# 对数据集中的数值型列进行缺失值填充
df[df.select_dtypes(include=['float64', 'int64']).columns] = imputer.fit_transform(df.select_dtypes(include=['float64', 'int64']))

# 步骤4：特征选择
X = df.drop(["是否发生透析中低血压"], axis=1)

# 步骤5：目标变量
y = df["是否发生透析中低血压"]

# 步骤6：划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步骤7：特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 重塑数据维度
X_train_scaled_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_scaled_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# 步骤8：建立RNN模型
model = Sequential()
model.add(SimpleRNN(64, activation='relu', input_shape=(1, X_train_scaled.shape[1])))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练历史记录
history = History()

# 步骤9：训练模型
model.fit(X_train_scaled_reshaped, y_train, epochs=20, batch_size=32, validation_data=(X_test_scaled_reshaped, y_test), callbacks=[history])

# 训练曲线可视化
plot_training_history(history)

# 步骤10：模型评估
y_pred = (model.predict(X_test_scaled_reshaped) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
Epoch 1/20
759/759 [==============================] - 2s 2ms/step - loss: 0.2902 - accuracy: 0.8724 - val_loss: 0.1996 - val_accuracy: 0.9181
Epoch 2/20
759/759 [==============================] - 1s 1ms/step - loss: 0.1862 - accuracy: 0.9207 - val_loss: 0.1648 - val_accuracy: 0.9298
Epoch 3/20
759/759 [==============================] - 1s 1ms/step - loss: 0.1583 - accuracy: 0.9317 - val_loss: 0.1529 - val_accuracy: 0.9375
Epoch 4/20
759/759 [==============================] - 2s 2ms/step - loss: 0.1400 - accuracy: 0.9408 - val_loss: 0.1352 - val_accuracy: 0.9433
Epoch 5/20
759/759 [==============================] - 2s 2ms/step - loss: 0.1259 - accuracy: 0.9455 - val_loss: 0.1279 - val_accuracy: 0.9456
Epoch 6/20
759/759 [==============================] - 2s 2ms/step - loss: 0.1135 - accuracy: 0.9523 - val_loss: 0.1167 - val_accuracy: 0.9492
Epoch 7/20
759/759 [==============================] - 2s 2ms/step - loss: 0.1043 - accuracy: 0.9571 - val_loss: 0.1146 - val_accuracy: 0.9514
Epoch 8/20
759/759 [==============================] - 1s 2ms/step - loss: 0.0992 - accuracy: 0.9592 - val_loss: 0.1035 - val_accuracy: 0.9573
Epoch 9/20
759/759 [==============================] - 1s 1ms/step - loss: 0.0924 - accuracy: 0.9608 - val_loss: 0.1097 - val_accuracy: 0.9557
Epoch 10/20
759/759 [==============================] - 1s 1ms/step - loss: 0.0879 - accuracy: 0.9633 - val_loss: 0.1046 - val_accuracy: 0.9548
Epoch 11/20
759/759 [==============================] - 1s 1ms/step - loss: 0.0840 - accuracy: 0.9657 - val_loss: 0.0962 - val_accuracy: 0.9609
Epoch 12/20
759/759 [==============================] - 1s 1ms/step - loss: 0.0798 - accuracy: 0.9671 - val_loss: 0.1003 - val_accuracy: 0.9591
Epoch 13/20
759/759 [==============================] - 2s 2ms/step - loss: 0.0769 - accuracy: 0.9683 - val_loss: 0.0905 - val_accuracy: 0.9614
Epoch 14/20
759/759 [==============================] - 2s 2ms/step - loss: 0.0742 - accuracy: 0.9685 - val_loss: 0.0933 - val_accuracy: 0.9599
Epoch 15/20
759/759 [==============================] - 2s 2ms/step - loss: 0.0705 - accuracy: 0.9696 - val_loss: 0.0914 - val_accuracy: 0.9624
Epoch 16/20
759/759 [==============================] - 2s 2ms/step - loss: 0.0679 - accuracy: 0.9708 - val_loss: 0.0891 - val_accuracy: 0.9631
Epoch 17/20
759/759 [==============================] - 2s 3ms/step - loss: 0.0665 - accuracy: 0.9721 - val_loss: 0.0893 - val_accuracy: 0.9632
Epoch 18/20
759/759 [==============================] - 2s 3ms/step - loss: 0.0634 - accuracy: 0.9739 - val_loss: 0.0880 - val_accuracy: 0.9629
Epoch 19/20
759/759 [==============================] - 1s 2ms/step - loss: 0.0614 - accuracy: 0.9742 - val_loss: 0.0935 - val_accuracy: 0.9611
Epoch 20/20
759/759 [==============================] - 1s 1ms/step - loss: 0.0563 - accuracy: 0.9765 - val_loss: 0.0926 - val_accuracy: 0.9642

190/190 [==============================] - 0s 842us/step
Accuracy: 0.9642268381140785
Confusion Matrix:
 [[3795  125]
 [  92 2054]]
Classification Report:
               precision    recall  f1-score   support

         0.0       0.98      0.97      0.97      3920
         1.0       0.94      0.96      0.95      2146

    accuracy                           0.96      6066
   macro avg       0.96      0.96      0.96      6066
weighted avg       0.96      0.96      0.96      6066


