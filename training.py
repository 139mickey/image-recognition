from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import svm

import numpy as np
import matplotlib.pyplot as plt
import joblib

# 加载数据
dt_digits = load_digits()
# 查看图像数组形状
dt_images = dt_digits.images
dt_target = dt_digits.target
# print(dt_images.shape)
# print(dt_target)
X = dt_digits.data
y = dt_digits.target

# 数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 SVM 模型并拟合数据
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# 使用训练好的模型进行预测
# y_pred = clf.predict(X_test)

# 先查看测试集中的样本
# print(X_test[0])


# 保存模型

joblib.dump(clf, r'./model/svm_model.joblib')
# 加载模型
clf_ld_mod = joblib.load(r'./model/svm_model.joblib')
# 使用加载的模型进行预测
y_prediction = clf_ld_mod.predict(X_test)

# 输出测试集的图片于预测结果和实际结果，进行观察
plt.figure(figsize=(20, 20))

for index in range(1, 101):
    axes = plt.subplot(10, 10, index)
    axes.axis('off')
    dt_img = X_test[index - 1]
    img = np.reshape(dt_img, (8, 8))
    true_value = y_test[index - 1]
    predict_value = y_prediction[index - 1]
    title = "T:{},P:{}".format(true_value, predict_value)
    color = "green"
    if true_value != predict_value:
        color = "red"
    plt.title(title, color="green")
    plt.imshow(img, cmap="gray")
#    plt.show()
# 保存图形
plt.savefig('test/figure.png')

# 计算模型的精度
accuracy = clf.score(X_test, y_test)
print("模型精度：", accuracy)
