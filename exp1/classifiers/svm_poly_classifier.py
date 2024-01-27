import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC


# 训练SVM模型（使用多项式内核）
def train_svm_poly_model(X_train, X_val, y_train, y_val, classifier_path, report_path, classifier_index):
    # 初始化SVM分类器（使用多项式内核）
    svm_poly_classifier = SVC(kernel='poly', degree=2, random_state=42)

    # 训练SVM分类器
    svm_poly_classifier.fit(X_train, y_train)

    # 评价SVM分类器
    y_pred = svm_poly_classifier.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"准确度: {accuracy:.3f}")
    # 保存分类报告，包括精确度、召回率和F1分数
    report = classification_report(y_val, y_pred, digits=3)
    with open(report_path[classifier_index], "w") as f:
        f.write(report)

    # 保存SVM分类器
    joblib.dump(svm_poly_classifier, classifier_path[classifier_index])


def adjust_params_svm_poly(X_train, X_val, y_train, y_val):
    # 调优参数：多项式的阶数(degree)
    degrees = np.linspace(2, 5, 4)

    accuracies = []

    for dg in degrees:
        # 初始化SVM分类器（使用多项式内核）
        svm_poly_classifier = SVC(kernel='poly', degree=int(dg), random_state=42)

        # 训练SVM分类器
        svm_poly_classifier.fit(X_train, y_train)

        # 评价SVM分类器
        y_pred = svm_poly_classifier.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(accuracy)
        print(dg, accuracy)

    print("\n")
    print(accuracies.index(max(accuracies)), max(accuracies))
