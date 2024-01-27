import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC


# 训练SVM模型（使用径向基函数内核）
def train_svm_rbf_model(X_train, X_val, y_train, y_val, classifier_path, report_path, classifier_index):
    # 初始化SVM分类器（使用径向基函数内核）
    svm_rbf_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

    # 训练SVM分类器
    svm_rbf_classifier.fit(X_train, y_train)

    # 评价SVM分类器
    y_pred = svm_rbf_classifier.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"准确度: {accuracy:.3f}")
    # 保存分类报告，包括精确度、召回率和F1分数
    report = classification_report(y_val, y_pred, digits=3)
    with open(report_path[classifier_index], "w") as f:
        f.write(report)

    # 保存SVM分类器
    joblib.dump(svm_rbf_classifier, classifier_path[classifier_index])


def adjust_params_svm_rbf(X_train, X_val, y_train, y_val):
    # 调优参数：正则化参数（C）、RBF内核的核参数（gamma）
    C_values = np.logspace(-2, 5, 8)
    gammas = np.logspace(-3, 4, 8)

    accuracies = []

    for C in C_values:
        for gamma in gammas:
            # 初始化SVM分类器（使用径向基函数内核）
            svm_rbf_classifier = SVC(kernel='rbf', C=1.0, gamma=0.1, random_state=42)

            # 训练SVM分类器
            svm_rbf_classifier.fit(X_train, y_train)

            # 评价SVM分类器
            y_pred = svm_rbf_classifier.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            accuracies.append(round(accuracy, 3))
            print(C, gamma, round(accuracy, 3))

    print("\n")
    print(accuracies.index(max(accuracies)), max(accuracies))
