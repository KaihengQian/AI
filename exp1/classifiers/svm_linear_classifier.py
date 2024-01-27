import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC


# 训练SVM模型（使用线性内核）
def train_svm_linear_model(X_train, X_val, y_train, y_val, classifier_path, report_path, classifier_index):
    # 初始化SVM分类器（使用线性内核）
    svm_linear_classifier = SVC(kernel='linear', random_state=42)

    # 训练SVM分类器
    svm_linear_classifier.fit(X_train, y_train)

    # 评价SVM分类器
    y_pred = svm_linear_classifier.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"准确度: {accuracy:.3f}")
    # 保存分类报告，包括精确度、召回率和F1分数
    report = classification_report(y_val, y_pred, digits=3)
    with open(report_path[classifier_index], "w") as f:
        f.write(report)

    # 保存SVM分类器
    joblib.dump(svm_linear_classifier, classifier_path[classifier_index])
