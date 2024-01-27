import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# 训练逻辑回归模型
def train_logistic_regression_model(X_train, X_val, y_train, y_val, classifier_path, report_path, classifier_index):
    # 初始化逻辑回归分类器
    logistic_regression_classifier = LogisticRegression(random_state=42, multi_class='ovr')

    # 训练逻辑回归分类器
    logistic_regression_classifier.fit(X_train, y_train)

    # 评价逻辑回归分类器
    y_pred = logistic_regression_classifier.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"准确度: {accuracy:.3f}")
    # 保存分类报告，包括精确度、召回率和F1分数
    report = classification_report(y_val, y_pred, digits=3)
    with open(report_path[classifier_index], "w") as f:
        f.write(report)

    # 保存逻辑回归分类器
    joblib.dump(logistic_regression_classifier, classifier_path[classifier_index])
