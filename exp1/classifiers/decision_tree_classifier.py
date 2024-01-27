import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier


# 训练决策树模型
def train_decision_tree_model(X_train, X_val, y_train, y_val, classifier_path, report_path, classifier_index):
    # 初始化决策树分类器
    decision_tree_classifier = DecisionTreeClassifier(max_depth=103, min_samples_leaf=1, random_state=42)

    # 训练决策树分类器
    decision_tree_classifier.fit(X_train, y_train)

    # 评价决策树分类器
    y_pred = decision_tree_classifier.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"准确度: {accuracy:.3f}")
    # 保存分类报告，包括精确度、召回率和F1分数
    report = classification_report(y_val, y_pred, digits=3)
    with open(report_path[classifier_index], "w") as f:
        f.write(report)

    # 保存决策树分类器
    joblib.dump(decision_tree_classifier, classifier_path[classifier_index])


# 调优参数
def adjust_params_decision_tree(X_train, X_val, y_train, y_val):
    # 调优参数：决策树的深度（max_depth）、叶节点的最⼩样本数（min_samples_leaf）
    depths = np.linspace(60, 114, 55)
    samples_leaf = np.linspace(1, 3, 3)

    accuracies = []

    for dp in depths:
        for sl in samples_leaf:
            # 初始化决策树分类器
            decision_tree_classifier = DecisionTreeClassifier(max_depth=int(dp), min_samples_leaf=int(sl),
                                                              random_state=42)

            # 训练决策树分类器
            decision_tree_classifier.fit(X_train, y_train)

            # 评价决策树分类器
            y_pred = decision_tree_classifier.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            accuracies.append(round(accuracy, 3))
            print(dp, sl, round(accuracy, 3))

    print("\n")
    print(accuracies.index(max(accuracies)), max(accuracies))
