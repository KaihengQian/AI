import os
import time
import numpy as np
import pandas as pd
import json
import joblib
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from classifiers.logistic_regression_classifier import train_logistic_regression_model
from classifiers.decision_tree_classifier import train_decision_tree_model, adjust_params_decision_tree
from classifiers.mlp_classifier import train_mlp_model, evaluate_mlp_model
from classifiers.svm_linear_classifier import train_svm_linear_model
from classifiers.svm_poly_classifier import train_svm_poly_model, adjust_params_svm_poly
from classifiers.svm_rbf_classifier import train_svm_rbf_model, adjust_params_svm_rbf


def initialize():
    current_directory = os.getcwd()

    folder_name1 = "models"
    folder_path1 = os.path.join(current_directory, folder_name1)
    if not os.path.exists(folder_path1):
        os.mkdir(folder_path1)
    else:
        print(f"文件夹 {folder_name1} 已经存在。")

    folder_name2 = "result"
    folder_path2 = os.path.join(current_directory, folder_name2)
    if not os.path.exists(folder_path2):
        os.mkdir(folder_path2)
    else:
        print(f"文件夹 {folder_name2} 已经存在。")

    folder_name3 = "report"
    folder_path3 = os.path.join(current_directory, folder_name3)
    if not os.path.exists(folder_path3):
        os.mkdir(folder_path3)
    else:
        print(f"文件夹 {folder_name3} 已经存在。")

    folder_name4 = "evaluation"
    folder_path4 = os.path.join(current_directory, folder_name4)
    if not os.path.exists(folder_path4):
        os.mkdir(folder_path4)
    else:
        print(f"文件夹 {folder_name4} 已经存在。")


# 读取训练集
def read_train_set(train_set_path):
    label = []
    text = []

    with open(train_set_path, 'r') as file:
        # 逐行读取
        for line in file:
            # 将每一行解析成字典
            data = json.loads(line)
            label.append(data['label'])
            text.append(data['raw'])

    return np.array(label), text


# 读取测试集
def read_test_set(test_set_path):
    index = []
    text = []

    with open(test_set_path, 'r') as file:
        # 读取第一行并忽略
        file.readline()
        # 剩余行逐行读取
        for line in file:
            id, txt = line.strip().split(', ', maxsplit=1)
            index.append(id)
            text.append(txt)

    return np.array(index), np.array(text)


# 使用TF-IDF方法将文本映射成向量
def text2vec_Tfidf(text):
    # 创建TF-IDF向量化器
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # 删除停用词，指定要保留的最大特征数量

    # 计算TF-IDF值
    tfidf_matrix = vectorizer.fit_transform(text)
    '''
    # 获取特征名（词汇）
    feature_names = vectorizer.get_feature_names_out()
    print(feature_names)
    '''
    return tfidf_matrix


# 使用Word2Vec模型前的文本预处理
def preprocess_text(text):
    # 分词
    tokens = word_tokenize(text)

    # 去除标点符号和数字
    tokens = [word for word in tokens if word.isalpha()]

    # 转换为小写字母
    tokens = [word.lower() for word in tokens]

    return tokens


# 使用Word2Vec模型将文本映射成向量
def text2vec_Word2Vec(text):
    text_matrix = []

    # 预处理文本
    preprocessed_text = [preprocess_text(tmp_text) for tmp_text in text]

    # 训练Word2Vec模型
    model = Word2Vec(preprocessed_text, vector_size=100, window=5, min_count=1, workers=4)

    # 使用Word2Vec模型获取向量
    for words in preprocessed_text:
        word_vectors = [model.wv[word] for word in words]
        text_vector = np.mean(word_vectors, axis=0)  # 文本向量为词向量求均值
        text_matrix.append(text_vector)

    return np.array(text_matrix)


# 使用BERT模型将文本映射成向量
def text2vec_Bert(text):
    # 选择一个预训练的BERT模型和相应的标记器
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # 将模型和输入数据移动到GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    # 使用BERT模型进行推理
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取位于BERT模型隐藏层的输出
    hidden_states = outputs.last_hidden_state

    return hidden_states.to('cpu').numpy()


# 训练分类器
def train_classifier(labels, vectors, classifier_path, report_path, classifier_index):
    # 将训练数据划分为训练集和验证集
    # 固定划分（8:2）
    X_train, X_val, y_train, y_val = train_test_split(vectors, labels, test_size=0.2, random_state=42)

    if classifier_index == 0:
        # 训练逻辑回归分类器
        train_logistic_regression_model(X_train, X_val, y_train, y_val, classifier_path, report_path, classifier_index)

    elif classifier_index == 1:
        # 调优参数
        # adjust_params_decision_tree(X_train, X_val, y_train, y_val)

        # 训练决策树分类器
        train_decision_tree_model(X_train, X_val, y_train, y_val, classifier_path, report_path, classifier_index)

    elif classifier_index == 2:
        # 训练MLP分类器
        train_mlp_model(X_train, X_val, y_train, y_val, classifier_path, report_path, classifier_index)

    elif classifier_index == 3:
        # 训练SVM分类器（使用线性内核）
        train_svm_linear_model(X_train, X_val, y_train, y_val, classifier_path, report_path, classifier_index)

    elif classifier_index == 4:
        # 调优参数
        # adjust_params_svm_poly(X_train, X_val, y_train, y_val)

        # 训练SVM分类器（使用多项式内核）
        train_svm_poly_model(X_train, X_val, y_train, y_val, classifier_path, report_path, classifier_index)

    elif classifier_index == 5:
        # 调优参数
        # adjust_params_svm_rbf(X_train, X_val, y_train, y_val)

        # 训练SVM分类器（使用径向基函数内核）
        train_svm_rbf_model(X_train, X_val, y_train, y_val, classifier_path, report_path, classifier_index)


# K折交叉验证评价分类器
def evaluate_classifier(labels, vectors, classifier_path, evaluation_path, classifier_index):
    classifier = joblib.load(classifier_path[classifier_index])

    K = 5
    kfold = KFold(n_splits=K, shuffle=True, random_state=42)

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    if classifier_index == 2:
        for train_index, val_index in kfold.split(vectors):
            X_train, X_val = vectors[train_index], vectors[val_index]
            y_train, y_val = labels[train_index], labels[val_index]

            accuracy = evaluate_mlp_model(X_train, X_val, y_train, y_val)
            accuracy_scores.append(accuracy)

        mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)

        with open(evaluation_path[classifier_index], "w") as f:
            f.write("Mean Accuracy: {:.3f}\n".format(mean_accuracy))

    else:
        for train_index, val_index in kfold.split(vectors):
            X_train, X_val = vectors[train_index], vectors[val_index]
            y_train, y_val = labels[train_index], labels[val_index]

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_val)

            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average='micro')
            recall = recall_score(y_val, y_pred, average='micro')
            f1 = f1_score(y_val, y_pred, average='micro')

            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        mean_precision = sum(precision_scores) / len(precision_scores)
        mean_recall = sum(recall_scores) / len(recall_scores)
        mean_f1 = sum(f1_scores) / len(f1_scores)

        with open(evaluation_path[classifier_index], "w") as f:
            f.write("Mean Accuracy: {:.3f}\n".format(mean_accuracy))
            f.write("Mean Precision: {:.3f}\n".format(mean_precision))
            f.write("Mean Recall: {:.3f}\n".format(mean_recall))
            f.write("Mean F1 Score: {:.3f}".format(mean_f1))


# 使用分类器进行预测
def classify(vectors, classifier_path, classifier_index):
    # 加载分类器
    classifier = joblib.load(classifier_path[classifier_index])

    # 在测试集上进行预测
    if classifier_index == 2:
        vectors_tensor = torch.tensor(vectors.toarray(), dtype=torch.float32)

        with torch.no_grad():
            predictions = classifier(vectors_tensor)
            _, predicted = torch.max(predictions, 1)
            results = predicted.numpy()

    elif classifier_index == 6 or classifier_index == 7:
        vectors_tensor = torch.tensor(vectors.toarray(), dtype=torch.long)

        with torch.no_grad():
            predictions = classifier(vectors_tensor)
            _, predicted = torch.max(predictions, 1)
            results = predicted.numpy()

    else:
        results = classifier.predict(vectors)

    return results


# 保存预测结果
def save_results(index, results, results_path, classifier_index):
    df = pd.DataFrame({'id': index, 'pred': results})
    df.to_csv(results_path[classifier_index], sep=',', index=False)

    with open(results_path[classifier_index], 'r') as f:
        lines = f.readlines()

    # 重新写入文件，添加空格
    with open(results_path[classifier_index], 'w') as f:
        for line in lines:
            line = line.replace(',', ', ')
            f.write(line)


def main(classifier_index):
    train_set_path = 'exp1_data/train_data.txt'
    test_set_path = 'exp1_data/test.txt'
    results_path = ['result/results1.txt',
                    'result/results2.txt',
                    'result/results3.txt',
                    'result/results4.txt',
                    'result/results5.txt',
                    'result/results6.txt']
    classifier_path = ['models/logistic_regression_classifier.pkl',
                       'models/decision_tree_classifier.pkl',
                       'models/mlp_classifier.joblib',
                       'models/svm_linear_classifier.pkl',
                       'models/svm_poly_classifier.pkl',
                       'models/svm_rbf_classifier.pkl']
    report_path = ['report/logistic_regression_report.txt',
                   'report/decision_tree_report.txt',
                   'report/mlp_report.txt',
                   'report/svm_linear_report.txt',
                   'report/svm_poly_report.txt',
                   'report/svm_rbf_report.txt']
    evaluation_path = ['evaluation/logistic_regression_evaluation.txt',
                       'evaluation/decision_tree_evaluation.txt',
                       'evaluation/mlp_evaluation.txt',
                       'evaluation/svm_linear_evaluation.txt',
                       'evaluation/svm_poly_evaluation.txt',
                       'evaluation/svm_rbf_evaluation.txt']

    initialize()

    labels, train_text = read_train_set(train_set_path)

    train_text_vectors = text2vec_Tfidf(train_text)

    train_classifier(labels, train_text_vectors, classifier_path, report_path, classifier_index)

    evaluate_classifier(labels, train_text_vectors, classifier_path, evaluation_path, classifier_index)

    indices, test_text = read_test_set(test_set_path)

    test_text_vectors = text2vec_Tfidf(test_text)

    results = classify(test_text_vectors, classifier_path, classifier_index)

    save_results(indices, results, results_path, classifier_index)


if __name__ == "__main__":
    train_set_path = 'exp1_data/train_data.txt'
    test_set_path = 'exp1_data/test.txt'
    results_path = ['result/results1.txt',
                    'result/results2.txt',
                    'result/results3.txt',
                    'result/results4.txt',
                    'result/results5.txt',
                    'result/results6.txt']
    classifier_path = ['models/logistic_regression_classifier.pkl',
                       'models/decision_tree_classifier.pkl',
                       'models/mlp_classifier.joblib',
                       'models/svm_linear_classifier.pkl',
                       'models/svm_poly_classifier.pkl',
                       'models/svm_rbf_classifier.pkl']
    report_path = ['report/logistic_regression_report.txt',
                   'report/decision_tree_report.txt',
                   'report/mlp_report.txt',
                   'report/svm_linear_report.txt',
                   'report/svm_poly_report.txt',
                   'report/svm_rbf_report.txt']
    evaluation_path = ['evaluation/logistic_regression_evaluation.txt',
                       'evaluation/decision_tree_evaluation.txt',
                       'evaluation/mlp_evaluation.txt',
                       'evaluation/svm_linear_evaluation.txt',
                       'evaluation/svm_poly_evaluation.txt',
                       'evaluation/svm_rbf_evaluation.txt']
    classifier_index = 3

    t1 = time.perf_counter()

    labels, train_text = read_train_set(train_set_path)

    train_text_vectors = text2vec_Tfidf(train_text)

    train_classifier(labels, train_text_vectors, classifier_path, report_path, classifier_index)

    evaluate_classifier(labels, train_text_vectors, classifier_path, evaluation_path, classifier_index)
    
    indices, test_text = read_test_set(test_set_path)

    test_text_vectors = text2vec_Tfidf(test_text)

    results = classify(test_text_vectors, classifier_path, classifier_index)

    save_results(indices, results, results_path, classifier_index)

    t2 = time.perf_counter()
    print("共用时", t2 - t1, "秒。")
