from TextClassification import main


# 本程序运行总时长约为7~8分钟
if __name__ == "__main__":
    # 依次使用6种不同的机器学习分类算法进行文本分类
    for i in range(6):
        classifier_index = i
        main(classifier_index)
        print("Done!")
