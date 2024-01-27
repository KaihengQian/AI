import argparse
from ImageClassification import main


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Example script with argparse')

    # 添加 model 参数
    parser.add_argument('--model', type=str, help='Specify the model type')

    # 添加 lr 参数
    parser.add_argument('--lr', default=0.01, type=float, help='Specify the learning rate')

    # 添加 dropout 参数
    parser.add_argument('--dropout', default=0.0, type=float, help='Specify the dropout rate')

    # 添加 epochs 参数
    parser.add_argument('--epochs', default=10, type=int, help='Specify the epochs')

    # 解析命令行参数
    args = parser.parse_args()

    # 打印解析后的参数
    print(f'Model: {args.model}')
    print(f'Learning Rate: {args.lr}')
    print(f'Dropout Rate: {args.dropout}')
    print(f'Epochs: {args.epochs}')
    print()

    # 接收参数，开始图像分类
    main(args)
