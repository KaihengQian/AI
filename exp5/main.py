import argparse
from SentimentAnalysis import main


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Example script with argparse')

    # 添加 text 参数
    parser.add_argument('--text', default='gru', type=str, help='Specify the text model type')

    # 添加 image 参数
    parser.add_argument('--image', default='mobilenet', type=str, help='Specify the image model type')

    # 添加 epochs 参数
    parser.add_argument('--epochs', default=30, type=int, help='Specify the epochs')

    # 解析命令行参数
    args = parser.parse_args()

    text_model_types = ['rnn', 'lstm', 'gru', 'bert']
    image_model_types = ['alexnet', 'mobilenet']
    if args.text not in text_model_types or args.image not in image_model_types:
        print("Nonexistent text/image model type!")

    else:
        # 打印解析后的参数
        print(f'Text Model: {args.text.upper()}')
        print(f'Image Model: {args.image.upper()}')
        print(f'Epochs: {args.epochs}')
        print()

        # 接收参数，开始多模态情感分析
        main(args)
