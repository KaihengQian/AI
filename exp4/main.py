import argparse
from TextSummarization import main


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Example script with argparse')

    # 添加 encoder 参数
    parser.add_argument('--encoder', default='lstm', type=str, help='Specify the encoder type')

    # 添加 decoder 参数
    parser.add_argument('--decoder', default='lstm', type=str, help='Specify the decoder type')

    # 添加 epochs 参数
    parser.add_argument('--epochs', default=50, type=int, help='Specify the epochs')

    # 解析命令行参数
    args = parser.parse_args()

    model_types = ['rnn', 'lstm', 'gru', 'bart']
    if args.encoder not in model_types or args.decoder not in model_types:
        print("Nonexistent encoder/decoder type!")

    else:
        # 打印解析后的参数
        print(f'Encoder: {args.encoder.upper()}')
        print(f'Decoder: {args.decoder.upper()}')
        print(f'Epochs: {args.epochs}')
        print()

        # 接收参数，开始文本摘要
        main(args)
