import torch
import torch.optim as optim
from transformers import BartForConditionalGeneration, BartTokenizer

from model.evaluation import calculate_bleu_4, calculate_rouge_2


# 定义BART模型封装类
class BART:
    def __init__(self):
        # 从本地加载BART模型和分词器，如果未下载到本地，传入参数应改为self.model_name
        self.model_name = 'facebook/bart-base'
        self.model_path = 'model/models--facebook--bart-base/snapshots/aadd2ab0ae0c8268c7c9693540e9904811f36177'
        self.tokenizer = BartTokenizer.from_pretrained(self.model_path)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_path)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def train(self, device, loader, num, epochs=10):
        self.model.to(device)
        self.model.train()

        train_loss = []
        for epoch in range(epochs):
            total_loss = 0
            for input_seq, target_seq in loader:
                # 将待更新参数的梯度置为零
                self.optimizer.zero_grad()

                # 分词器处理，token转index
                input_encoded = self.tokenizer.batch_encode_plus(list(input_seq), padding=True, return_tensors='pt').to(
                    device)
                target_encoded = self.tokenizer.batch_encode_plus(list(target_seq), padding=True, return_tensors='pt'
                                                                  ).to(device)

                # 模型训练
                output = self.model(input_ids=input_encoded["input_ids"], attention_mask=input_encoded["attention_mask"]
                                    , labels=target_encoded["input_ids"])

                loss = output.loss

                # 反向传播，计算梯度
                loss.backward()

                # 利用梯度更新参数
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / num
            print(f'Epoch: {epoch + 1}/{epochs}')
            print(f'Train Loss: {avg_loss}')
            train_loss.append(avg_loss)

        return train_loss

    def evaluate(self, device, loader, num):
        self.model.to(device)
        self.model.eval()

        sum_bleu4 = 0
        sum_rouge2 = 0

        with torch.no_grad():
            for input_seq, target_seq in loader:
                # 分词器处理，token转index
                input_encoded = self.tokenizer.batch_encode_plus(list(input_seq), padding=True, return_tensors='pt').to(
                    device)

                # 模型预测
                output_ids = self.model.generate(input_ids=input_encoded["input_ids"],
                                                 attention_mask=input_encoded["attention_mask"], max_length=60,
                                                 num_beams=4, length_penalty=2.0, no_repeat_ngram_size=3)

                # 分词器处理，index转token
                outputs = []
                for i in range(len(output_ids)):
                    output = self.tokenizer.decode(output_ids[i], skip_special_tokens=True)
                    outputs.append(output)

                # 计算评估指标
                bleu4 = 0
                rouge_2 = 0

                for i in range(len(target_seq)):
                    output_valid = outputs[i].split()
                    target_seq_valid = target_seq[i].split()
                    bleu4 += calculate_bleu_4(output_valid, target_seq_valid)
                    rouge_2 += calculate_rouge_2(output_valid, target_seq_valid)

                sum_bleu4 += bleu4
                sum_rouge2 += rouge_2

        avg_bleu4 = sum_bleu4 / num
        avg_rouge2 = sum_rouge2 / num

        print('BLEU-4: ', avg_bleu4)
        print('ROUGE-2: ', avg_rouge2)
        print()

        return avg_bleu4, avg_rouge2
