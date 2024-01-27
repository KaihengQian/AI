import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.evaluation import calculate_bleu_4, calculate_rouge_2


# 定义Transformer编码器封装类
class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, num_heads=4):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)

    def forward(self, x):
        embedded = self.embedding(x).permute(1, 0, 2)
        output = self.transformer(embedded)
        return output


# 定义Transformer解码器封装类
class TransformerDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=2, num_heads=4):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.transformer_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads
        )
        self.transformer = nn.TransformerDecoder(self.transformer_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, memory=None):
        embedded = self.embedding(x).permute(1, 0, 2)
        output = self.transformer(embedded, memory)
        output = self.fc_out(output)
        return output


# 定义Luong注意力层
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)


# 定义Seq2Seq模型封装类
class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, embed_size, hidden_size, encoder_type, decoder_type, sos_token, eos_token, pad_token):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.embedding = nn.Embedding(input_size, embed_size)
        if self.encoder_type == 'rnn':
            self.encoder = nn.RNN(embed_size, hidden_size)
        elif self.encoder_type == 'lstm':
            self.encoder = nn.LSTM(embed_size, hidden_size)
        elif self.encoder_type == 'gru':
            self.encoder = nn.GRU(embed_size, hidden_size)
        elif self.encoder_type == 'transformer':
            self.encoder = TransformerEncoder(input_size, hidden_size)
        if self.decoder_type == 'rnn':
            self.decoder = nn.RNN(embed_size, hidden_size)
        elif self.decoder_type == 'lstm':
            self.decoder = nn.LSTM(embed_size, hidden_size)
        elif self.decoder_type == 'gru':
            self.decoder = nn.GRU(embed_size, hidden_size)
        elif self.decoder_type == 'transformer':
            self.decoder = TransformerDecoder(output_size, hidden_size)
        self.attention = Attention(hidden_size)
        self.fc_out = nn.Linear(2 * hidden_size, output_size)

    def forward(self, device, input_seq, input_lengths, target_seq, state):
        if self.encoder_type == 'transformer':  # 编码器为Transformer
            # 编码器处理
            encoder_outputs = self.encoder(input_seq)

            if state == 'train':  # 训练阶段
                # 解码器处理
                output = self.decoder(target_seq, encoder_outputs)

            else:  # 验证及测试阶段
                # 贪婪解码
                output = self.greedy_decode(device, None, encoder_outputs, input_lengths)

        else:  # 编码器为非Transformer
            # 词嵌入处理
            embedded = self.embedding(input_seq).permute(1, 0, 2)

            # 编码器处理
            encoder_state = torch.zeros(1, len(input_seq), self.hidden_size).to(device)
            if self.encoder_type == 'lstm':  # 编码器为LSTM，state=(h_n, c_n)
                encoder_outputs, encoder_state = self.encoder(embedded, (encoder_state, encoder_state))
            else:  # 编码器为非LSTM
                encoder_outputs, encoder_state = self.encoder(embedded, encoder_state)

            # 使用编码器的最后一个隐藏状态初始化解码器
            decoder_state = encoder_state

            if state == 'train':  # 训练阶段
                # 词嵌入处理
                embedded = self.embedding(target_seq).permute(1, 0, 2)
                # 解码器处理
                decoder_outputs, decoder_states = self.decoder(embedded, decoder_state)

                # 使用注意力机制加权平均编码器输出，得到上下文
                if self.decoder_type == 'lstm':  # 解码器为LSTM，state=(h_n, c_n)
                    decoder_states = decoder_states[0]  # 取出h_n
                attn_weights = self.attention(decoder_states, encoder_outputs)
                attn_weights = attn_weights.transpose(1, 2)
                encoder_outputs = encoder_outputs.transpose(0, 1)
                context = encoder_outputs * attn_weights

                # 将上下文添加到解码器输出中
                decoder_outputs = torch.cat([decoder_outputs, context.transpose(0, 1)], dim=2)

                # 输出层
                output = self.fc_out(decoder_outputs)

            else:  # 验证及测试阶段
                # 贪婪解码
                output = self.greedy_decode(device, encoder_outputs, decoder_state, len(input_lengths))

        return output

    def greedy_decode(self, device, encoder_outputs, decoder_state, batch_size, max_len=60):
        # 初始化解码器的输入序列均为起始标记 <SOS>
        generated_sequences = torch.tensor([[self.sos_token]] * batch_size, dtype=torch.long)
        decoder_input = generated_sequences

        for _ in range(max_len):
            generated_sequences = generated_sequences.to(device)
            decoder_input = decoder_input.to(device)

            # 解码器处理
            if self.decoder_type == 'transformer':  # 解码器为Transformer
                decoder_output = self.decoder(decoder_input, decoder_state)
            else:  # 解码器为非Transformer
                # 词嵌入处理
                embedded = self.embedding(decoder_input).permute(1, 0, 2)
                # 解码器处理
                decoder_output, decoder_state = self.decoder(embedded, decoder_state)

            # 使用注意力机制加权平均编码器输出，得到上下文
            if self.decoder_type == 'lstm':  # 解码器为LSTM，state=(h_n, c_n)
                decoder_states = decoder_state[0]  # 取出h_n
            else:  # 解码器为非LSTM
                decoder_states = decoder_state
            attn_weights = self.attention(decoder_states, encoder_outputs)
            context = torch.bmm(attn_weights, encoder_outputs.transpose(0, 1))

            # 将上下文添加到解码器输出中
            decoder_output = torch.cat([decoder_output, context.transpose(0, 1)], dim=2)

            # 输出层
            output = self.fc_out(decoder_output)

            # 选择预测概率最高的单词
            _, predicted_index = output.max(2)
            predicted_token = predicted_index.transpose(0, 1)

            # 将选择的单词拼接到生成序列中
            generated_sequences = torch.cat([generated_sequences, predicted_token], dim=1)

            if self.decoder_type == 'transformer':
                # 将到当前时间步为止的生成序列作为下一个时间步的输入
                decoder_input = generated_sequences
            else:
                # 将当前时间步预测的单词作为下一个时间步的输入
                decoder_input = predicted_token

            # 检查整个批次是否均生成了终止标记（部分已生成终止标记的序列继续生成，后续根据终止标记作截断处理）
            if (predicted_token == self.eos_token).all():
                break

        return generated_sequences.tolist()


def train(device, model, optimizer, loader, num, epochs, pad_token):
    model.to(device)
    model.train()

    train_loss = []
    for epoch in range(epochs):
        total_loss = 0
        for description, description_len, diagnosis_input, diagnosis_output in loader:
            # 将待更新参数的梯度置为零
            optimizer.zero_grad()

            description = description.to(device)
            diagnosis_input = diagnosis_input.to(device)
            diagnosis_output = diagnosis_output.to(device)

            # 模型训练
            output = model(device, description, description_len, diagnosis_input, 'train')
            output = output.transpose(0, 1)

            # 计算损失
            loss = 0
            for i in range(len(description_len)):
                loss += F.cross_entropy(output[i], diagnosis_output[i], ignore_index=pad_token)

            # 反向传播，计算梯度
            loss.backward()

            # 利用梯度更新参数
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num
        print(f'Epoch: {epoch + 1}/{epochs}')
        print(f'Train Loss: {avg_loss}')
        train_loss.append(avg_loss)

    return train_loss, model


def evaluate(device, model, loader, num, vocab, eos_token, pad_token, state):
    model.to(device)
    model.eval()

    sum_bleu4 = 0
    sum_rouge2 = 0

    with torch.no_grad():
        for description, description_len, diagnosis_input, diagnosis_output in loader:
            description = description.to(device)
            diagnosis_input = diagnosis_input.to(device)
            diagnosis_output = diagnosis_output.to(device)

            # 模型预测
            predict = model(device, description, description_len, None, state)

            # 计算评估指标
            bleu4 = 0
            rouge_2 = 0

            for i in range(len(diagnosis_output)):
                # 根据终止标记对预测序列作截断处理
                eos_index = predict[i].index(eos_token) if eos_token in predict[i] else None
                if eos_index is not None:
                    predict_valid = predict[i][:eos_index + 1]
                else:
                    predict_valid = predict[i]
                # 词表处理，index转token
                predict_valid = vocab.convert_ids_to_tokens(predict_valid[1:])
                target_valid = [x for x in diagnosis_output[i] if x != pad_token]
                target_valid = vocab.convert_ids_to_tokens(target_valid)

                bleu4 += calculate_bleu_4(predict_valid, target_valid)
                rouge_2 += calculate_rouge_2(predict_valid, target_valid)

            sum_bleu4 += bleu4
            sum_rouge2 += rouge_2

    avg_bleu4 = sum_bleu4 / num
    avg_rouge2 = sum_rouge2 / num

    print('BLEU-4: ', avg_bleu4)
    print('ROUGE-2: ', avg_rouge2)

    return avg_bleu4, avg_rouge2
