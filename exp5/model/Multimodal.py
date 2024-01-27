import torch
import torch.nn as nn
from model.TextModel import TextRNN, TextLSTM, TextGRU, TextBERT
from model.ImageModel import ImageAlexNet, ImageMobileNetV1


class MultimodalModel(nn.Module):
    def __init__(self, text_model_type, image_model_type, vocab_size, text_embed_dim, text_hidden_dim, output_dim,
                 num_classes):
        super(MultimodalModel, self).__init__()
        self.text_model_type = text_model_type
        self.image_model_type = image_model_type

        if text_model_type == 'rnn':
            self.text_model = TextRNN(vocab_size, text_embed_dim, text_hidden_dim, output_dim)
        elif text_model_type == 'lstm':
            self.text_model = TextLSTM(vocab_size, text_embed_dim, text_hidden_dim, output_dim)
        elif text_model_type == 'gru':
            self.text_model = TextGRU(vocab_size, text_embed_dim, text_hidden_dim, output_dim)
        elif text_model_type == 'bert':
            self.text_model = TextBERT(output_dim)

        if image_model_type == 'alexnet':
            self.image_model = ImageAlexNet(output_dim)
        elif image_model_type == 'mobilenet':
            self.image_model = ImageMobileNetV1(output_dim)

        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(2 * output_dim, num_classes)

    def forward(self, text, attention_mask, image):
        if self.text_model_type == 'bert':
            text_output = self.text_model(text, attention_mask)
        else:
            text_output = self.text_model(text)
        image_output = self.image_model(image)
        combined = torch.cat((text_output, image_output), dim=1)
        combined = self.relu(combined)
        output = self.fc(combined)
        return output
