import torch
import torch.nn as nn
import torchvision.models as models

class Encoder_(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(Encoder_, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn. ReLU()
        self.dropput = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)

        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN

        return self.dropput(self.relu(features))

class Decoder_(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(Decoder_, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

class enc_dec(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(enc_dec, self).__init__()
        self.encoder_=Encoder_(embed_size)
        self.decoder_=Decoder_(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder_(images)
        outputs = self.decoder_(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoder_(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoder_.lstm(x, states)
                output = self.decoder_.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)

                result_caption.append(predicted.item())
                x = self.decoder_.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
        return [vocabulary.itos(idx) for idx in result_caption]

