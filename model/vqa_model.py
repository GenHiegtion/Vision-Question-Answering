import torch
import torch.nn as nn
from .my_tokenizer import load_bert_tokenizer

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class CNNModel(nn.Module):
    def __init__(self, output_dim=64):
        super(CNNModel, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_dim)

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(Block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(Block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class VQAModel(nn.Module):
    def __init__(
        self,
        n_classes,
        embedding_dim=32,
        n_layers=2,
        hidden_size=64,
        drop_p=0.0,
        proj_dim=32,
        bidirect=False,
    ):
        super(VQAModel, self).__init__()
        
        self.image_encoder = CNNModel(output_dim = hidden_size)

        tokenizer = load_bert_tokenizer()
        self.embedding = nn.Embedding(tokenizer.vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirect,
            dropout=drop_p
        )

        self.MLP = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, n_classes)
        )

        self.regressor = nn.Linear(hidden_size + hidden_size, 128)

    def forward(self, img, text):
        img_features = self.image_encoder(img)

        text_emb = self.embedding(text)
        lstm_out, _ = self.lstm(text_emb)
        lstm_out = lstm_out[:, -1, :]

        x = torch.cat((img_features, lstm_out), dim=1)

        regressor_output = self.regressor(x)

        x = self.MLP(x)

        return x, regressor_output
    
def load_model(device):
    model = VQAModel(
        n_classes=2,
        embedding_dim=32,
        n_layers=2,
        hidden_size=64,
        drop_p=0.0,
        proj_dim=32,
        bidirect=False
    ).to(device)
    model.load_state_dict(torch.load("model/vqa_model.pt", map_location=device))
    model.eval()
    return model