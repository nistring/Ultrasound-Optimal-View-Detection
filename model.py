import torch.nn as nn
import torchvision.models as models
import torch
from config import *
import torch

# Supported CNN models
cnn_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}

# Supported RNN models
rnn_dict = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}


def create_model(cnn, pretrained=True, in_chans=1, num_classes=1):
    # For resnet
    model = cnn_dict[cnn](pretrained=pretrained)
    # Modify the first conv layer
    model.conv1 = nn.Conv2d(in_chans, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # Modify the fc layer so that the number of output's channel equals num_classes
    if num_classes==0:
        model.fc = nn.Identity()
    else:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    return model


class RNN(nn.Module):
    def __init__(self, version, in_features, seq_len, latent_dim=128, hidden_dim=256, num_layers=1, num_classes=1, batch_first=True):
        super(RNN, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        args = [
            latent_dim,
            hidden_dim,
            num_layers,
            batch_first,
        ]

        self.fc1 = nn.Linear(in_features, latent_dim)
        self.recurrent_layer = rnn_dict[version](*args)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        # B*L x latent_dim
        x = x.view(-1, self.seq_len, self.latent_dim)
        # B x L x latent_dim
        x, _ = self.recurrent_layer(x)
        # B x L x hidden_dim
        x = x[:, -1]  # Take only the last output of sequence
        # B x hidden_dim
        x = self.fc2(x)
        # B x 1
        return x


class CNNRNN(nn.Module):
    # is_realtime = True only for realtime applications
    def __init__(self, cnn, rnn, seq_len, weights=None, in_chans=1, is_realtime=False):
        super(CNNRNN, self).__init__()

        self.feature_extractor = create_model(cnn, in_chans=in_chans)
        self.feature_extractor.fc = nn.Identity()
        self.seq_len = seq_len
        out_features = self.feature_extractor(torch.randn(1, 1, input_size, input_size)).shape[-1]
        self.rnn = RNN(rnn, out_features, seq_len)
        self.is_realtime = is_realtime
        self.queue = None

        if weights is not None:
            try:
                self.load_state_dict(torch.load(weights, map_location=device))
                print("CNN+RNN loaded")
            except:
                self.feature_extractor = create_model(cnn, in_chans=in_chans)
                self.feature_extractor.load_state_dict(torch.load(weights, map_location=device))
                self.feature_extractor.fc = nn.Identity()
                print("Only CNN loaded")
                # Freeze CNN
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.feature_extractor(x)
        if not self.is_realtime:
            return self.rnn(x)
        else:
            """
            inputs : 1 x 1 x H x W x C
            queue : (1 ~ L) x latent_dim
            incoming : 1 x latent_dim
            """
            self.queue = torch.cat([self.queue, x], dim=0)  # push
            if self.queue.shape[0] > self.seq_len:
                self.queue = self.queue[1:]  # pop left
                return self.rnn(self.queue)  # CNN+RNN
            else:
                return self.feature_extractor.fc(x)  # VanilaCNN


def get_model(seq_len, cnn, rnn=None, weights=None, is_realtime=False, contrast_learn=False, in_chans=1):
    """Returns model.

    Args:
        seq_len (int):  RNN sequence length in case of CNN+RNN model.
        cnn (str): Specify name of CNN model.
        rnn (str, optional): Specify name of RNN model if needed. Defaults to None.
        weights (str, optional): Path to model weights. Defaults to None.
        is_realtime (bool, optional): Runs realtime application if indicated. Defaults to False.
        contrast_learn (bool, optional): Use SupConLoss. Defaults to False.
        in_chans (int, optional): Number of input channels. Defaults to 1.

    Returns:
        nn.Module: Model
    """

    if rnn == None:
        if contrast_learn:
            # Contrastive Learning
            if weights is None:
                model = create_model(cnn, pretrained=True, in_chans=in_chans, num_classes=0)
            else:
                model = create_model(cnn, pretrained=True, in_chans=in_chans, num_classes=1)
                model.load_state_dict(torch.load(weights, map_location=device))
            model.name = cnn + "_SCL"
        else:
            # Traditional CNN model
            model = create_model(cnn, pretrained=True, in_chans=in_chans, num_classes=1)
            if weights is not None:
                model.load_state_dict(torch.load(weights, map_location=device))
            model.name = cnn

    else:
        # CNNRNN
        model = CNNRNN(cnn, rnn, seq_len, weights, in_chans, is_realtime)
        model.name = cnn + "_" + rnn

    return model
