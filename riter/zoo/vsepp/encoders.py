# Code is from https://github.com/fartashf/vsepp/blob/master/model.py

import numpy as np
import torch
import torchvision
from collections import OrderedDict

from riter import utils

VSEPP_PATH = "vsepp"

VSEPP_CONFIGS = OrderedDict(
    [
        (
            "vsepp-resnet-coco",
            {
                "embed_size": 1024,
                "finetune": False,
                "cnn_type": "resnet152",
                "use_abs": False,
                "no_imgnorm": False,
            },
        ),
        (
            "vsepp-vgg-coco",
            {
                "embed_size": 1024,
                "finetune": False,
                "cnn_type": "vgg19",
                "use_abs": False,
                "no_imgnorm": False,
            },
        ),
        (
            "vsepp-gru-coco",
            {
                "vocab_size": 11755,
                "word_dim": 300,
                "embed_size": 1024,
                "num_layers": 1,
                "use_abs": False,
            },
        ),
    ]
)


class VseppImageEncoder(torch.nn.Module):
    def __init__(
        self,
        embed_size,
        finetune=False,
        cnn_type="resnet152",
        use_abs=False,
        no_imgnorm=False,
    ):
        """Load pretrained ResNet152 and replace top fc layer."""
        super(VseppImageEncoder, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        # Load a pre-trained model
        self.cnn = torchvision.models.__dict__[cnn_type](pretrained=True)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith("vgg"):
            self.fc = torch.nn.Linear(self.cnn["6"].in_features, embed_size)
            self.cnn.classifier = torch.nn.Sequential(
                *list(self.cnn.classifier.children())[:-1]
            )
        elif cnn_type.startswith("resnet"):
            self.fc = torch.nn.Linear(self.cnn.fc.in_features, embed_size)
            self.cnn.fc = torch.nn.Sequential()

        self._init_weights()

    def load_state_dict(self, state_dict):
        """Handle the models saved before commit pytorch/vision@989d52a."""
        if "cnn.classifier.1.weight" in state_dict:
            state_dict["cnn.classifier.0.weight"] = state_dict[
                "cnn.classifier.1.weight"
            ]
            del state_dict["cnn.classifier.1.weight"]
            state_dict["cnn.classifier.0.bias"] = state_dict["cnn.classifier.1.bias"]
            del state_dict["cnn.classifier.1.bias"]
            state_dict["cnn.classifier.3.weight"] = state_dict[
                "cnn.classifier.4.weight"
            ]
            del state_dict["cnn.classifier.4.weight"]
            state_dict["cnn.classifier.3.bias"] = state_dict["cnn.classifier.4.bias"]
            del state_dict["cnn.classifier.4.bias"]

        super(VseppImageEncoder, self).load_state_dict(state_dict)

    def _init_weights(self):
        """Xavier initialization for the fully connected layer."""
        r = np.sqrt(6.0) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)

        # normalization in the image embedding space
        features = l2norm(features)

        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

    @classmethod
    def from_pretrained(cls, pretrained_model_name):
        if pretrained_model_name not in VSEPP_CONFIGS:
            raise ValueError(
                f"`{pretrained_name}` is not available. "
                f"Available options are: {list(VSEPP_CONFIGS.keys())}"
            )
        config = VSEPP_CONFIGS[pretrained_model_name]
        model = cls(**config)
        path = utils.download_if_needed(
            f"{VSEPP_PATH}/{pretrained_model_name.replace('-', '_')}.pt"
        )
        model.load_state_dict(torch.load(path))
        return model


class VseppTextEncoder(torch.nn.Module):
    def __init__(self, vocab_size, word_dim, embed_size, num_layers, use_abs=False):
        super(VseppTextEncoder, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        # word embedding
        self.embed = torch.nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = torch.nn.GRU(word_dim, embed_size, num_layers, batch_first=True)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input_ids, lengths):
        """Handles variable size captions."""
        device = input_ids.device
        # Embed word ids to vectors
        input_ids = self.embed(input_ids)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            input_ids, lengths.cpu(), batch_first=True
        )

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        I = lengths.view(-1, 1, 1)
        I = (I.expand(input_ids.size(0), 1, self.embed_size) - 1).to(device)
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out

    @classmethod
    def from_pretrained(cls, pretrained_model_name):
        if pretrained_model_name not in VSEPP_CONFIGS:
            raise ValueError(
                f"`{pretrained_name}` is not available. "
                f"Available options are: {list(VSEPP_CONFIGS.keys())}"
            )
        config = VSEPP_CONFIGS[pretrained_model_name]
        model = cls(**config)
        path = utils.download_if_needed(
            f"{VSEPP_PATH}/{pretrained_model_name.replace('-', '_')}.pt"
        )
        model.load_state_dict(torch.load(path))
        return model


def l2norm(X):
    """L2-normalize columns of X."""
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X
