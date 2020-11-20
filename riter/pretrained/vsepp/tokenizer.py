import os
import pickle
import nltk
from collections import OrderedDict

from .encoders import VSEPP_PATH
from riter import utils


TOKENIZER_MAP = OrderedDict([("vsepp-gru-coco", "vsepp_tokenizer_coco.pkl")])


class VseppTokenizer:
    """Tokenizer for GRU model that encodes caption to joint embedding
    space."""

    PAD_TOKEN = "<pad>"
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    UNK_TOKEN = "<unk>"

    def __init__(self):
        self._word2idx = {}
        self._idx2word = []
        self.add_word(VseppTokenizer.PAD_TOKEN)
        self.add_word(VseppTokenizer.START_TOKEN)
        self.add_word(VseppTokenizer.END_TOKEN)
        self.add_word(VseppTokenizer.UNK_TOKEN)

    @property
    def pad_token(self):
        return VseppTokenizer.PAD_TOKEN

    @property
    def pad_token_id(self):
        return self._word2idx[self.pad_token]

    @property
    def unk_token(self):
        return VseppTokenizer.UNK_TOKEN

    @property
    def unk_token_id(self):
        return self._word2idx[self.unk_token]

    @property
    def start_token(self):
        return VseppTokenizer.START_TOKEN

    @property
    def start_token_id(self):
        return self._word2idx[self.start_token]

    @property
    def end_token(self):
        return VseppTokenizer.END_TOKEN

    @property
    def end_token_id(self):
        return self._word2idx[self.end_token]

    def add_word(self, word):
        """Add a word to the dictionary."""
        if word not in self._word2idx:
            self._idx2word.append(word)
            self._word2idx[word] = len(self._idx2word) - 1
        return self._word2idx[word]

    def word2idx(self, word):
        """Given a word, return its corresponding id."""
        if word not in self._word2idx:
            return self.unk_token_id
        else:
            return self._word2idx[word]

    def idx2word(self, idx):
        """Given an id, return its corresponding word."""
        if idx < 0 or idx > len(self._idx2word):
            raise ValueError(f"Out-of-bounds index {idx} for tokenizer.")
        return self._idx2word[idx]

    def __len__(self):
        return len(self._idx2word)

    def __call__(self, text):
        """Given a string, tokenize it and return the ids.

        Args:
            text (str): input string
        Returns:
            ids (list[int]): List of ids
        """
        words = nltk.tokenize.word_tokenize(text.lower())
        words = [self.start_token] + words + [self.end_token]
        ids = [self.word2idx(w) for w in words]
        return ids

    def save(self, path):
        if os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            corpus = pickle.load(f)
        ccorpus = VseppTokenizer()
        ccorpus._idx2word = corpus._idx2word
        ccorpus._word2idx = corpus._word2idx
        ccorpus.save("vsepp_tokenizer_coco.pkl")
        return ccorpus

    @classmethod
    def from_pretrained(cls, pretrained_model_name):
        if pretrained_model_name not in TOKENIZER_MAP:
            raise ValueError(
                f"`{pretrained_name}` is not available. "
                f"Available options are: {list(TOKENIZER_MAP.keys())}"
            )
        path = utils.download_if_needed(
            f"{VSEPP_PATH}/{TOKENIZER_MAP[pretrained_model_name]}"
        )
        return cls.load(path)
