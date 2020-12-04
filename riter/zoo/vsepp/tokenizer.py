import os
import pickle
import nltk
import torch
from collections import OrderedDict

from .encoders import VSEPP_PATH
from riter import utils


TOKENIZER_MAP = OrderedDict([
    ("vsepp-gru-resnet-coco", "vsepp_tokenizer_coco.pkl"),
    ("vsepp-gru-vgg-coco", "vsepp_tokenizer_coco.pkl"),
    ("vsepp-gru-resnet-flickr", "vsepp_tokenizer_flickr.pkl"),
    ("vsepp-gru-vgg-flickr", "vsepp_tokenizer_flickr.pkl"),
])


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
    def pad_token(self) -> str:
        return VseppTokenizer.PAD_TOKEN

    @property
    def pad_token_id(self) -> int:
        return self._word2idx[self.pad_token]

    @property
    def unk_token(self) -> str:
        return VseppTokenizer.UNK_TOKEN

    @property
    def unk_token_id(self) -> int:
        return self._word2idx[self.unk_token]

    @property
    def start_token(self) -> str:
        return VseppTokenizer.START_TOKEN

    @property
    def start_token_id(self) -> int:
        return self._word2idx[self.start_token]

    @property
    def end_token(self) -> str:
        return VseppTokenizer.END_TOKEN

    @property
    def end_token_id(self) -> int:
        return self._word2idx[self.end_token]

    def add_word(self, word: str):
        """Add a word to the dictionary."""
        if word not in self._word2idx:
            self._idx2word.append(word)
            self._word2idx[word] = len(self._idx2word) - 1
        return self._word2idx[word]

    def word2idx(self, word: str) -> int:
        """Given a word, return its corresponding id.
        If word cannot be found in the vocabulary, return <unk> token id.
        """
        if word not in self._word2idx:
            return self.unk_token_id
        else:
            return self._word2idx[word]

    def idx2word(self, idx: int) -> str:
        """Given an id, return its corresponding word."""
        if idx < 0 or idx > len(self._idx2word):
            raise ValueError(f"Out-of-bounds index {idx} for tokenizer.")
        return self._idx2word[idx]

    def __len__(self) -> int:
        return len(self._idx2word)

    def __call__(self, texts, return_tensor=True, padding=True, max_seq_len=-1):
        """Given a string, tokenize it and return the ids.

        Args:
            text (Union[str, list[str]]): input string
        Returns:
            ids (list[int]): List of ids
        """
        if isinstance(texts, str):
            return self.__call__(
                [texts],
                return_tensor=return_tensor,
                padding=padding,
                max_seq_len=max_seq_len,
            )
        elif isinstance(texts, list):
            ids_list = []
            for text in texts:
                words = nltk.tokenize.word_tokenize(text.lower())
                if max_seq_len > 0:
                    words = words[:max_seq_len]
                words = [self.start_token] + words + [self.end_token]
                ids = [self.word2idx(w) for w in words]
                ids_list.append(ids)
            ids_list.sort(key=lambda x: len(x), reverse=True)
            lengths = [len(i) for i in ids_list]
            if padding:
                max_len = max(lengths)
                for i in range(len(ids_list)):
                    num_pad_tokens = max_len - len(ids_list[i])
                    ids_list[i] += [self.pad_token_id] * num_pad_tokens

            if return_tensor:
                return {
                    "input_ids": torch.tensor(ids_list),
                    "lengths": torch.tensor(lengths, dtype=torch.int64),
                }
            else:
                return {"input_ids": ids_list, "lengths": lengths}

    def save(self, path: str):
        """Save tokenizer to path"""
        if os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            tokenizer = pickle.load(f)
        return tokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name: str):
        if pretrained_model_name not in TOKENIZER_MAP:
            raise ValueError(
                f"`{pretrained_name}` is not available. "
                f"Available options are: {list(TOKENIZER_MAP.keys())}"
            )
        path = utils.download_if_needed(
            f"{VSEPP_PATH}/{TOKENIZER_MAP[pretrained_model_name]}"
        )
        return cls.load(path)
