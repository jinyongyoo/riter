import os
import pickle
import nltk


class Tokenizer:
    """Tokenizer for GRU model that encodes caption to joint embedding
    space."""

    PAD_TOKEN = "<pad>"
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    UNK_TOKEN = "<unk>"

    def __init__(self):
        self._word2idx = {}
        self._idx2word = []
        self.add_word(Tokenizer.PAD_TOKEN)
        self.add_word(Tokenizer.START_TOKEN)
        self.add_word(Tokenizer.END_TOKEN)
        self.add_word(Tokenizer.UNK_TOKEN)

    def add_word(self, word):
        """Add a word to the dictionary."""
        if word not in self._word2idx:
            self._idx2word.append(word)
            self._word2idx[word] = len(self._idx2word) - 1
        return self._word2idx[word]

    def word2idx(self, word):
        """Given a word, return its corresponding id."""
        if word not in self._word2idx:
            return self._word2idx[Tokenizer.UNK_TOKEN]
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
        words = [Tokenizer.START_TOKEN] + words + [Tokenizer.END_TOKEN]
        ids = [self.word2idx(w) for w in words]
        return ids

    def save(self, path):
        if os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            corpus = pickle.load(f)
        return corpus
