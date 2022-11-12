import os
import re

import compress_fasttext
import numpy as np
import torch


class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, "train.txt"))
        self.valid = self.tokenize(os.path.join(path, "valid.txt"))

    def tokenize(self, path):
        """Tokenizes a text file."""
        TOKEN_RE = re.compile(r"[а-яА-Я]+")

        with open(path, "r", encoding="utf8") as file:
            texts_to_ids = []
            for line in file:
                line = line.lower()
                words = TOKEN_RE.findall(line)
                ids = []
                for word in words:
                    # Add word to the dictionary
                    self.dictionary.add_word(word)
                    # Convert word to corresponding idx
                    ids.append(self.dictionary.word2idx[word])
                texts_to_ids.append(torch.tensor(ids, dtype=torch.int64))

        return torch.cat(texts_to_ids)


class WordEmbedder:
    """Encodes words in dictionary into a sequence of word embeddings.
    Inputs:
        - List of unique words.
    Outputs:
        - List of embeddings.
    """

    def __init__(self):
        self._embedder = compress_fasttext.models.CompressedFastTextKeyedVectors.load(
            "https://github.com/avidale/compress-fasttext/releases/download/gensim-4-draft/geowac_tokens_sg_300_5_2020-100K-20K-100.bin"
        )

    @property
    def dim(self):
        """Embedding dimension."""
        return self._embedder.vector_size

    def __call__(self, dictionary):
        embeddings = [self._embedder[word] for word in dictionary]
        return torch.tensor(np.array(embeddings), dtype=torch.float32)
