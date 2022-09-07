import re


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
        self.data = self.tokenize(path)

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
                texts_to_ids.append(ids)

        return texts_to_ids
