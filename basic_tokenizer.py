import re

class Tokenizer:
    def __init__(self):
        self.vocabulary = {"<pad>": 0, "<unk>": 1}
        self.token_map = {v: k for k, v in self.vocabulary.items()}
        self.vocabulary_size = len(self.vocabulary)

    def tokenize(self, text):
        words = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        return [item.strip() for item in words if item.strip()]
    
    def __add_to_vocabulary(self, word):
        if word not in self.vocabulary:
            self.vocabulary[word] = self.vocabulary_size
            self.token_map[self.vocabulary_size] = word
            self.vocabulary_size += 1
    
    def __add_words_to_vocabulary(self, words):
        for word in words:
            self.__add_to_vocabulary(word)
    
    def encode(self, text):
        words = self.tokenize(text)
        self.__add_words_to_vocabulary(words)
        return [self.vocabulary[word] if word in self.vocabulary else self.vocabulary["<unk>"] for word in words]

    def decode(self, tokens):
        text = " ".join([self.token_map[token] for token in tokens])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        text = text.replace(" <unk> ", " <unk>")
        return text