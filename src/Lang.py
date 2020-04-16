from packages import *

class Lang:
    def __init__(self, name, SOS_token, EOS_token, UNK_token):
        self.name = name  
        self.word2index,self.word2count,self.index2word = {},{},{}
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS", UNK_token:"UNK"}
        self.n_words = 3  # Count SOS and EOS and UNK

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1