import re
import os
import nltk
import json
# # import torch
from tqdm import tqdm
# from gensim.models import KeyedVectors
# filename = os.getcwd() + '/word2vec/GoogleNews-vectors-negative300.bin'
# model = KeyedVectors.load_word2vec_format(filename, binary=True)
# model.wv["hotel"]
# print(type(model))
# print(type(model.wv))
# print(type(model.wv["hotelasdasfasfasfasf"]))
# idx = 0

# for idx, key in enumerate(model.wv.vocab):
# 	print(key, idx)

# eval_data = os.getcwd() + '/data/eval_data.csv'
train_original = os.getcwd() + '/data/train.original'
train_compressed = os.getcwd() + '/data/train.compressed'
test_original = os.getcwd() + '/data/test.original'
test_compressed = os.getcwd() + '/data/test.compressed'

# eval_data = open(eval_data, 'r').readlines()
# train_original = open(train_original, 'r').readlines()
# train_compressed = open(train_compressed, 'r').readlines()
# test_original = open(test_original, 'r').readlines()
# test_compressed = open(test_compressed, 'r').readlines()

# f = []
# l = []

# def test(train_original):
# 	for i in tqdm(range(len(train_original))):
# 		line = train_original[i].split()
# 		for item in line :
# 			if item.strip() in model.wv:
# 				f.append(item.strip())
# 			else:
# 				l.append(item.strip())



from gensim.models.fasttext import FastText
from gensim.test.utils import get_tmpfile
import csv

def get_file_content(filename):

	file_content = []
	f = open(filename, "r")
	for x in f:
		file_content.append(x.strip())
	return file_content

def process_text(text,contraction_mapping):
    # stop_words = stopwords.words('english')
    text = text.lower()
    text = text.replace('\n','')
    text = re.sub(r'\(.*\)','',text)
    text = re.sub(r'[^a-zA-Z0-9. ]','',text)
    text = re.sub(r'\.',' . ',text)
    text = text.replace('.','')
    text = text.split()
    new_list = ['SOS']
    for i in range(len(text)):
        if text[i] in contraction_mapping:
            text[i] = contraction_mapping[text[i]]
        new_list.append(text[i])
    new_list.append('EOS')
    return new_list

with open(os.getcwd() + '/data/contraction_mapping.json') as f:
	contraction_mapping = json.load(f)

# eval_sentences = get_file_content(eval_data)
train_org_sentences = get_file_content(train_original)
train_com_sentences = get_file_content(train_compressed)
test_org_sentences = get_file_content(test_original)
test_com_sentences = get_file_content(test_compressed)

# wpt = nltk.WordPunctTokenizer()

# eval_corpus = [wpt.tokenize(document) for document in eval_sentences]
train_org_corpus = [process_text(document,contraction_mapping) for document in train_org_sentences]
train_com_corpus = [process_text(document,contraction_mapping) for document in train_com_sentences]
test_org_corpus = [process_text(document,contraction_mapping) for document in test_org_sentences]
test_com_corpus = [process_text(document,contraction_mapping) for document in test_com_sentences]

print(test_com_corpus)
vocab = ['SOS','EOS','UNK']

def get_vocab(corpus):
	for i in tqdm(range(len(corpus))):
		for j in range(len(corpus[i])):
			vocab.append(corpus[i][j])

get_vocab(train_org_corpus)
get_vocab(train_com_corpus)
get_vocab(test_org_corpus)
get_vocab(test_com_corpus)

vocab = list(set(vocab))
print(vocab)
print(len(vocab))


with open(os.getcwd()+"/data/vocab.txt", "w") as f:
    for s in vocab:
        f.write(str(s) +"\n")

vocab = []
with open(os.getcwd()+"/data/vocab.txt", "r") as f:
  for line in f:
    vocab.append(line.strip())
print(len(vocab))

# print(train_org_corpus)

fname = get_tmpfile(os.getcwd()+"/pretrained_layer/fasttext.model")

model = FastText(size=100, min_count=2)
model.build_vocab(sentences=train_org_corpus)
model.train(sentences=train_org_corpus, total_examples=len(train_org_corpus), epochs=2)

model.save(fname)
model = FastText.load(fname)

model.build_vocab(train_com_corpus, update=True)
model.train(train_com_corpus, total_examples=len(train_com_corpus), epochs=2)

model.save(fname)
model = FastText.load(fname)

model.build_vocab(test_org_corpus, update=True)
model.train(test_org_corpus, total_examples=len(test_org_corpus), epochs=2)

model.save(fname)
model = FastText.load(fname)

model.build_vocab(test_com_corpus, update=True)
model.train(test_com_corpus, total_examples=len(test_com_corpus), epochs=2)

model.save(fname)
model = FastText.load(fname)

print(model.wv)
print(model.wv.vocab)

if "corona" not in model.wv.vocab:
	print("Not present")

print(model.wv["corona"])
print(model.wv.most_similar(["corona"]))       

# feature_size = 20    # Word vector dimensionality  
# window_context = 50          # Context window size                                                                                    
# min_word_count = 5   # Minimum word count                        
# sample = 1e-3   # Downsample setting for frequent words

# # sg decides whether to use the skip-gram model (1) or CBOW (0)
# ft_model = FastText(tokenized_corpus, size=feature_size, window=window_context,min_count=min_word_count,sample=sample, sg=1, iter=1)
                    
# print(ft_model.wv["hotel"])
# print(ft_model.wv.most_similar(["hotel"]))       
# # view similar words based on gensim's FastText model
# # similar_words = {search_term: [item[0] for item in ft_model.wv.most_similar([search_term], topn=5)]
# #                   for search_term in ['god', 'jesus', 'noah', 'egypt', 'john', 'gospel', 'moses','famine']}
# # similar_words                    