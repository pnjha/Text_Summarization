from __future__ import unicode_literals, print_function, division

import os
import numpy as np
import pandas as pd
from numpy import array

import re
import time
import math
import random
import string
import unicodedata
from io import open
from tqdm import tqdm
from rouge import Rouge
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from nltk.translate import bleu
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

debug = True
to_print_epoch = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 200
root_directory = ""
MAX_SIZE = 10000
epoch = 250
print_every_epoch = 10
no_of_hidden_size = 256
lcl_learning_rate = 0.001
min_loss = 0.7
dropout = 0.1
n_layers = 1
teacher_forcing_ratio = 0.5
batch_size = 1

SOS_token = 0
EOS_token = 1
UNK_token = 2

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not","didn't": "did not",
"doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
"he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
"I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
"i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
"it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
"mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
"mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
"oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
"she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
"should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
"this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
"there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
"they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
"wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
"we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
"what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
"where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
"why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
"would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
"y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
"you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
"you're": "you are", "you've": "you have"}


def process_text(text,flag=False):
    stop_words = stopwords.words('english')
    text = text.lower()
    text = text.replace('\n','')
    text = re.sub(r'\(.*\)','',text)
    text = re.sub(r'[^a-zA-Z0-9. ]','',text)
    text = re.sub(r'\.',' . ',text)
    text = text.replace('.','')
    text = text.split()
    for i in range(len(text)):
        word = text[i]
        if word in contraction_mapping:
            text[i] = contraction_mapping[word]
    newtext = []
    for word in text:
        if word not in stop_words and len(word)>0:
            newtext.append(word)
    text = newtext
    if flag:
        text = text[::-1]
    text = " ".join(text)
    text = text.replace("'s",'') 
    return text

def prepareInput(sourcefilePath,destFilePath,fileName):
    source = sourcefilePath
    target = destFilePath
    save_trans = fileName

    corpus_source = open(source, 'r').readlines()
    corpus_target = open(target, 'r').readlines()
  
    writer = open(save_trans, 'w')
    num_lines = 0
    for k, v in zip(corpus_source, corpus_target):
        k = process_text(k,True)
        v = process_text(v,False)
        writer.write(k + '\t' + v + '\n')
        num_lines += 1
        if num_lines > MAX_SIZE:
            break
    writer.flush()
    writer.close()

class Lang:
    def __init__(self, name):
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

def parseLanguageInput(lang1,lang2,fullFilePath):

    lines = open(fullFilePath, encoding='utf-8').read().strip().split('\n')
    pairs = [[s for s in l.split('\t')] for l in lines]

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData(lang1,lang2,fullFilePath):
    input_lang, output_lang, pairs = None,None,None
    input_lang, output_lang, pairs = parseLanguageInput(lang1,lang2,fullFilePath)

    for pair in pairs:
      input_lang.addSentence(pair[0])
      output_lang.addSentence(pair[1])

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def indexesFromSentence(lang, sentence):

    return [lang.word2index[word] if word in lang.word2index else UNK_token for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, epochs, learning_rate):
    
    start_training = time.time()
    plot_losses = []
    no_of_epoch = []
    print_loss_total = 0 
    plot_loss_total = 0
    n_iters = len(pairs)
    print("Dataset Size: ",n_iters)

    encoder_optimizer = optim.ASGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.ASGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    epoch_loss_list = []
    
    for i in range(epochs):
        
        loss = 0
        start = time.time()
        training_pairs = [tensorsFromPair(pair) for pair in pairs]
        
        for iter in tqdm(range(1, n_iters + 1)):

            training_pair = training_pairs[iter - 1]
            input_tensor,target_tensor = training_pair[0],training_pair[1]

            loss += train(input_tensor, target_tensor, encoder,decoder, encoder_optimizer, decoder_optimizer, criterion)
        
        loss = loss/n_iters
        print('%s (%d %d%%) %.4f' % (timeSince(start, (i+1)/epochs), (i+1), (i+1)/epochs*100, loss))
        epoch_loss_list.append(loss)
        
        if loss < min_loss:
            break

    while len(epoch_loss_list) < epochs:
        epoch_loss_list.append(epoch_loss_list[-1])

    end_training = time.time()
    print("Total time taken for training: %.5f",asMinutes(start_training - end_training))    
    return epoch_loss_list,epochs    

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):

            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data

            topv, topi = decoder_output.data.topk(1)
            if topi.item() == UNK_token:
                decoded_words.append('<UNK>')
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def load_saved_encoder(lcl_input_lang,lcl_hidden_size,lcl_encoder_model_path):
    device = torch.device('cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_model = EncoderRNN(lcl_input_lang.n_words, lcl_hidden_size).to(device)
    encoder_model.load_state_dict(torch.load(lcl_encoder_model_path, map_location=device))
    return encoder_model

def load_saved_decoder(lcl_hidden_size,lcl_output_lang,lcl_decoder_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder_model = AttnDecoderRNN(lcl_hidden_size,lcl_output_lang.n_words,dropout,MAX_LENGTH).to(device)
    decoder_model.load_state_dict(torch.load(lcl_decoder_model_path, map_location=device))
    return decoder_model

def load_obj(obj_type,obj_name_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj_type = torch.load(obj_name_path, map_location=device)
    return obj_type

def calculate_rouge(rouge, pred_trg, real_trg):

    pred_trg = " ".join(pred_trg)
    real_trg = " ".join(real_trg[0])
    if len(pred_trg) > len(real_trg):
        diff = len(pred_trg) - len(real_trg)
        real_trg = real_trg +" "+  "#"*(diff-1)
    elif len(pred_trg) < len(real_trg):
        diff = len(real_trg) - len(pred_trg)
        pred_trg = pred_trg +" "+ "#"*(diff-1)
    scores = rouge.get_scores(pred_trg, real_trg)
    return scores 

def calculate_Result(encoder, decoder,lcl_pairs, n=50):
    result_value_rouge_score = []
    rouge = Rouge()
    
    for i in range(n):
        pair = random.choice(lcl_pairs)

        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)

        reference = [pair[1].split()]

        output_words = output_words[:-1]
        target_predicted = output_words
        
        score = calculate_rouge(rouge,target_predicted,reference)
        result_value_rouge_score.append((pair[0],pair[1].split(),target_predicted,score))

    return result_value_rouge_score


prefix = "train"
sourceLangPath = root_directory +prefix+".original"
sourcePrefix = "original"
targetLangPath = root_directory+prefix+".compressed"
targetPrefix = "compressed"
fullFilePathForData = prefix + "_" + sourcePrefix + "_" + targetPrefix + ".txt"


# create Custom File having both Langauges  
#isFilePresent = os.path.isfile(fullFilePathForData)
#if isFilePresent == False:
#    print("New File Created")
prepareInput(sourceLangPath,targetLangPath,fullFilePathForData)
#else:
 # print("File is not created Again")

input_lang, output_lang, pairs = prepareData(sourcePrefix, targetPrefix,fullFilePathForData)
print(random.choice(pairs))

print(pairs[0])
print(pairs[1])
print(pairs[2])
  
encoder_model_path = "Encoder_Model.pt"
decoder_model_path = "Decoder_Model.pt"
params = "params.pt"
vocab_params = "vocab.pt"

train_result_data_path = "train_Result_Epoch_"+ str(epoch) +"_Hid_size_"+ str(no_of_hidden_size) +".pt"
test_result_data_path = "test_Result_Epoch_"+ str(epoch) +"_Hid_size_"+ str(no_of_hidden_size) + ".pt"   
encoder_model_path = "Ep_"+ str(epoch) +"_Hd_"+ str(no_of_hidden_size) + "_lr_"+ str(lcl_learning_rate) +"_"+ encoder_model_path   
decoder_model_path = "Ep_"+ str(epoch) +"_Hd_"+ str(no_of_hidden_size) + "_lr_"+ str(lcl_learning_rate) +"_"+ decoder_model_path 
vocab_params = "Ep_"+ str(epoch) +"_Hd_"+ str(no_of_hidden_size) + "_lr_"+ str(lcl_learning_rate) +"_"+ vocab_params 

print("Encoder Model Path :" ,encoder_model_path)
print("Decoder Model Path :" ,decoder_model_path)
print("params Path :" ,params)
print("train file Path :",train_result_data_path)
print("test file Path :",test_result_data_path)
print("Encoder Model Exist " ,os.path.isfile(encoder_model_path))
print("Decoder Model Exist " ,os.path.isfile(decoder_model_path))

encoder_model = None
decoder_model = None
model_performance = {}
vocab = None

if os.path.isfile(encoder_model_path) == False or os.path.isfile(decoder_model_path) == False:

     hidden_size = no_of_hidden_size
     encoder_model = EncoderRNN(input_lang.n_words, hidden_size).to(device)
     decoder_model = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=dropout).to(device)
     plot_losses,no_of_epoch = trainIters(encoder_model, decoder_model, epoch ,learning_rate=lcl_learning_rate)
     folder_path = encoder_model_path
     torch.save(encoder_model.state_dict(), encoder_model_path)
     folder_path = decoder_model_path
     torch.save(decoder_model.state_dict(), decoder_model_path)

     lcl_params = {'loss_list':plot_losses,'epoch_list':no_of_epoch,'lr':lcl_learning_rate,"dropout":dropout, "MAX_LENGTH":MAX_LENGTH,"epoch":epoch, "no_of_hidden_size":no_of_hidden_size}
     lcl_params["lcl_learning_rate"] = lcl_learning_rate

     vocab = {'input_lang':input_lang, 'output_lang':output_lang}
     model_performance = lcl_params
     torch.save(lcl_params, params)
     torch.save(vocab, vocab_params)
     plt.plot(plot_losses)
     #plt.show()
     plt.savefig('loss_vs_ep_{}_lr_{}_hs_{}_dslen_{}.png'.format(epoch,lcl_learning_rate,no_of_hidden_size,MAX_SIZE))
else:
     print("Model Already Exist")
     hidden_size = no_of_hidden_size
     encoder_model = load_saved_encoder(input_lang,hidden_size,encoder_model_path)
     decoder_model = load_saved_decoder(hidden_size,output_lang,decoder_model_path)
     model_performance = load_obj(model_performance,params)
     vocab = load_obj(vocab,vocab_params)

print(model_performance)
print(vocab)
print(encoder_model)
print(decoder_model)

#plt.plot(plot_losses)
#plt.show()
#plt.savefig('loss_vs_ep_{}_lr_{}_hs_{}_dslen_{}.png'.
#    format(epoch,lcl_learning_rate,no_of_hidden_size,MAX_SIZE))

result_value_rouge_score = calculate_Result(encoder_model, decoder_model,pairs)
result_value_rouge_score_dict = {}
result_value_rouge_score_dict['result'] = result_value_rouge_score 
torch.save(result_value_rouge_score_dict, train_result_data_path)

limit = 10

for item in result_value_rouge_score:
  print(" Source Language ",item[0])
  print(" Input Target",item[1])
  print(" Output Target",item[2])
  print(" Score ",item[3])
  limit -= 1

prefix = "test"
sourceLangPath = root_directory +prefix+".original"
sourcePrefix = "original"
targetLangPath = root_directory+prefix+".compressed"
targetPrefix = "compressed"
fullFilePathForData = prefix + "_" + sourcePrefix + "_" + targetPrefix + ".txt"


#isFilePresent = os.path.isfile(fullFilePathForData)
#if isFilePresent == False:
prepareInput(sourceLangPath,targetLangPath,fullFilePathForData)
test_input_lang, test_output_lang, test_pairs = prepareData(sourcePrefix, targetPrefix,fullFilePathForData)
print(random.choice(test_pairs))

result_value_rouge_score_test = calculate_Result(encoder_model, decoder_model,test_pairs)
result_value_rouge_score_dict_test = {}
result_value_rouge_score_dict_test['result'] = result_value_rouge_score_test 
torch.save(result_value_rouge_score_dict_test, test_result_data_path)

limit = 10

for item in result_value_rouge_score_test:
    print(" Source Language ",item[0])
    print(" Input Target",item[1])
    print(" Output Target",item[2])
    print(" Score ",item[3])
    limit -= 1
