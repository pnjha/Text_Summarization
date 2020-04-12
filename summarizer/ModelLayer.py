from __future__ import unicode_literals, print_function, division

import os
import time
import math
import numpy as np 
import pandas as pd

import re
import string
import random
import unicodedata
from io import open
from rouge import Rouge

from nltk.translate import bleu
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import Lang
from Lang import Lang

debug = True
to_print_epoch = True

SOS_token = 0
EOS_token = 1
UNK_token = 2

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

    def __init__(self, hidden_size, output_size, dropout_p, max_length):
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

class ModelLayer:

	def __init__(self):
		self.MAX_LENGTH = 1000
		self.root_directory = "."
		self.epoch = 500
		self.print_every_epoch = (self.epoch / 1000)
		self.no_of_hidden_size = 1024
		self.lcl_learning_rate = 0.00001
		self.dropout = 0.1
		self.input_lang = None
		self.output_lang = None
		self.encoder_model = None
		self.decoder_model = None
		self.vocab = {}
		self.load_models_params()
		self.contraction_mapping = {
			"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not","didn't": "did not",
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

	def process_text(self,text,flag=False):
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
	        if word in self.contraction_mapping:
	            text[i] = self.contraction_mapping[word]
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

	def indexesFromSentence(self, lang, sentence):

		return [lang.word2index[word] if word in lang.word2index else UNK_token for word in sentence.split(' ')]

	def tensorFromSentence(self, lang, sentence):
	    indexes = self.indexesFromSentence(lang, sentence)
	    indexes.append(EOS_token)
	    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

	def showPlot(self, points):
		plt.plot(points)
		plt.savefig("plot.png")
		plt.show()

	def evaluate(self,encoder, decoder, sentence, max_length, input_lang, output_lang):

	    with torch.no_grad():
	        input_tensor = self.tensorFromSentence(input_lang, sentence)
	        input_length = input_tensor.size()[0]
	        encoder_hidden = encoder.initHidden()

	        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

	        for ei in range(input_length):
	            encoder_output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
	            encoder_outputs[ei] += encoder_output[0, 0]

	        decoder_input = torch.tensor([[SOS_token]], device=device)
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

	def load_saved_encoder(self,lcl_input_lang,lcl_hidden_size,lcl_encoder_model_path):
	  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	  encoder_model = EncoderRNN(lcl_input_lang.n_words, lcl_hidden_size).to(device)
	  encoder_model.load_state_dict(torch.load(lcl_encoder_model_path, map_location=device))
	  return encoder_model

	def load_saved_decoder(self,lcl_hidden_size,lcl_output_lang,lcl_decoder_model_path):
	  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	  decoder_model = AttnDecoderRNN(lcl_hidden_size,lcl_output_lang.n_words,self.dropout,self.MAX_LENGTH).to(device)
	  decoder_model.load_state_dict(torch.load(lcl_decoder_model_path, map_location=device))
	  return decoder_model

	def load_obj(self,obj_name_path):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		obj_type = torch.load(obj_name_path, map_location=device)
		return obj_type

	def get_summary(self,encoder, decoder,input_text,input_lang,output_lang):
		output_words, attentions = self.evaluate(encoder, decoder, input_text,self.MAX_LENGTH, input_lang, output_lang)
		output_words = output_words[:-1]
		output_sentence = ' '.join(output_words)
		return output_sentence

	def load_models_params(self):

		encoder_model_path = "Encoder_Model.pt"
		decoder_model_path = "Decoder_Model.pt"
		params = "params.pt"
		vocab_params = "vocab.pt"
		
		model_performance = {}
		
		# print("Parameters File Exist" ,os.path.isfile(params))
		if os.path.isfile(params) == True:
			model_performance = self.load_obj(params)
		else:
			print("parameters not found")

		self.MAX_LENGTH = model_performance["MAX_LENGTH"]
		self.epoch = model_performance["epoch"]
		self.no_of_hidden_size = model_performance["no_of_hidden_size"]
		self.lcl_learning_rate = model_performance["lr"]
		self.dropout = model_performance["dropout"]

		encoder_model_path = "Ep_"+ str(self.epoch) +"_Hd_"+ str(self.no_of_hidden_size) + "_lr_"+ str(self.lcl_learning_rate) +"_"+ encoder_model_path   
		decoder_model_path = "Ep_"+ str(self.epoch) +"_Hd_"+ str(self.no_of_hidden_size) + "_lr_"+ str(self.lcl_learning_rate) +"_"+ decoder_model_path 
		vocab_params = "Ep_"+ str(self.epoch) +"_Hd_"+ str(self.no_of_hidden_size) + "_lr_"+ str(self.lcl_learning_rate) +"_"+ vocab_params 
		
		# print("Encoder Model Path :" ,encoder_model_path)
		# print("Decoder Model Path :" ,decoder_model_path)
		# print("Params Path :" ,params)
		# print("Vocab params Path :" ,vocab_params)
		# print("Encoder Model Exist " ,os.path.isfile(encoder_model_path))
		# print("Decoder Model Exist " ,os.path.isfile(decoder_model_path))
		# print("Vocab File Exist" ,os.path.isfile(vocab_params))


		if os.path.isfile(encoder_model_path) == True and os.path.isfile(decoder_model_path) == True:
			
			hidden_size = self.no_of_hidden_size
			self.vocab = self.load_obj(vocab_params)
			self.input_lang = self.vocab["input_lang"]
			self.output_lang = self.vocab["output_lang"]
			self.encoder_model = self.load_saved_encoder(self.input_lang,hidden_size,encoder_model_path)
			self.decoder_model = self.load_saved_decoder(hidden_size,self.output_lang,decoder_model_path)
			self.vocab = None

		# print(model_performance)
		# print(encoder_model)
		# print(decoder_model)
		# print(input_lang)
		# print(output_lang)

	def perform_summarization(self,input_text):

		input_text = self.process_text(input_text)
	
		summary = self.get_summary(self.encoder_model, self.decoder_model,input_text,self.input_lang,self.output_lang)

		data = {}
		data["summary"] = summary

		return data
		