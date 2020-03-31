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

	def indexesFromSentence(self, lang, sentence):

		return [lang.word2index[word] if word in lang.word2index else UNK_token for word in sentence.split(' ')]

	def tensorFromSentence(self, lang, sentence):
	    indexes = self.indexesFromSentence(lang, sentence)
	    indexes.append(EOS_token)
	    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

	def showPlot(self, points):
		plt.figure()
		fig, ax = plt.subplots()
		loc = ticker.MultipleLocator(base=0.2)
		ax.yaxis.set_major_locator(loc)
		plt.plot(points)

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

	def perform_summarization(self,input_text):

		encoder_model_path = "Encoder_Model.pt"
		decoder_model_path = "Decoder_Model.pt"
		params = "params.pt"
		vocab_params = "vocab.pt"

		vocab = {}
		encoder_model = None
		decoder_model = None
		model_performance = {}
		input_lang,output_lang = None, None

		print("Parameters File Exist" ,os.path.isfile(params))
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
		
		print("Encoder Model Path :" ,encoder_model_path)
		print("Decoder Model Path :" ,decoder_model_path)
		print("Params Path :" ,params)
		print("Vocab params Path :" ,vocab_params)
		print("Encoder Model Exist " ,os.path.isfile(encoder_model_path))
		print("Decoder Model Exist " ,os.path.isfile(decoder_model_path))
		print("Vocab File Exist" ,os.path.isfile(vocab_params))


		if os.path.isfile(encoder_model_path) == True and os.path.isfile(decoder_model_path) == True:
			
			hidden_size = self.no_of_hidden_size
			vocab = self.load_obj(vocab_params)
			input_lang = vocab["input_lang"]
			output_lang = vocab["output_lang"]
			encoder_model = self.load_saved_encoder(input_lang,hidden_size,encoder_model_path)
			decoder_model = self.load_saved_decoder(hidden_size,output_lang,decoder_model_path)
			vocab = None

		print(model_performance)
		print(encoder_model)
		print(decoder_model)
		print(input_lang)
		print(output_lang)

		summary = self.get_summary(encoder_model, decoder_model,input_text,input_lang,output_lang)

		data = {}
		data["summary"] = summary

		return data
		