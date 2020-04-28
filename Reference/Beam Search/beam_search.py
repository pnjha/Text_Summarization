import numpy as np
import torch
import math
from random import random
from constants import BOS,EOS

"""
For the implementation of beam search we maintain a list of length k that looks as follows:
[
	[[list 1],score 1],
	[[list 2],score 2],
	...
]
"""
def next_logit(decoder,idx,tgt_seq,tgt_pos,src_seq,enc_output):
	"""
	Function that gives next logit
	"""
	dec_output, *rest = decoder(tgt_seq,tgt_pos,src_seq,enc_output)
	return [[i,float(dec_output[idx][i])] for i in range(len(dec_output[idx]))]

def next_logit_exp(idx,dims=7):
	"""
	Function used for demonstration
	"""
	return [[i,round(random(),2)] for i in range(dims)]

def maintain_k(candidate_list,new_score_list,beam_size):
	"""
	Maintains top k candidates after observing new logits
	"""
	assert len(new_score_list) == beam_size
	assert len(candidate_list) == beam_size
	for i in range(len(new_score_list)):
		if candidate_list[-1][-1] < new_score_list[i][-1]:
			for j in range(len(candidate_list)):
				if candidate_list[j][-1] < new_score_list[i][-1]:
					break
			candidate_list.insert(j,new_score_list[i])
			candidate_list = candidate_list[:-1]
			assert len(candidate_list) == beam_size
	return candidate_list


def beam_decode_one_line_exp(beam_size, max_len):
	"""
	Code for demo

	Inputs
	
	beam_size: size of beam
	max_len: maximum output length
	
	Outputs

	result_seq: beam decoded sequence
	"""
	print("score_list")
	score_list = [[[BOS],0] for i in range(beam_size)]
	print(score_list)
	for idx in range(max_len):
		print(idx," start")
		for candidate_idx in range(beam_size):
			print("\t",candidate_idx," start")
			tgt_seq = score_list[candidate_idx][0]
			print("tgt_seq")
			print(tgt_seq)
			tgt_pos = list(range(1, idx + 2))
			print("tgt_pos")
			print(tgt_pos)
			logit_list = next_logit_exp(idx)
			logit_list = sorted(logit_list,key = lambda word:word[-1],reverse = True)
			print("logit_list")
			print(logit_list)
			new_score_list = logit_list[:beam_size]
			new_score_list = [[score_list[candidate_idx][0]+[new_score_list[i][0]],round(
				score_list[candidate_idx][-1] + math.log(new_score_list[i][-1]),2)] for i in range(beam_size)]
			print("new_score_list")
			print(new_score_list)
			if candidate_idx == 0:
				candidate_list = new_score_list
			else:
				candidate_list = maintain_k(candidate_list,new_score_list,beam_size)
			print("candidate_list")
			print(candidate_list)
		score_list = candidate_list
		print("score_list")
		print(score_list)
	assert len(score_list) == beam_size
	score_list = np.array(score_list)
	print(score_list[np.argmax(score_list[:,-1])][0])
	return score_list[np.argmax(score_list[:,-1])][0]


def beam_decode_one_line(src_seq,enc_output,decoder,beam_size, max_len):
	"""
	Simple implementation of Beam Search
	Inputs
	
	src_seq: source sequence
	enc_output: encoder output
	decoder: decoder model to use
	beam_size: size of beam
	max_len: maximum output length
	
	Outputs

	result_seq: beam decoded sequence
	"""
	score_list = [[[BOS],1.0] for i in range(beam_size)]
	for idx in range(max_len):
		for candidate_idx in range(beam_size):
			tgt_seq = torch.LongTensor(score_list[candidate_idx][0])
			tgt_pos = torch.LongTensor(range(1, idx + 2))
			logit_list = next_logit(decoder,idx,tgt_seq,tgt_pos,src_seq,enc_output)
			logit_list = sorted(logit_list,key = lambda word:word[-1],reverse = True)
			new_score_list = logit_list[:beam_size]
			new_score_list = [[score_list[candidate_idx][0]+[new_score_list[i][0]],
				score_list[candidate_idx][-1] + math.log(new_score_list[i][-1]),2] for i in range(beam_size)]
			if candidate_idx == 0:
				candidate_list = new_score_list
			else:
				candidate_list = maintain_k(candidate_list,new_score_list,beam_size)
		score_list = candidate_list
	assert len(score_list) == beam_size
	score_list = np.array(score_list)
	return score_list[np.argmax(score_list[:,-1])][0]

"""
For Demo:
"""
if __name__ == '__main__':
	beam_decode_one_line_exp(3,8)



