import os,errno
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, send_file,session, jsonify
from flask_bootstrap import Bootstrap
import requests
import sys
import json
import time
from fuzzywuzzy import fuzz

def load_data(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    # device_file_name = json.dumps(device_file_name)
    return data

def save_data(filename,data):
    with open(filename, 'w') as fp:
        json.dump(data, fp, indent=4, sort_keys=True)

list_file_name = ['Ep_250_Ds_10000_Lr_0.01_Hs_256_Ml_0.2_Tf_0.5_GR_100_test_result.json',\
'Ep_250_Ds_10000_Lr_0.01_Hs_256_Ml_0.3_Tf_0.5_GR_100_test_result.json',\
'Ep_250_Ds_10000_Lr_0.001_Hs_256_Ml_0.5_Tf_0.5_GR_100_test_result.json',\
'Ep_250_Ds_10000_Lr_0.01_Hs_256_Ml_0.08_Tf_0.5_GR_100_test_result.json',\
'Ep_250_Ds_20000_Lr_0.001_Hs_256_Ml_0.4_Tf_0.5_GR_100_test_result.json']

lst_rouge_1_f = []
lst_rouge_1_p = []
lst_rouge_1_r = []

lst_rouge_2_f = []
lst_rouge_2_p = []
lst_rouge_2_r = []

lst_rouge_l_f = []
lst_rouge_l_p = []
lst_rouge_l_r = []

statement_data = {}
metric_data = {}

def similarity(file_name,index,st_1,st_2):
	ratio = fuzz.ratio(st_1.lower(),st_2.lower())

	if ratio<=80 or ratio>=98:
		return

	if file_name not in statement_data.keys(): 
		statement_data[file_name] = {}

	statement_data[file_name][index] = {}
	statement_data[file_name][index]['Predicted'] = st_1
	statement_data[file_name][index]['Original'] = st_2
	statement_data[file_name][index]['Ratio'] = ratio	
	
def get_avg(lcl_lst_rouge_1_f):
	avg_lst_rouge_1_f = 0.0
	for item in lcl_lst_rouge_1_f:
		avg_lst_rouge_1_f = avg_lst_rouge_1_f + item 
	avg_lst_rouge_1_f = avg_lst_rouge_1_f / len(lcl_lst_rouge_1_f)
	return avg_lst_rouge_1_f

def processJson(item,json_data):

	lst_rouge_1_f = []
	lst_rouge_1_p = []
	lst_rouge_1_r = []

	lst_rouge_2_f = []
	lst_rouge_2_p = []
	lst_rouge_2_r = []

	lst_rouge_l_f = []
	lst_rouge_l_p = []
	lst_rouge_l_r = []

	for key,value in json_data.items():
		# print(json_data[key]['Score'][0]['rouge-1']['f'])
		lst_rouge_1_f\
		.append(float(json_data[key]['Score'][0]['rouge-1']['f']))
		lst_rouge_1_p\
		.append(float(json_data[key]['Score'][0]['rouge-1']['p']))
		lst_rouge_1_r\
		.append(float(json_data[key]['Score'][0]['rouge-1']['r']))

		lst_rouge_2_f\
		.append(float(json_data[key]['Score'][0]['rouge-2']['f']))
		lst_rouge_2_p\
		.append(float(json_data[key]['Score'][0]['rouge-2']['p']))
		lst_rouge_2_r\
		.append(float(json_data[key]['Score'][0]['rouge-2']['r']))

		lst_rouge_l_f\
		.append(float(json_data[key]['Score'][0]['rouge-l']['f']))
		lst_rouge_l_p\
		.append(float(json_data[key]['Score'][0]['rouge-l']['p']))
		lst_rouge_l_r\
		.append(float(json_data[key]['Score'][0]['rouge-l']['r']))

		similarity(item,key,json_data[key]['Generated_Summary'],json_data[key]['Orignal_Summary'])
	
	metric_data[item] = {}

	items_data_key = {'lst_rouge_1_f':lst_rouge_1_f,'lst_rouge_1_p':lst_rouge_1_p,'lst_rouge_1_r':lst_rouge_1_r,\
			  'lst_rouge_2_f':lst_rouge_2_f,'lst_rouge_2_p':lst_rouge_2_p,'lst_rouge_2_r':lst_rouge_2_r,\
			  'lst_rouge_l_f':lst_rouge_l_f,'lst_rouge_l_p':lst_rouge_l_p,'lst_rouge_l_r':lst_rouge_l_r}

	
	'''	
	var = get_avg(lst_rouge_1_f)
	print(" lst_rouge_1_f = ",var)
	print(" lst_rouge_1_p = ",get_avg(lst_rouge_1_p))
	print(" lst_rouge_1_r = ",get_avg(lst_rouge_1_r))

	print(" lst_rouge_2_f = ",get_avg(lst_rouge_2_f))
	print(" lst_rouge_2_p = ",get_avg(lst_rouge_2_p))
	print(" lst_rouge_2_r = ",get_avg(lst_rouge_2_r))

	print(" lst_rouge_l_f = ",get_avg(lst_rouge_l_f))
	print(" lst_rouge_l_p = ",get_avg(lst_rouge_l_p))
	print(" lst_rouge_l_r = ",get_avg(lst_rouge_l_r))

	'''

	for _item_,_value_ in items_data_key.items():
		variable = get_avg(_value_)
		print(_item_," ",variable)
		metric_data[item][_item_] = variable
		

	'''
	metric_data[key]['avg_rouge_1_f'] = {}
	metric_data[key]['avg_rouge_1_p'] = {}
	metric_data[key]['avg_rouge_1_r'] = {}
	metric_data[key] = {}
	metric_data[key] = {}
	metric_data[key] = {}
	metric_data[key] = {}
	metric_data[key] = {}
	'''


for item in list_file_name:
	print(item)
	json_data = load_data(item)
	# print(json_data)
	processJson(item,json_data)

save_data('Test_Results.json',statement_data)
save_data('Metric_Result.json',metric_data)
