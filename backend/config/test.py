# from temp import * 

def load_data(data_path):
	with open(data_path, 'r') as fp:
		return json.load(fp)

# def save_data(data_path,data):
# 	with open(data_path, 'w') as fp:
# 		json.dump(data, fp, indent=4, sort_keys=True)

# h = hello()
# print(h.geta())
# params = {}

# params["MAX_LENGTH"] = 200
# params["root_directory"] = ""
# params["DATA_SIZE"] = 10000
# params["epoch"] = 250
# params["no_of_hidden_size"] = 256
# params["learning_rate"] = 0.001
# params["min_loss"] = 0.7
# params["dropout"] = 0.1
# params["n_layers"] = 1
# params["teacher_forcing_ratio"] = 0.5
# params["batch_size"] = 1
# params["SOS_token"] = 0
# params["EOS_token"] = 1
# params["UNK_token"] = 2

# save_data("params.json",params)
# print(load_data("params.json"))

def h(a):
	print(a)