from ModelLayer import ModelLayer
import json

class Summarizer:
    
    def __init__(self):
        self.ml = ModelLayer()
        self.data_path = "data.json"
        self.data = {}
        self.save_data()

    def get_summary(self,input_text):
        model_output = self.ml.perform_summarization(input_text)
        return model_output["summary"]

    def update_model(self,input_text,expected_summary):
    	self.load_data()
    	self.data["input_text"] = expected_summary
    	self.save_data()

    def load_data(self):
    	with open(self.data_path, 'r') as fp:
    		self.data = json.load(fp)

    def save_data(self):
	    with open(self.data_path, 'w') as fp:
	        json.dump(self.data, fp, indent=4, sort_keys=True)

	
	# def update_model_params():
	# 	while True:
			