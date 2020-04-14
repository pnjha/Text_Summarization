from ModelLayer import ModelLayer
import json, gc

class Summarizer:
    
    def __init__(self):
        self.ml = ModelLayer()

    def get_summary(self,input_text):
        model_output = self.ml.perform_summarization(input_text)
        return model_output["summary"]

    def update_model(self,input_text,expected_summary):
        self.ml = None
        self.ml = ModelLayer()
        gc.collect()
        flag = self.ml.perform_update(input_text,expected_summary)
        self.ml = None
        gc.collect()
        self.ml = ModelLayer()
        return flag