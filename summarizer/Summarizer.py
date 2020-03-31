from ModelLayer import ModelLayer

ml = ModelLayer()

class Summarizer:
    
    def __init__(self):
        self.a = 1

    def get_summary(self,input_text):
        data = ml.perform_summarization(input_text)
        return data["summary"]