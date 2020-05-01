from Lang import *
from utils import *
from Decoder import  *
from Encoder import  *
from packages import *
from data_processing import *
from train import *

class ModelLayer:

	def __init__(self):
		self.load_models_params()		

	def evaluate(self, sentence):

	    with torch.no_grad():
	        input_tensor = tensorFromSentence(self.input_lang, sentence,device, self.EOS_TOKEN, self.UNK_TOKEN)
	        input_length = input_tensor.size()[0]
	        encoder_hidden = self.encoder_model.initHidden()

	        encoder_outputs = torch.zeros(self.MAX_LENGTH, self.encoder_model.hidden_size, device=device)

	        for ei in range(input_length):
	            encoder_output, encoder_hidden = self.encoder_model(input_tensor[ei],encoder_hidden)
	            encoder_outputs[ei] += encoder_output[0, 0]

	        decoder_input = torch.tensor([[self.SOS_TOKEN]], device=device)
	        decoder_hidden = encoder_hidden
	        decoded_words = []
	        decoder_attentions = torch.zeros(self.MAX_LENGTH, self.MAX_LENGTH)

	        for di in range(self.MAX_LENGTH):
	 
	            decoder_output, decoder_hidden, decoder_attention = self.decoder_model(decoder_input, decoder_hidden, encoder_outputs)
	            decoder_attentions[di] = decoder_attention.data

	            topv, topi = decoder_output.data.topk(1)
	            if topi.item() == self.UNK_TOKEN:
	                decoded_words.append('<UNK>')
	            if topi.item() == self.EOS_TOKEN:
	                decoded_words.append('<EOS>')
	                break
	            else:
	                decoded_words.append(self.output_lang.index2word[topi.item()])

	            decoder_input = topi.squeeze().detach()

	        return decoded_words, decoder_attentions[:di + 1]

	def load_saved_encoder(self,encoder_path):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		encoder_model = EncoderRNN(self.input_lang.n_words, self.HIDDEN_SIZE).to(device)
		encoder_model.load_state_dict(torch.load(encoder_path, map_location=device))
		return encoder_model

	def load_saved_decoder(self,decoder_path):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		decoder_model = AttnDecoderRNN(self.HIDDEN_SIZE,self.output_lang.n_words,self.DROPOUT,self.MAX_LENGTH).to(device)
		decoder_model.load_state_dict(torch.load(decoder_path, map_location=device))
		return decoder_model

	def load_obj(self,obj_name_path):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		obj_type = torch.load(obj_name_path, map_location=device)
		return obj_type

	def get_summary(self,input_text):
		output_words, attentions = self.evaluate(input_text)
		output_words = output_words[:-1]
		output_sentence = ' '.join(output_words)
		return output_sentence

	def load_models_params(self):

		CWD = os.getcwd()
		self.params_path = CWD + "/config/params_2.json"

		self.params = load_data(self.params_path)
		self.EPOCHS = self.params["EPOCHS"]
		self.DROPOUT = self.params["DROPOUT"]
		self.MAX_TRAIN_LOSS = self.params["MAX_TRAIN_LOSS"]
		self.LAYERS = self.params["LAYERS"]
		self.SOS_TOKEN = self.params["SOS_TOKEN"]
		self.EOS_TOKEN = self.params["EOS_TOKEN"]
		self.UNK_TOKEN = self.params["UNK_TOKEN"]
		self.DATA_SIZE = self.params["DATA_SIZE"]
		self.MAX_GRADIENT = self.params["MAX_GRADIENT"]
		self.MAX_LENGTH = self.params["MAX_LENGTH"]
		self.BATCH_SIZE = self.params["BATCH_SIZE"]
		self.HIDDEN_SIZE = self.params["HIDDEN_SIZE"]
		self.LEARNING_RATE = self.params["LEARNING_RATE"]
		self.TEACHER_FORCING_RATE = self.params["TEACHER_FORCING_RATE"]

		contraction_mapping_path = CWD +  "/data/contraction_mapping.json"
		self.contraction_mapping = load_data(contraction_mapping_path)

		#path to load the model
		model_name = "Ep_{}_Ds_{}_Lr_{}_Hs_{}_Ml_{}_Tf_{}_GR_{}".format(self.EPOCHS,
	                    self.DATA_SIZE,self.LEARNING_RATE,self.HIDDEN_SIZE,
	                    self.MAX_TRAIN_LOSS,self.TEACHER_FORCING_RATE,
	                    self.MAX_GRADIENT)

		model_folder_path = CWD + "/model/" + model_name

		self.encoder_path = "{}/{}_Encoder.pt".format(model_folder_path,model_name)
		self.decoder_path = "{}/{}_Decoder.pt".format(model_folder_path,model_name)
		params_path = "{}/{}_params.pt".format(model_folder_path,model_name)
		self.vocab_path = "{}/{}_vocab.pt".format(model_folder_path,model_name)

		print("Encoder Model Path :" ,self.encoder_path)
		print("Decoder Model Path :" ,self.decoder_path)
		print("Params Path :" ,params_path)
		print("Vocab Path :" ,self.vocab_path)

		print("Encoder Model Exist " ,os.path.isfile(self.encoder_path))
		print("Decoder Model Exist " ,os.path.isfile(self.decoder_path))
		print("Params File Exist " ,os.path.isfile(params_path))
		print("Vocab File Exist " ,os.path.isfile(self.vocab_path))
		
		if os.path.isfile(self.encoder_path) == False \
			or os.path.isfile(self.decoder_path) == False \
	        or os.path.isfile(params_path) == False \
	        or os.path.isfile(self.vocab_path) == False:
	        	print("Model does not exists. Cannot generate summary")
	        	return 0

		self.vocab = self.load_obj(self.vocab_path)
		self.input_lang = self.vocab["input_lang"]
		self.output_lang = self.vocab["output_lang"]
		self.encoder_model = self.load_saved_encoder(self.encoder_path)
		self.decoder_model = self.load_saved_decoder(self.decoder_path)


	def perform_summarization(self,input_text):

		input_text = process_text(input_text,self.contraction_mapping)
		summary = self.get_summary(input_text)
		data = {}
		data["summary"] = summary
		return data

	def perform_update(self,input_txt,output_txt):
		try:
			pair = [[input_txt,output_txt]]
			trainer = Train(self.params_path)
			# self.input_lang.addSentence(input_txt)
			# self.output_lang.addSentence(output_txt)
			plot_losses,no_of_epoch = trainer.trainIters(pair,self.input_lang,
							self.output_lang,self.encoder_model,self.decoder_model)
			
			# self.vocab = {"input_lang":self.input_lang,"output_lang":self.output_lang}
			torch.save(self.vocab, self.vocab_path)
			torch.save(self.encoder_model.state_dict(), self.encoder_path)
			torch.save(self.decoder_model.state_dict(), self.decoder_path)
			print(plot_losses)
			print(no_of_epoch)
			return True
		except Exception as e:
			print(e)
			return False