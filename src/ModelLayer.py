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

    def evaluate(self,encoder, decoder, sentence, input_lang, output_lang):
        encoder.eval()
        decoder.eval()
        max_length = self.MAX_LENGTH
        
        with torch.no_grad():
        
            input_tensor = tensorFromSentence(input_lang, sentence, device, self.EOS_TOKEN, self.UNK_TOKEN)
            input_length = input_tensor.size()[0]

            encoder_hidden_forward = encoder.initHidden()
            encoder_hidden_backward = encoder.initHidden()

            encoder_outputs_forward = torch.zeros(max_length, encoder.hidden_size, device=device)
            encoder_outputs_backward = torch.zeros(max_length, encoder.hidden_size, device=device)

            for ei in range(input_length):
                encoder_output_forward, encoder_hidden_forward = encoder(input_tensor[ei],encoder_hidden_forward,True)
                encoder_outputs_forward[ei] += encoder_output_forward[0, 0]

            for ei in range(input_length-1,-1,-1):
                encoder_output_backward, encoder_hidden_backward = encoder(input_tensor[ei], encoder_hidden_backward,False)
                encoder_outputs_backward[ei] = encoder_output_backward[0, 0]

            decoder_input_forward = torch.tensor([[self.SOS_TOKEN]], device=device)  # SOS
            decoder_input_backward = torch.tensor([[self.EOS_TOKEN]], device=device)  # SOS

            decoder_hidden_forward = encoder_hidden_backward
            decoder_hidden_backward = encoder_hidden_forward

            forward_flag = True
            backward_flag = True
            decoded_words = []
            decoder_output = []
            decoder_output_forward_list = []
            decoder_output_backward_list = []
                       
            # decoder_attentions_forward = torch.zeros(max_length, max_length)
            # decoder_attentions_backward = torch.zeros(max_length, max_length)
            # decoder_attentions[di] = decoder_attention.data

            for di in range(max_length):

                if forward_flag:
                    decoder_output_forward, decoder_hidden_forward, decoder_attention_forward = decoder(decoder_input_forward, decoder_hidden_forward, encoder_outputs_backward,True)
                    decoder_output_forward_list.append(decoder_output_forward)
                    topv, topi = decoder_output_forward.data.topk(1)
                    decoder_input_forward = topi.squeeze().detach()
                    if topi.item() == self.EOS_TOKEN:
                        forward_flag = False                    
                
                if backward_flag:
                    decoder_output_backward, decoder_hidden_backward, decoder_attention_backward = decoder(decoder_input_backward, decoder_hidden_backward, encoder_outputs_forward,True)
                    decoder_output_backward_list.append(decoder_output_backward)
                    topv, topi = decoder_output_backward.data.topk(1)
                    decoder_input_backward = topi.squeeze().detach()
                    if topi.item() == self.SOS_TOKEN:
                        backward_flag = False
                
                if forward_flag == False and backward_flag == False:
                    break

            decoder_output_backward_list = decoder_output_backward_list[::-1]

            for i in range(min(len(decoder_output_backward_list), len(decoder_output_forward_list))):
                decoder_output.append(decoder_output_backward_list[i] + decoder_output_forward_list[i])

            if len(decoder_output_backward_list) < len(decoder_output_forward_list):
                for i in range(len(decoder_output_backward_list)-1,len(decoder_output_forward_list)):
                    decoder_output.append(decoder_output_forward_list[i])
            
            elif len(decoder_output_backward_list) > len(decoder_output_forward_list):                     
                for i in range(len(decoder_output_forward_list)-1,len(decoder_output_backward_list)):
                    decoder_output.append(decoder_output_backward_list[i])
            decoder_output = bi_directional_beam_decode(decoder_output_forward_list,decoder_output_backward_list)
            for i in range(len(decoder_output)):
                # topv, topi = decoder_output[i].data.topk(1)
                if topi.item() == self.UNK_TOKEN:
                    decoded_words.append('<UNK>')
                else:
                    if topi.item() == self.EOS_TOKEN:
                        decoded_words.append('<EOS>')
                        break
                    else:
                        decoded_words.append(output_lang.index2word[topi.item()])


            # return decoded_words, decoder_attentions[:di + 1]
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