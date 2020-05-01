from Lang import *
from utils import *
from Decoder import  *
from Encoder import  *
from packages import *
from data_processing import *
from Beam_Search import *

class Evaluate():

    def __init__(self,params_path):
        self.load_params(params_path)

    def load_params(self,params_path):
        self.params = load_data(params_path)
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
            return decoded_words

    def load_saved_encoder(self,input_lang,encoder_model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder_model = EncoderRNN(input_lang.n_words, self.HIDDEN_SIZE,self.LAYERS).to(device)
        encoder_model.load_state_dict(torch.load(encoder_model_path, map_location=device))
        return encoder_model

    def load_saved_decoder(self,output_lang,decoder_model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder_model = AttnDecoderRNN(self.HIDDEN_SIZE,output_lang.n_words,self.DROPOUT,self.MAX_LENGTH,self.LAYERS).to(device)
        decoder_model.load_state_dict(torch.load(decoder_model_path, map_location=device))
        return decoder_model

    def load_obj(self,obj_name_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj_type = torch.load(obj_name_path, map_location=device)
        return obj_type

    def calculate_rouge(self,rouge, pred_trg, real_trg):

        pred_trg = " ".join(pred_trg)
        real_trg = " ".join(real_trg[0])
        if len(pred_trg) > len(real_trg):
            diff = len(pred_trg) - len(real_trg)
            real_trg = real_trg +" "+  "#"*(diff-1)
        elif len(pred_trg) < len(real_trg):
            diff = len(real_trg) - len(pred_trg)
            pred_trg = pred_trg +" "+ "#"*(diff-1)
        if len(pred_trg) == 0:
            pred_trg = "#"
        if len(real_trg) == 0:
            real_trg = "#"
        scores = rouge.get_scores(pred_trg, real_trg)
        return scores 

    def calculate_Result(self,encoder, decoder,pairs,input_lang, output_lang):
        result_value_rouge_score = {}
        rouge = Rouge()
        
        idx = 1

        for pair in pairs:
            
            # output_words, attentions = self.evaluate(encoder, decoder, pair[0], input_lang, output_lang)
            output_words = self.evaluate(encoder, decoder, pair[0], input_lang, output_lang)
            output_sentence = ' '.join(output_words)

            reference = [pair[1].split()]

            output_words = output_words[:-1]
            target_predicted = output_words
            
            score = self.calculate_rouge(rouge,target_predicted,reference)
            
            result_value_rouge_score[idx] = {}
            result_value_rouge_score[idx]["Original_Text"] = pair[0]
            result_value_rouge_score[idx]["Orignal_Summary"] = pair[1]
            result_value_rouge_score[idx]["Generated_Summary"] = " ".join(target_predicted)
            result_value_rouge_score[idx]["Score"] = score
            idx += 1

        return result_value_rouge_score

def main():

    CWD = os.getcwd()
    params_path = CWD + "/config/params.json"

    eval = Evaluate(params_path)

    test_original_path = CWD + "/data/test.original"
    test_compressed_path = CWD + "/data/test.compressed"
    test_combined = CWD + "/data/test.combined"

    train_original_path = CWD + "/data/train.original"
    train_compressed_path = CWD + "/data/train.compressed"
    train_combined = CWD + "/data/train.combined"
    

    source_prefix,target_prefix = "original", "compressed"
    contraction_mapping_path = CWD +  "/data/contraction_mapping.json"
    contraction_mapping = load_data(contraction_mapping_path)
    
    #path to load the model
    model_name = "Ep_{}_Ds_{}_Lr_{}_Hs_{}_Ml_{}_Tf_{}_GR_{}".format(eval.EPOCHS,
                    eval.DATA_SIZE,eval.LEARNING_RATE,eval.HIDDEN_SIZE,
                    eval.MAX_TRAIN_LOSS,eval.TEACHER_FORCING_RATE,
                    eval.MAX_GRADIENT)

    model_folder_path = CWD + "/model/" + model_name

    encoder_path = "{}/{}_Encoder.pt".format(model_folder_path,model_name)
    decoder_path = "{}/{}_Decoder.pt".format(model_folder_path,model_name)
    params_path = "{}/{}_params.pt".format(model_folder_path,model_name)
    vocab_path = "{}/{}_vocab.pt".format(model_folder_path,model_name)

    output_folder_path = model_folder_path + "/output"

    train_result_filename = model_name+"_train_result.json"
    test_result_filename = model_name+"_test_result.json"

    train_result_path = "{}/{}".format(output_folder_path,train_result_filename)
    test_result_path = "{}/{}".format(output_folder_path,test_result_filename)
        
    print("Encoder Model Path :" ,encoder_path)
    print("Decoder Model Path :" ,decoder_path)
    print("Params Path :" ,params_path)
    print("Vocab Path :" ,vocab_path)
    print("Train Result Path :",train_result_path)
    print("Test Result Path :",test_result_path)

    print("Encoder Model Exist " ,os.path.isfile(encoder_path))
    print("Decoder Model Exist " ,os.path.isfile(decoder_path))
    print("Params File Exist " ,os.path.isfile(params_path))
    print("Vocab File Exist " ,os.path.isfile(vocab_path))

    if os.path.isfile(encoder_path) == False or os.path.isfile(decoder_path) == False \
        or os.path.isfile(params_path) == False \
        or os.path.isfile(vocab_path) == False:
        print("Model does not exists. Evaluation over")
        return 0


    vocab = eval.load_obj(vocab_path)
    params = eval.load_obj(params_path)
    train_input_lang = vocab["input_lang"]
    train_output_lang = vocab["output_lang"]
    encoder_model = eval.load_saved_encoder(train_input_lang,encoder_path)
    decoder_model = eval.load_saved_decoder(train_output_lang,decoder_path)

    prepareInput(train_original_path,train_compressed_path,train_combined,contraction_mapping,eval.DATA_SIZE)
    prepareInput(test_original_path,test_compressed_path,test_combined,contraction_mapping,eval.DATA_SIZE)

    train_input_lang, train_output_lang, train_pairs = prepareData(source_prefix, target_prefix,train_combined,eval.SOS_TOKEN,eval.EOS_TOKEN,eval.UNK_TOKEN)
    test_input_lang, test_output_lang, test_pairs = prepareData(source_prefix, target_prefix,test_combined,eval.SOS_TOKEN,eval.EOS_TOKEN,eval.UNK_TOKEN)

    train_result = eval.calculate_Result(encoder_model, decoder_model,train_pairs,train_input_lang, train_output_lang)
    test_result = eval.calculate_Result(encoder_model, decoder_model,test_pairs,train_input_lang, train_output_lang)

    save_data(train_result_path,train_result)
    save_data(test_result_path,test_result)

if __name__=='__main__':
    main()