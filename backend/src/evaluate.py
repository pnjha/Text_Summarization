from Lang import *
from utils import *
from Decoder import  *
from Encoder import  *
from packages import *
from data_processing import *

CWD = os.getcwd()

test_original_path = CWD + "/data/test.original"
test_compressed_path = CWD + "/data/test.compressed"
params_path = CWD + "/config/params.json"
contraction_mapping_path = CWD +  "/data/contraction_mapping.json"

params = load_data(params_path)

epoch = params["epoch"]
dropout = params["dropout"]
min_loss = params["min_loss"]
n_layers = params["n_layers"]
SOS_token = params["SOS_token"]
EOS_token = params["EOS_token"]
UNK_token = params["UNK_token"]
DATA_SIZE = params["DATA_SIZE"]
MAX_LENGTH = params["MAX_LENGTH"]
batch_size = params["batch_size"]
no_of_hidden_size = params["no_of_hidden_size"]
lcl_learning_rate = params["lcl_learning_rate"]
teacher_forcing_ratio = params["teacher_forcing_ratio"]

#path to save the model
model_name = "Ep_{}_Ds_{}_Lr_{}_Hs_{}_Ml_{}_Tf_{}".format(epoch,DATA_SIZE,
                no_of_hidden_size, min_loss, teacher_forcing_ratio)
model_folder_path = CWD + "/model/" + model_name

encoder_path = "{}/{}_Encoder.pt".format(model_folder_path,model_name)
decoder_path = "{}/{}_Decoder.pt".format(model_folder_path,model_name)

contraction_mapping = load_data(contraction_mapping_path)

def evaluate(encoder, decoder, sentence, max_length):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device, EOS_token, UNK_token)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

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

def load_saved_encoder(lcl_input_lang,lcl_hidden_size,lcl_encoder_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_model = EncoderRNN(lcl_input_lang.n_words, lcl_hidden_size).to(device)
    encoder_model.load_state_dict(torch.load(lcl_encoder_model_path, map_location=device))
    return encoder_model

def load_saved_decoder(lcl_hidden_size,lcl_output_lang,lcl_decoder_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder_model = AttnDecoderRNN(lcl_hidden_size,lcl_output_lang.n_words,dropout,MAX_LENGTH).to(device)
    decoder_model.load_state_dict(torch.load(lcl_decoder_model_path, map_location=device))
    return decoder_model

def load_obj(obj_type,obj_name_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj_type = torch.load(obj_name_path, map_location=device)
    return obj_type

def calculate_rouge(rouge, pred_trg, real_trg):

    pred_trg = " ".join(pred_trg)
    real_trg = " ".join(real_trg[0])
    if len(pred_trg) > len(real_trg):
        diff = len(pred_trg) - len(real_trg)
        real_trg = real_trg +" "+  "#"*(diff-1)
    elif len(pred_trg) < len(real_trg):
        diff = len(real_trg) - len(pred_trg)
        pred_trg = pred_trg +" "+ "#"*(diff-1)
    scores = rouge.get_scores(pred_trg, real_trg)
    return scores 

def calculate_Result(encoder, decoder,lcl_pairs, n=50):
    result_value_rouge_score = []
    rouge = Rouge()
    
    for i in range(n):
        pair = random.choice(lcl_pairs)

        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)

        reference = [pair[1].split()]

        output_words = output_words[:-1]
        target_predicted = output_words
        
        score = calculate_rouge(rouge,target_predicted,reference)
        result_value_rouge_score.append((pair[0],pair[1].split(),target_predicted,score))

    return result_value_rouge_score

def main():

    test_result_data_path = "{}/{}_vocab.pt".format(model_folder_path,model_name)

    print("Model Already Exist")
         hidden_size = no_of_hidden_size
         encoder_model = load_saved_encoder(input_lang,hidden_size,encoder_model_path)
         decoder_model = load_saved_decoder(hidden_size,output_lang,decoder_model_path)
         model_performance = load_obj(model_performance,params)
         vocab = load_obj(vocab,vocab_params)

result_value_rouge_score = calculate_Result(encoder_model, decoder_model,pairs)
    result_value_rouge_score_dict = {}
    result_value_rouge_score_dict['result'] = result_value_rouge_score 
    torch.save(result_value_rouge_score_dict, train_result_data_path)

    limit = 10

    for item in result_value_rouge_score:
      print(" Source Language ",item[0])
      print(" Input Target",item[1])
      print(" Output Target",item[2])
      print(" Score ",item[3])
      limit -= 1

    prefix = "test"
    sourceLangPath = root_directory +prefix+".original"
    sourcePrefix = "original"
    targetLangPath = root_directory+prefix+".compressed"
    targetPrefix = "compressed"
    train_combined = prefix + "_" + sourcePrefix + "_" + targetPrefix + ".txt"


    #isFilePresent = os.path.isfile(train_combined)
    #if isFilePresent == False:
    prepareInput(sourceLangPath,targetLangPath,train_combined,contraction_mapping)
    test_input_lang, test_output_lang, test_pairs = prepareData(sourcePrefix, targetPrefix,train_combined)
    print(random.choice(test_pairs))

    result_value_rouge_score_test = calculate_Result(encoder_model, decoder_model,test_pairs)
    result_value_rouge_score_dict_test = {}
    result_value_rouge_score_dict_test['result'] = result_value_rouge_score_test 
    torch.save(result_value_rouge_score_dict_test, test_result_data_path)

    limit = 10

    for item in result_value_rouge_score_test:
        print(" Source Language ",item[0])
        print(" Input Target",item[1])
        print(" Output Target",item[2])
        print(" Score ",item[3])
        limit -= 1