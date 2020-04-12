from Lang import *
from utils import *
from Decoder import  *
from Encoder import  *
from packages import *
from data_processing import *

CWD = os.getcwd()

params_path = CWD + "/config/params.json"

print(params_path)
params = load_data(params_path)

print(params)
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


def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    max_length = MAX_LENGTH
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

def load_obj(obj_name_path):
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

def calculate_Result(encoder, decoder,pairs,input_lang, output_lang):
    result_value_rouge_score = {}
    rouge = Rouge()
    
    idx = 1

    for pair in pairs:
        
        output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)

        reference = [pair[1].split()]

        output_words = output_words[:-1]
        target_predicted = output_words
        
        score = calculate_rouge(rouge,target_predicted,reference)
        
        result_value_rouge_score[idx] = {}
        result_value_rouge_score[idx]["Original_Text"] = pair[0]
        result_value_rouge_score[idx]["Orignal_Summary"] = pair[1]
        result_value_rouge_score[idx]["Generated_Summary"] = " ".join(target_predicted)
        result_value_rouge_score[idx]["Score"] = score
        idx += 1

    return result_value_rouge_score

def main():

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
    model_name = "Ep_{}_Ds_{}_Lr_{}_Hs_{}_Ml_{}_Tf_{}".format(epoch,DATA_SIZE,
                    lcl_learning_rate,no_of_hidden_size, min_loss, teacher_forcing_ratio)
    model_folder_path = CWD + "/model/" + model_name

    encoder_path = "{}/{}_Encoder.pt".format(model_folder_path,model_name)
    decoder_path = "{}/{}_Decoder.pt".format(model_folder_path,model_name)
    params_path = "{}/{}_params.pt".format(model_folder_path,model_name)
    vocab_path = "{}/{}_vocab.pt".format(model_folder_path,model_name)    
    train_result_path = CWD + "/results/{}_train_result.json".format(model_name)
    test_result_path = CWD + "/results/{}_test_result.json".format(model_name)
        
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
        print("Model doen not exists. Evaluation over")
        return 0


    vocab = load_obj(vocab_path)
    params = load_obj(params_path)
    train_input_lang = vocab["input_lang"]
    train_output_lang = vocab["output_lang"]
    encoder_model = load_saved_encoder(train_input_lang,no_of_hidden_size,encoder_path)
    decoder_model = load_saved_decoder(no_of_hidden_size,train_output_lang,decoder_path)

    prepareInput(train_original_path,train_compressed_path,train_combined,contraction_mapping,DATA_SIZE)
    prepareInput(test_original_path,test_compressed_path,test_combined,contraction_mapping,DATA_SIZE)

    train_input_lang, train_output_lang, train_pairs = prepareData(source_prefix, target_prefix,train_combined,SOS_token,EOS_token,UNK_token)
    test_input_lang, test_output_lang, test_pairs = prepareData(source_prefix, target_prefix,test_combined,SOS_token,EOS_token,UNK_token)

    train_result = calculate_Result(encoder_model, decoder_model,train_pairs,train_input_lang, train_output_lang)
    test_result = calculate_Result(encoder_model, decoder_model,test_pairs,train_input_lang, train_output_lang)

    save_data(train_result_path,train_result)
    save_data(test_result_path,test_result)

if __name__=='__main__':
    main()