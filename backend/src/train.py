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


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(pairs,input_lang,output_lang, encoder, decoder, epochs, learning_rate):
    
    start_training = time.time()
    plot_losses = []
    no_of_epoch = []
    print_loss_total = 0 
    plot_loss_total = 0
    n_iters = len(pairs)
    print("Dataset Size: ",n_iters)

    encoder_optimizer = optim.ASGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.ASGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    epoch_loss_list = []
    
    for i in range(epochs):
        
        loss = 0
        start = time.time()
        training_pairs = [tensorsFromPair(pair,input_lang,output_lang, device, EOS_token, UNK_token) for pair in pairs]
        
        for iter in tqdm(range(1, n_iters + 1)):

            training_pair = training_pairs[iter - 1]
            input_tensor,target_tensor = training_pair[0],training_pair[1]

            loss += train(input_tensor, target_tensor, encoder,decoder, encoder_optimizer, decoder_optimizer, criterion)
        
        loss = loss/n_iters
        print('%s (%d %d%%) %.4f' % (time_since(start, (i+1)/epochs), (i+1), (i+1)/epochs*100, loss))
        epoch_loss_list.append(loss)
        
        if loss < min_loss:
            break

    while len(epoch_loss_list) < epochs:
        epoch_loss_list.append(epoch_loss_list[-1])

    end_training = time.time()
    print("Total time taken for training: %.5f",as_minutes(start_training - end_training))    
    return epoch_loss_list,epochs    

def main():

    #path to save the model
    model_name = "Ep_{}_Ds_{}_Lr_{}_Hs_{}_Ml_{}_Tf_{}".format(epoch,DATA_SIZE,
                    lcl_learning_rate,no_of_hidden_size, min_loss, teacher_forcing_ratio)
    model_folder_path = CWD + "/model/" + model_name
    output_folder_path = model_folder_path + "/output"
    plot_file_name = model_name+"_ep_vs_loss.png"
    plot_path = "{}/{}".format(output_folder_path,plot_file_name)

    try:
        os.system("mkdir {}".format(model_folder_path))
        os.system("mkdir {}".format(output_folder_path))
    except Exception as e:
        print(e)

    encoder_path = "{}/{}_Encoder.pt".format(model_folder_path,model_name)
    decoder_path = "{}/{}_Decoder.pt".format(model_folder_path,model_name)
    params_path = "{}/{}_params.pt".format(model_folder_path,model_name)
    vocab_path = "{}/{}_vocab.pt".format(model_folder_path,model_name)    
    
    
    print("Encoder Model Path :" ,encoder_path)
    print("Decoder Model Path :" ,decoder_path)
    print("Params Path :" ,params_path)
    print("Vocab Path :" ,vocab_path)
    
    print("Encoder Model Exist " ,os.path.isfile(encoder_path))
    print("Decoder Model Exist " ,os.path.isfile(decoder_path))
    print("Params File Exist " ,os.path.isfile(params_path))
    print("Vocab File Exist " ,os.path.isfile(vocab_path))

    if os.path.isfile(encoder_path) and os.path.isfile(decoder_path) and os.path.isfile(params_path) and os.path.isfile(vocab_path):
        print("Model already exists, training over")
        return 0

    train_original_path = CWD + "/data/train.original"
    train_compressed_path = CWD + "/data/train.compressed"
    train_combined = CWD + "/data/train.combined"
    contraction_mapping_path = CWD +  "/data/contraction_mapping.json"
    source_prefix,target_prefix = "original", "compressed"
    contraction_mapping = load_data(contraction_mapping_path)

    # if os.path.isfile(train_combined) == False:
    prepareInput(train_original_path,train_compressed_path,train_combined,contraction_mapping,DATA_SIZE)
    
    input_lang, output_lang, pairs = prepareData(source_prefix,target_prefix,train_combined,SOS_token,EOS_token,UNK_token)

    encoder_model = None
    decoder_model = None
    params = None
    vocab = None

    encoder_model = EncoderRNN(input_lang.n_words, no_of_hidden_size).to(device)
    decoder_model = AttnDecoderRNN(no_of_hidden_size, output_lang.n_words,dropout,MAX_LENGTH).to(device)
    plot_losses,no_of_epoch = trainIters(pairs,input_lang,output_lang,encoder_model, decoder_model, epoch ,lcl_learning_rate)
    
    torch.save(encoder_model.state_dict(), encoder_path)
    torch.save(decoder_model.state_dict(), decoder_path)

    params = {'loss_list':plot_losses,'epochs':no_of_epoch,'lr':lcl_learning_rate,
        "dropout":dropout, "MAX_LENGTH":MAX_LENGTH,"epoch":epoch, 
        "no_of_hidden_size":no_of_hidden_size}

    vocab = {'input_lang':input_lang, 'output_lang':output_lang}

    torch.save(params, params_path)
    torch.save(vocab, vocab_path)
    
    show_plot(plot_losses,plot_path)

if __name__=='__main__':
    main()