from Lang import *
from utils import *
from Decoder import  *
from Encoder import  *
from packages import *
from data_processing import *

class Train():

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

    def train(self, input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
        encoder_hidden_forward = encoder.initHidden()
        encoder_hidden_backward = encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs_forward = torch.zeros(self.MAX_LENGTH, encoder.hidden_size, device=device)
        encoder_outputs_backward = torch.zeros(self.MAX_LENGTH, encoder.hidden_size, device=device)

        loss = 0

        for ei in range(input_length):
            encoder_output_forward, encoder_hidden_forward = encoder(input_tensor[ei], encoder_hidden_forward,True)
            encoder_outputs_forward[ei] = encoder_output_forward[0, 0]

        for ei in range(input_length-1,-1,-1):
            encoder_output_backward, encoder_hidden_backward = encoder(input_tensor[ei], encoder_hidden_backward,False)
            encoder_outputs_backward[ei] = encoder_output_backward[0, 0]


        decoder_input_forward = torch.tensor([[self.SOS_TOKEN]], device=device)
        decoder_input_backward = torch.tensor([[self.EOS_TOKEN]], device=device)
        
        decoder_hidden_backward = encoder_hidden_forward
        decoder_hidden_forward = encoder_hidden_backward

        forward_output = []
        backward_output = []

        forward_flag, backward_flag = True, True

        use_teacher_forcing = True if random.random() < self.TEACHER_FORCING_RATE else False

        if use_teacher_forcing:

            for di in range(target_length):

                if forward_flag:
                    decoder_output_forward, decoder_hidden_forward, decoder_attention_forward = decoder(decoder_input_forward, decoder_hidden_forward, encoder_outputs_backward,True)
                    forward_output.append(decoder_output_forward)
                    decoder_input_forward = target_tensor[di] 
                if backward_flag:
                    decoder_output_backward, decoder_hidden_backward, decoder_attention_backward = decoder(decoder_input_backward, decoder_hidden_backward, encoder_outputs_forward,False)
                    backward_output.append(decoder_output_backward)
                    decoder_input_backward = target_tensor[target_length -1 -di] 
                    

        else:
            for di in range(target_length):
                
                if forward_flag:
                    decoder_output_forward, decoder_hidden_forward, decoder_attention_forward = decoder(decoder_input_forward, decoder_hidden_forward, encoder_outputs_backward,True)
                    forward_output.append(decoder_output_forward)
                    topv, topi = decoder_output_forward.topk(1)
                    decoder_input_forward = topi.squeeze().detach()
                    if decoder_input_forward.item() == self.EOS_TOKEN:
                        forward_flag = False

                if backward_flag:
                    decoder_output_backward, decoder_hidden_backward, decoder_attention_backward = decoder(decoder_input_backward, decoder_hidden_backward, encoder_outputs_forward,False)
                    backward_output.append(decoder_output_backward)
                    topv, topi = decoder_output_backward.topk(1)
                    decoder_input_backward = topi.squeeze().detach()
                    if decoder_input_backward.item() == self.SOS_TOKEN:
                        backward_flag = False

        backward_output = backward_output[::-1]

        for di in range(min(len(forward_output),len(backward_output))):
            loss += criterion((forward_output[di]+backward_output[di]), target_tensor[di])

        loss.backward()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), self.MAX_GRADIENT)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), self.MAX_GRADIENT)
        
        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length

    def trainIters(self,pairs,input_lang,output_lang, encoder, decoder, encoder_path, decoder_path):
    
        start_training = time.time()
        plot_losses = []
        no_of_epoch = []
        
        n_iters = len(pairs)
        print("Dataset Size: ",n_iters)

        encoder_optimizer = optim.ASGD(encoder.parameters(), lr=self.LEARNING_RATE)
        decoder_optimizer = optim.ASGD(decoder.parameters(), lr=self.LEARNING_RATE)
        criterion = nn.NLLLoss()

        epoch_loss_list = []
        
        training_pairs = [tensorsFromPair(pair,input_lang,output_lang, device, self.EOS_TOKEN, self.UNK_TOKEN) for pair in pairs]
        
        for i in range(self.EPOCHS):
            
            loss = 0
            start = time.time()
            
            for iter in tqdm(range(1, n_iters + 1)):

                training_pair = training_pairs[iter - 1]
                input_tensor,target_tensor = training_pair[0],training_pair[1]

                loss += self.train(input_tensor, target_tensor, encoder,decoder, encoder_optimizer, decoder_optimizer, criterion)
            
            loss = loss/n_iters
            print('%s (%d %d%%) %.4f' % (time_since(start, (i+1)/self.EPOCHS), (i+1), (i+1)/self.EPOCHS*100, loss))
            epoch_loss_list.append(loss)
            no_of_epoch.append(i+1)
            if loss < self.MAX_TRAIN_LOSS:
                break

            torch.save(encoder.state_dict(), encoder_path)
            torch.save(decoder.state_dict(), decoder_path)

        end_training = time.time()
        print("Total time taken for training: %.5f",as_minutes(start_training - end_training))    
        return epoch_loss_list,no_of_epoch    

    # def save_model(self,encoder_model,decoder_model,params,vocab,
    #                encoder_path,decoder_path,params_path,vocab_path):
    #     torch.save(encoder_model.state_dict(), encoder_path)
    #     torch.save(decoder_model.state_dict(), decoder_path)
    #     torch.save(params, params_path)
    #     torch.save(vocab, vocab_path)


def main():

    CWD = os.getcwd()
    params_path = CWD + "/config/params.json"

    trainer = Train(params_path)

    #path to save the model
    model_name = "Ep_{}_Ds_{}_Lr_{}_Hs_{}_Ml_{}_Tf_{}_GR_{}".format(trainer.EPOCHS,
                    trainer.DATA_SIZE,trainer.LEARNING_RATE,trainer.HIDDEN_SIZE,
                    trainer.MAX_TRAIN_LOSS,trainer.TEACHER_FORCING_RATE,
                    trainer.MAX_GRADIENT)

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
    prepareInput(train_original_path,train_compressed_path,train_combined,contraction_mapping,trainer.DATA_SIZE)
    input_lang, output_lang, pairs = prepareData(source_prefix,target_prefix,train_combined,trainer.SOS_TOKEN,trainer.EOS_TOKEN,trainer.UNK_TOKEN)

    encoder_model = None
    decoder_model = None
    params = None
    vocab = None

    vocab = {'input_lang':input_lang, 'output_lang':output_lang}
    torch.save(trainer.params, params_path)
    torch.save(vocab, vocab_path)

    encoder_model = EncoderRNN(input_lang.n_words, trainer.HIDDEN_SIZE,trainer.LAYERS).to(device)
    decoder_model = AttnDecoderRNN(trainer.HIDDEN_SIZE, output_lang.n_words,trainer.DROPOUT,trainer.MAX_LENGTH,trainer.LAYERS).to(device)
    plot_losses,no_of_epoch = trainer.trainIters(pairs,input_lang,output_lang,encoder_model, decoder_model,encoder_path,decoder_path)
    

    # trainer.save_model(encoder_model,decoder_model,trainer.params,vocab,
    #                    encoder_path,decoder_path,params_path,vocab_path)

    torch.save(encoder_model.state_dict(), encoder_path)
    torch.save(decoder_model.state_dict(), decoder_path)
    
    show_plot(no_of_epoch,plot_losses,plot_path)

if __name__=='__main__':
    main()