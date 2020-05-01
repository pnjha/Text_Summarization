from packages import *

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,pretrained_weight):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.lstm_forward = nn.LSTM(hidden_size, hidden_size, num_layers=self.n_layers)
        self.lstm_backward = nn.LSTM(hidden_size, hidden_size,num_layers=self.n_layers)
    def forward(self, input, hidden, flag):

        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers-1):
            output = torch.cat((output,embedded))

        if flag:
            output, hidden = self.lstm_forward(output,(hidden,output))
        else:
            output, hidden = self.lstm_backward(output,(hidden,output))
        hidden = hidden[0] + hidden[1]
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size, device=device).to(device)

