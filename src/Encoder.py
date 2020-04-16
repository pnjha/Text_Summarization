from packages import *

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru_forward = nn.GRU(hidden_size, hidden_size)
        self.gru_backward = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden, flag):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        if flag:
            output, hidden = self.gru_forward(output, hidden)
        else:
            output, hidden = self.gru_backward(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)