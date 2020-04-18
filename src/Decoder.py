from packages import *

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout, max_length, n_layers):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.max_length = max_length
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        self.attn_general = nn.Linear(self.max_length, self.max_length)
        self.attn_concat = nn.Linear(self.hidden_size*2, self.max_length)
        self.lcl_wa_into_hs = nn.Linear(self.hidden_size,self.hidden_size)

        self.attn_coverage = nn.Linear(self.max_length, self.hidden_size)
        self.attn_coverage_cat = nn.Linear(self.hidden_size*3, self.max_length)

        self.dropout_layer = nn.Dropout(self.dropout)
        self.lstm_forward = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.n_layers)
        self.lstm_backward = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.n_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, flag):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout_layer(embedded)
        
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        lcl_cumulative_sum = torch.cumsum(attn_weights, -1)
        lcl_Y1 = self.attn_coverage(lcl_cumulative_sum)
        lcl_Y2 = torch.cat((torch.cat((embedded[0], hidden[0]), 1), lcl_Y1),1)
        attn_weights_lcl = self.attn_coverage_cat(lcl_Y2)
        attn_weights = F.softmax(attn_weights_lcl, dim = 1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        out = output
        for i in range(self.n_layers-1):
            output = torch.cat((output,out))
        
        if flag:
            output, hidden = self.lstm_forward(embedded, (output, hidden))
        else:
            output, hidden = self.lstm_backward(embedded, (output, hidden))

        hidden = hidden[0] + hidden[1]
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size, device=device).to(device)