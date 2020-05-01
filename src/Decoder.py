from packages import *

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout, max_length,pretrained_weight):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        self.attn_general = nn.Linear(self.max_length, self.max_length)
        self.attn_concat = nn.Linear(self.hidden_size*2, self.max_length)
        self.lcl_wa_into_hs = nn.Linear(self.hidden_size,self.hidden_size)

        self.attn_coverage = nn.Linear(self.max_length, self.hidden_size)
        self.attn_coverage_cat = nn.Linear(self.hidden_size*3, self.max_length)

        self.dropout_layer = nn.Dropout(self.dropout)
        self.gru_forward = nn.GRU(self.hidden_size, self.hidden_size)
        self.gru_backward = nn.GRU(self.hidden_size, self.hidden_size)
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
        if flag:
            output, hidden = self.gru_forward(output, hidden)
        else:
            output, hidden = self.gru_backward(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)