import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout, weights):
        super().__init__()
        self.encoder = nn.Embedding.from_pretrained(weights)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.ntoken = ntoken

    def forward(self, input):
        emb = self.encoder(input)
        output, _ = self.rnn(emb)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return decoded
