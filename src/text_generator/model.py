import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, weights):
        super().__init__()
        self.encoder = nn.Embedding.from_pretrained(weights)
        self.rnn = nn.RNN(ninp, nhid, nlayers)
        self.decoder = nn.Linear(nhid, ntoken)

        self.ntoken = ntoken
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input):
        emb = self.encoder(input)
        output, _ = self.rnn(emb)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return nn.functional.softmax(decoded, dim=1)
