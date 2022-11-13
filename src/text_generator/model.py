import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        ninp: int,
        nhid: int,
        nlayers: int,
        dropout: float,
        weights: torch.Tensor,
    ) -> None:
        super().__init__()
        self.encoder = nn.Embedding.from_pretrained(weights)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.ntoken = ntoken

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(input)
        output, _ = self.rnn(emb)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return decoded
