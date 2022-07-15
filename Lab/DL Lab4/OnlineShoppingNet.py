import torch
import torch.nn as nn
import RNN_series


class RNN(nn.Module):
    def __init__(self, embeddings, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings=embeddings, freeze=False, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text = [sent len, batch size]

        embedded = self.embedding(text)  # embedded = [batch size, sent len, emb dim]

        output, hidden = self.rnn(embedded)  # output = [sent len, batch size, hid dim]

        # hidden = [1, batch size, hid dim]

        a = output[:, -1, :]
        b = hidden.squeeze(0)

        assert torch.equal(output[:, -1, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))


class LSTM(nn.Module):
    def __init__(self, embeddings, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings=embeddings, freeze=False, padding_idx=0)
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text = [sent len, batch size]

        embedded = self.embedding(text)  # embedded = [batch size, sent len, emb dim]

        output, hidden = self.LSTM(embedded)  # output = [sent len, batch size, hid dim]

        # hidden = [1, batch size, hid dim]

        a = output[:, -1, :]

        return self.fc(a)


class GRU(nn.Module):
    def __init__(self, embeddings, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings=embeddings, freeze=False, padding_idx=0)
        self.GRU = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text = [sent len, batch size]

        embedded = self.embedding(text)  # embedded = [batch size, sent len, emb dim]

        output, hidden = self.GRU(embedded)  # output = [sent len, batch size, hid dim]

        # hidden = [1, batch size, hid dim]

        a = output[:, -1, :]
        b = hidden.squeeze(0)

        assert torch.equal(output[:, -1, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))


class Bi_LSTM(nn.Module):
    def __init__(self, embeddings, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings=embeddings, freeze=False, padding_idx=0)
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        # text = [sent len, batch size]

        embedded = self.embedding(text)  # embedded = [batch size, sent len, emb dim]

        output, hidden = self.LSTM(embedded)  # output = [sent len, batch size, hid dim]

        # hidden = [1, batch size, hid dim]

        a = output[:, -1, :]

        return self.fc(a)
