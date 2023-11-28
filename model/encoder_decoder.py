from random import random

import torch
from torch import nn
from model.embed import Embedding


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers,len_his, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.len_his = len_his

        self.embedding = Embedding(d_feature=input_dim, d_model=emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.conv = nn.Conv2d(1, len_his, (1, 1))

    def forward(self, src):
        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))
        #print(src.shape)
        # embedded = [src len, batch size, emb dim]
        output_list = []
        cell_list = []
        for input in embedded:
            #print(input.shape)
            output, (hidden, cell) = self.rnn(input)
            output_list.append(hidden)
            cell_list.append(cell)
        hidden = torch.stack(output_list)
        hidden = self.conv(hidden)
        cell = torch.stack(cell_list)
        #print(hidden.shape)
        #outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden,cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hiddens, cells):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        #print(input.shape)torch.Size([32, 1, 25, 1])

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]
        prediction_list = []
        output_list = []
        hidden_list = []
        cell_list = []
        for i,hidden,cell in zip(embedded,hiddens,cells):
            #print(hidden.shape)torch.Size([1, 25, 64])
            #print(cell.shape)
            output, (hidden, cell) = self.rnn(i, (hidden, cell))
            prediction = self.fc_out(output.squeeze(0)).unsqueeze(0)

            prediction_list.append(prediction)
            output_list.append(output)
            hidden_list.append(hidden)
            cell_list.append(cell)
        prediction = torch.stack(prediction_list)
        output = torch.stack(output_list)
        hidden = torch.stack(hidden_list)
        cell = torch.stack(cell_list)
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        #prediction = self.fc_out(output)

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


# class Seq2Seq(nn.Module):
#     def __init__(self, encoder, decoder, device):
#         super().__init__()
#
#         self.encoder = encoder
#         self.decoder = decoder
#         self.device = device
#
#         assert encoder.hid_dim == decoder.hid_dim, \
#             "Hidden dimensions of encoder and decoder must be equal!"
#         assert encoder.n_layers == decoder.n_layers, \
#             "Encoder and decoder must have equal number of layers!"
#
#     def forward(self, src, trg, teacher_forcing_ratio=0.5):
#         # src = [src len, batch size]
#         # trg = [trg len, batch size]
#         # teacher_forcing_ratio is probability to use teacher forcing
#         # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
#
#         batch_size = trg.shape[1]
#         trg_len = trg.shape[0]
#         trg_vocab_size = self.decoder.output_dim
#
#         # tensor to store decoder outputs
#         outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
#
#         # last hidden state of the encoder is used as the initial hidden state of the decoder
#         hidden, cell = self.encoder(src)
#
#         # first input to the decoder is the <sos> tokens
#         input = trg[0, :]
#
#         for t in range(1, trg_len):
#             # insert input token embedding, previous hidden and previous cell states
#             # receive output tensor (predictions) and new hidden and cell states
#             output, hidden, cell = self.decoder(input, hidden, cell)
#
#             # # place predictions in a tensor holding predictions for each token
#             outputs[t] = output
#             #
#             # # decide if we are going to use teacher forcing or not
#             teacher_force = random.random() < teacher_forcing_ratio
#             #
#             # # get the highest predicted token from our predictions
#             top1 = output.argmax(1)
#             #
#             # # if teacher forcing, use actual next token as next input
#             # # if not, use predicted token
#             input = trg[t] if teacher_force else top1
#
#         return outputs


# INPUT_DIM = len(SRC.vocab)
# OUTPUT_DIM = len(TRG.vocab)
# ENC_EMB_DIM = 256
# DEC_EMB_DIM = 256
# HID_DIM = 512
# N_LAYERS = 2
# ENC_DROPOUT = 0.5
# DEC_DROPOUT = 0.5
#
# enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
# dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
#
# model = Seq2Seq(enc, dec, device).to(device)