from keras.datasets import imdb
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import argparse


class LSTMIMDB(nn.Module):
    # def __init__(self, hidden_dim, embedding_dim):
    def __init__(self, embeddings, conf):
        self.conf = conf
        super(LSTMIMDB, self).__init__()
        # self.hidden_dim = hidden_dim
        # self.embedding_layer = nn.Embedding(self.conf.num_words, self.conf.embedding_dim)
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings,freeze=False)
        # print('embedding: ',self.embedding_layer)

        ### Add 1d convolution
        self.conv = nn.Conv1d(in_channels=self.conf.embed_dim, out_channels=self.conf.embed_dim,
                              kernel_size=7, padding=2, stride=1) #to keep same dimension
        # # print('conv: ', self.conv)
        self.maxpooling = nn.MaxPool1d(kernel_size=2)
        # # print('max pooling: ', self.maxpooling)

        ### Add dropout before LSTM layer
        self.dropout1 = nn.Dropout(self.conf.dropout1)

        self.lstm_layer = nn.LSTM(self.conf.embed_dim, self.conf.hidden_dim,batch_first=True)

        ### Add dropout within LSTM hidden dimensions
        # self.lstm_layer = nn.LSTM(self.conf.embed_dim, self.conf.hidden_dim, dropout=self.conf.dropout_rnn, batch_first=True)
        # print('lstm', self.lstm_layer)

        ### Add dropout after LSTM layer
        self.dropout2 = nn.Dropout(self.conf.dropout2)

        self.linear_layer = nn.Linear(self.conf.hidden_dim, 2)
        # print('linear layer', self.linear_layer)

    def forward(self, inputs, hidden):
        x = self.embedding_layer(inputs)
            # .view(len(inputs), 1, -1)

        ## Add 1d convolution
        # print('x size: ', x.size())
        x = x.transpose(1, 2)
        # print('xt size: ', x.size())
        y = self.conv(x)
        # y = x
        # print('y conv size: ', y.size())
        z = self.maxpooling(y)
        # print('z max pooling size: ', z.size())
        p = F.relu(z)
        # print('p relu size: ', p.size())
        p = p.transpose(1, 2)
        x = p

        ### Add dropout before LSTM layer
        x = self.dropout1(x)
        lstm_out, lstm_hidden = self.lstm_layer(x, hidden)
        # lstm_hidden (last_hidden_state, last_cell_state)
        # print('lstm out size', lstm_out.size())
        # print('lstm hidden size', lstm_hidden[0].size())
        # print('lstm cell size', lstm_hidden[1].size())
        # batch_first = lstm_out.transpose(0, 1)
        # linear_input = batch_first.view(, -1)
        lstm_last_out = lstm_out.transpose(0, 1)[-1]
        # print('lstm last out size',lstm_last_out.size())

        ### Add dropout after LSTM layer
        lstm_last_out = self.dropout2(lstm_last_out)
        hidden2linear = self.linear_layer(lstm_last_out)
        predicted = F.log_softmax(hidden2linear)
        return predicted, lstm_hidden

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(1, batch_size, self.conf.hidden_dim)),
                autograd.Variable(torch.zeros(1, batch_size, self.conf.hidden_dim)))