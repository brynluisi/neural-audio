import torch.nn as nn
from torch.nn import Module, Parameter


class IIRNN(Module):
    def __init__(self, n_input=1, n_output=1, hidden_size=80, n_channel=1):
        super(IIRNN, self).__init__()
        self.hidden_size = hidden_size
        #
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size, batch_first=True)

        self.fc1 = nn.Linear(self.hidden_size, 1)

        self.mlp_layer = nn.Sequential(
            self.fc1,
        )

    def forward(self, x):
        # sequence_length: nsamples in 1 training example
        # batch_size: number of groups of audio samples
        # input_size: nchannels for each audio sample: 1
        # hidden_size: number of features for a single audio sample

        x, hn = self.lstm(x.permute(0, 2, 1))  # output; (sequence_length, batch_size,
        # hidden_size)

        #         print(x.shape)

        x = self.mlp_layer(x)

        return x.permute(0, 2, 1)
