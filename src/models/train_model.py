import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.nn import Module, Parameter
from torch import FloatTensor
from scipy import signal
import numpy as np
from torchaudio import transforms
import matplotlib.pyplot as plt
import IPython.display as ipd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from scipy.signal import sosfiltfilt
import os

from torchaudio.functional import lfilter

dirname = os.path.abspath('')
rootdir = os.path.split(dirname)[0]
rootdir = os.path.split(rootdir)[0]

H1_TRAINING_INPUT_PATH = "".join([rootdir, "/data/train/ht1-input.wav"])
H1_TRAINING_TARGET_PATH = "".join([rootdir, "/data/train/ht1-target.wav"])

metadata = torchaudio.info(H1_TRAINING_INPUT_PATH)
print(metadata)


class NeuralAudioDataSet(Dataset):
    def __init__(self, input, target, sequence_length):
        self.input = input
        self.target = target

        self._sequence_length = sequence_length
        self.input_sequence = self.wrap_to_sequences(self.input, self._sequence_length)
        self.target_sequence = self.wrap_to_sequences(self.target, self._sequence_length)
        self._len = self.input_sequence.shape[0]

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return {'input': self.input_sequence[index, :, :]
            , 'target': self.target_sequence[index, :, :]}

    def wrap_to_sequences(self, data, sequence_length):
        num_sequences = int(np.floor(data.shape[0] / sequence_length))
        print(num_sequences)
        truncated_data = data[0:(num_sequences * sequence_length)]
        wrapped_data = truncated_data.reshape((num_sequences, sequence_length, 1))
        wrapped_data = wrapped_data.permute(0, 2, 1)
        print(wrapped_data.shape)
        return np.float32(wrapped_data)


def train(criterion, model, loader, optimizer):
    model.train()
    device = next(model.parameters()).device
    total_loss = 0

    for ind, batch in enumerate(loader):
        input_seq_batch = batch['input'].to(device)
        target_seq_batch = batch['target'].to(device)

        optimizer.zero_grad()
        predicted_output = model(input_seq_batch)

        target_seq_batch_filt = lfilter(target_seq_batch, torch.Tensor([1, 0]),
                                        torch.Tensor([1, -0.95]))
        predicted_output_filt = lfilter(predicted_output, torch.Tensor([1, 0]),
                                        torch.Tensor([1, -0.95]))

        loss = criterion(target_seq_batch_filt, predicted_output_filt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    #         print(f"Loss: {loss}")

    total_loss /= len(loader)

    return total_loss
