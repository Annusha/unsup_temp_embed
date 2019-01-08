#!/usr/bin/python2.7

import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from viterbi_utils.src.utils.dataset import Dataset


# wrapper class to provide videos from the dataset as pytorch tensors
class DatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        # datastructure for frame indexing
        self.selectors = []
        for video in self.dataset.features:
            self.selectors += [ (video, i) for i in range(self.dataset.features[video].shape[1]) ]

    def __len__(self):
        return len(self.selectors)

    def __getitem__(self, idx):
        assert idx < len(self)
        video = self.selectors[idx][0]
        frame = self.selectors[idx][1]
        features = torch.from_numpy( self.dataset.features[video][:, frame] )
        labels = []
        for c in range(self.dataset.n_classes):
            labels.append( torch.LongTensor([1 if c in self.dataset.action_set[video] else 0]) )
        return features, labels


# the neural network
class Net(nn.Module):

    def __init__(self, input_dim, n_classes):
        super(Net, self).__init__()
        self.n_classes = n_classes
        self.fc = nn.Linear(input_dim, 256)
        self.out_fc = []
        for c in range(n_classes):
            self.out_fc.append( nn.Linear(256, 2) )
        self.out_fc = nn.Sequential(*self.out_fc)

    def forward(self, x):
        x = nn.functional.relu(self.fc(x))
        outputs = []
        for c in range(self.n_classes):
            tmp = self.out_fc[c](x)
            tmp = nn.functional.log_softmax(tmp, dim = 1)
            outputs.append(tmp)
        return outputs


# class for network training
class Trainer(object):

    def __init__(self, dataset):
        self.dataset_wrapper = DatasetWrapper(dataset)
        self.net = Net(dataset.input_dimension, dataset.n_classes)
        self.net.cuda()

    def train(self, batch_size = 512, n_epochs = 2, learning_rate = 0.1):
        dataloader = torch.utils.data.DataLoader(self.dataset_wrapper, batch_size = batch_size, shuffle = True)
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(self.net.parameters(), lr = learning_rate)
        # run for n epochs
        for epoch in range(n_epochs):
            # loop over all training data
            for i, data in enumerate(dataloader, 0):
                optimizer.zero_grad()
                input, target = data
                input = Variable(input.cuda())
                outputs = self.net(input)
                loss = 0
                for c, output in enumerate(outputs):
                    labels = Variable(target[c].cuda())
                    labels = labels.view(-1)
                    loss += criterion(output, labels)
                loss.backward()
                optimizer.step()
            print(loss)

    def save_model(self, model_file):
        torch.save(self.net.state_dict(), model_file)


# class to forward videos through a trained network
class Forwarder(object):

    def __init__(self, model_file):
        self.model_file = model_file
        self.net = None

    def forward(self, dataset):
        # read the data
        dataset_wrapper = DatasetWrapper(dataset)
        dataloader = torch.utils.data.DataLoader(dataset_wrapper, batch_size = 512, shuffle = False)
        # load net if not yet done
        if self.net == None:
            self.net = Net(dataset.input_dimension, dataset.n_classes)
            self.net.load_state_dict( torch.load(self.model_file) )
            self.net.cuda()
        # output probability container
        log_probs = np.zeros( (dataset.n_frames, dataset.n_classes), dtype=np.float32 )
        offset = 0
        # forward all frames
        for data in dataloader:
            input, _ = data
            input = Variable(input.cuda())
            outputs = self.net(input)
            for c, output in enumerate(outputs):
                log_probs[offset : offset + output.shape[0], c] = output.data.cpu()[:, 1]
            offset += output.shape[0]
        return log_probs

