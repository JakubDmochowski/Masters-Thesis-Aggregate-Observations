from tkinter import Y
import torch
from XBNet.base_model import Model
from data.dataset import Dataset
from typing import Callable
from itertools import chain
import numpy as np

class StandardModel(Model):
  
    # def train(self, trainDataload, criterion, optimizer):
    def train(self, dataset: Dataset, optimizer, loss: Callable, batch_size: int) -> None:
        '''
        Training function for training the model with the given data
        :param model(XBNET Classifier/Regressor): model to be trained
        :param trainDataload(object of DataLoader): DataLoader with training data
        :param criterion(object of loss function): Loss function to be used for training
        :param optimizer(object of Optimizer): Optimizer used for training
        :param epochs(int,optional): Number of epochs for training the model. Default value: 100
        :return:
        list of training accuracy, training loss, testing accuracy, testing loss for all the epochs
        '''
        data_y_batch_indices = np.random.choice(
            len(dataset.observations), size=batch_size)
        observations_batch = np.array(
            dataset.observations).take(data_y_batch_indices)
        data_x_batch_indices = list(
            chain(*[obs.entries_indices for obs in observations_batch]))

        x_batch = dataset.data_x[data_x_batch_indices]
        y_batch = dataset.data_y[data_x_batch_indices]
        l = None
        inp = x_batch
        out = y_batch
        try:
            if out.shape[0] >= 1:
                out = torch.squeeze(out, 1)
        except:
            pass
        self.model.get(out.float())
        l = loss(self.model(inp.float()), out.float())
        # if self.model.first_layer_output_dim == 1:
        #     l = loss(y_pred, out.view(-1, 1).float())
        # else:
        #     l = loss(y_pred, out.long())
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        for i, p in enumerate(self.model.parameters()):
            if i < self.model.num_layers_boosted:
                l0 = torch.unsqueeze(self.model.sequential.boosted_layers[i], 1)
                lMin = torch.min(p.grad)
                lPower = torch.log(torch.abs(lMin))
                if lMin != 0:
                    l0 = l0 * 10 ** lPower
                    p.grad += l0
                else:
                    pass
            else:
                pass
            
        self.model.feature_importances_ = torch.nn.Softmax(dim=0)(self.model.layers["0"].weight[1]).detach().numpy()
        return l

