import torch
from models.XBNet.base_model import Model
from data.dataset import Dataset
from aggregate_utils import length_to_range
from typing import Callable
from itertools import chain
import numpy as np

LABELS_AVAILABLE = False
LABEL_ESTIMATE_USE_PROBABILITIES = False


def default_aggregate_by(z: torch.tensor):
    return z.mean(axis=0)


class AggregateModel(Model):
    def __init__(self, layers_raw: list[dict], classification: bool = False, aggregate_by: Callable = default_aggregate_by):
        super().__init__(classification=classification, layers_raw=layers_raw)
        self.aggregate_by = aggregate_by

    def apply_aggregate_loss(self, loss: Callable, entry_predictions: torch.tensor, observations: torch.tensor,
                             lengths: list[int]):
        ranges = length_to_range(lengths)

        predictions = torch.stack(
            [self.aggregate_by(entry_predictions[r]) for r in ranges])
        return loss(predictions, observations) * (np.array(lengths).sum() / len(lengths))

    def train(self, dataset: Dataset, optimizer, loss: Callable, batch_size: int) -> None:
        '''
        Training function for training the model with the given data
        :param batch_size: number of elements in data training for one epoch
        :param loss: Loss function to be used for training
        :param dataset: Dataset object with observations metadata
        :param optimizer: Optimizer used for training
        :return:
        loss value
        '''
        data_y_batch_indices = np.random.choice(
            len(dataset.observations), size=batch_size)
        observations_batch = np.array(
            dataset.observations).take(data_y_batch_indices)
        data_x_batch_indices = list(
            chain(*[obs.entries_indices for obs in observations_batch]))
        obs_y_batch_indices = [
            obs.value_vec_index for obs in observations_batch]

        x_batch = dataset.data_x[data_x_batch_indices]
        l_batch = [obs.length for obs in observations_batch]
        y_batch = dataset.obs_y[obs_y_batch_indices]
        if LABELS_AVAILABLE:
            y_batch_est = dataset.data_y[data_x_batch_indices]
        else:
            if LABEL_ESTIMATE_USE_PROBABILITIES:
                lengths = [obs.length for obs in observations_batch]
                y_batch_est = torch.tensor(
                    np.repeat(dataset.obs_y[obs_y_batch_indices].numpy(), lengths, axis=0))
            else:
                y_batch_est = np.array([])
                for obs in observations_batch:
                    prob_0 = dataset.obs_y[obs.value_vec_index][0]
                    prob_1 = 1 - prob_0
                    p = np.array([prob_0, prob_1])
                    p /= p.sum()
                    choices = [[0, 1], [1, 0]]
                    for i in range(obs.length):
                        choice = np.random.choice([0, 1], p=p)
                        y_batch_est = np.append(
                            y_batch_est, choices[choice], axis=0)
                y_batch_est = torch.tensor(y_batch_est)

        l = None
        inp = x_batch
        out = y_batch
        try:
            if out.shape[0] >= 1:
                out = torch.squeeze(out, 1)
        except:
            pass
        self.model.get(y_batch_est.float())
        l = self.apply_aggregate_loss(
            loss, self.model(inp.float()), out.float(), l_batch)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        for i, p in enumerate(self.model.parameters()):
            if i < self.model.num_layers_boosted:
                l0 = torch.unsqueeze(
                    self.model.sequential.boosted_layers[i], 1)
                lMin = torch.min(p.grad)
                lPower = torch.log(torch.abs(lMin))
                if lMin != 0:
                    l0 = l0 * 10 ** lPower
                    p.grad += l0
                else:
                    pass
            else:
                pass

        self.model.feature_importances_ = torch.nn.Softmax(dim=0)(
            self.model.layers["0"].weight[1]).detach().numpy()
        return l
