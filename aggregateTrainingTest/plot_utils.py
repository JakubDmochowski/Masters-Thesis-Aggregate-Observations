import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Callable
import torch


def plotLosses(loss_history: list[list[float]]):
    fig, ax = plt.subplots(figsize=(8, 8))
    x = range(0, len(loss_history))
    aggregate = [losses[0].detach().numpy() for losses in loss_history]
    standard = [losses[1].detach().numpy() for losses in loss_history]
    ax.plot(x, aggregate, label="aggregate")
    ax.plot(x, standard, label="standard")
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.legend()
    plt.yscale("log")
    fig.show()


def plotXY(data_x: torch.tensor, expected_y: torch.tensor, valueFunc: Callable, series: list[dict]):
    fig, ax = plt.subplots(figsize=(8, 8))
    x = np.array([x[0] for x in data_x.numpy()])
    expected_y = np.array([y[0] for y in expected_y.numpy()])
    x_min, x_max = [np.min(x), np.max(x)]
    x_lin = np.linspace(x_min, x_max, 500)
    y_lin = list(map(lambda x: valueFunc([x]), x_lin))
    ax.plot(x_lin, y_lin, color="k",
            linewidth=3, label="valueFunc")
    for s in series:
        ax.scatter(s["data_x"], s["data_y"],
                   label=s["label"], marker=s["marker"])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    fig.show()
