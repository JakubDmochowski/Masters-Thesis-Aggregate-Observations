import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import torch
from sklearn import metrics

PLOTSIZE = (6, 6)
PLOTSIZE_SM = (4, 4)


def plot_losses(loss_history: list[list[float]]):
    fig, ax = plt.subplots(figsize=PLOTSIZE)
    x = range(0, len(loss_history))
    models = loss_history[0].keys()
    for model in models:
        history = [losses[model].detach().numpy() for losses in loss_history]
        ax.plot(x, history, label=model)
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.legend()
    plt.yscale("log")
    fig.show()


def plot_xy(data_x: torch.tensor, expected_y: torch.tensor, series: list[dict], value_func: Callable = None):
    fig, ax = plt.subplots(figsize=PLOTSIZE)
    if value_func is not None:
        x = np.array([x[0] for x in data_x.numpy()])
        expected_y = np.array([y[0] for y in expected_y.numpy()])
        x_min, x_max = [np.min(x), np.max(x)]
        x_lin = np.linspace(x_min, x_max, 500)
        y_lin = list(map(lambda x: value_func([x]), x_lin))
        ax.plot(x_lin, y_lin, color="k",
                linewidth=3, label="value_func")
    for s in series:
        ax.scatter(s["data_x"], s["data_y"],
                   label=s["label"], marker=s["marker"])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    fig.show()


def plot_xy3d(data_x: torch.tensor, expected_y: torch.tensor, series: list[dict], value_func: Callable = None):
    fig = plt.figure(figsize=PLOTSIZE)
    ax = fig.add_subplot(projection='3d')
    DETAIL = 15
    if value_func is not None:
        x = np.array([x[0] for x in data_x.numpy()])
        x_min, x_max = [np.min(x), np.max(x)]
        x_lin = np.linspace(x_min, x_max, DETAIL)
        y = np.array([], dtype=float)
        for y_ in x_lin:
            y = np.concatenate((y, np.array(list(map(lambda x: value_func([x, y_])[0], x_lin)))))
        x1, x2 = np.meshgrid(x_lin, x_lin)
        y = y.reshape((DETAIL, DETAIL))
        ax.contour(x1, x2, y, zdir='z', offset=0)
        ax.contour(x1, x2, y, zdir='x', offset=10)
        ax.contour(x1, x2, y, zdir='y', offset=10)
    for s in series:
        ax.scatter(s["data_x"][:, 0], s["data_x"][:, 1], 0,
                   label=s["label"], marker=s["marker"])
        ax.scatter(10, s["data_x"][:, 1], s["data_y"][:, 0],
                   label=s["label"], marker=s["marker"])
        ax.scatter(s["data_x"][:, 0], 10, s["data_y"][:, 0],
                   label=s["label"], marker=s["marker"])
    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    ax.set_zlabel('z')
    fig.show()


def plot_roc(targets, predictions, title):
    fig, ax = plt.subplots(figsize=PLOTSIZE)
    fpr, tpr, _ = metrics.roc_curve(
        targets.reshape(-1), predictions.reshape(-1))
    ax.plot(fpr, tpr)
    ax.set_xlabel('True Positive Rate')
    ax.set_ylabel('False Positive Rate')
    ax.set_title(title)
    fig.show()


def plot_auc(models, targets, every):
    fig, ax = plt.subplots(figsize=PLOTSIZE)
    for model in models:
        auc_history = []
        for index, predictions in enumerate(model["prediction_history"]):
            auc_history.append([index * every, metrics.roc_auc_score(
                targets.reshape(-1), predictions.reshape(-1))])
        auc_history = np.array(auc_history)
        ax.plot(auc_history[:, 0], auc_history[:, 1], label=model["label"])
    ax.set_xlabel('iteration')
    ax.set_ylabel('AUC Score')
    ax.set_title(f"AUC")
    ax.legend()
    fig.show()


def plot_precision(models, targets, every):
    fig, ax = plt.subplots(figsize=PLOTSIZE)
    for model in models:
        precision_history = []
        for index, predictions in enumerate(model["prediction_history"]):
            predictions = torch.tensor(np.array(
                list(map(lambda x: x.round(), predictions.numpy()))), dtype=torch.int)
            precision = metrics.precision_score(
                targets.reshape(-1), predictions.reshape(-1))
            precision_history.append([index * every, precision])
        precision_history = np.array(precision_history)
        ax.plot(precision_history[:, 0],
                precision_history[:, 1], label=model["label"])
    ax.set_xlabel('iteration')
    ax.set_ylabel('Precision Score')
    ax.set_title(f"Precision")
    ax.legend()
    fig.show()


def plot_recall(models, targets, every):
    fig, ax = plt.subplots(figsize=PLOTSIZE)
    for model in models:
        recall_history = []
        for index, predictions in enumerate(model["prediction_history"]):
            predictions = torch.tensor(np.array(
                list(map(lambda x: x.round(), predictions.numpy()))), dtype=torch.int)
            recall = metrics.recall_score(
                targets[:, 0].reshape(-1), predictions[:, 0].reshape(-1))
            recall_history.append([index * every, recall])
        recall_history = np.array(recall_history)
        ax.plot(recall_history[:, 0],
                recall_history[:, 1], label=model["label"])
    ax.set_xlabel('iteration')
    ax.set_ylabel('Recall Score')
    ax.set_title(f"Recall")
    ax.legend()
    fig.show()


def plot_confusion_matrix(models, targets):
    for model in models:
        fig, ax = plt.subplots(figsize=PLOTSIZE_SM)
        predictions = model["prediction_history"][len(
            model["prediction_history"]) - 1]
        predictions = torch.tensor(np.array(
            list(map(lambda x: x.round(), predictions.numpy()))), dtype=torch.int)
        metrics.ConfusionMatrixDisplay.from_predictions(
            targets[:, 0].reshape(-1), predictions[:, 0].reshape(-1), ax=ax)
        ax.set_title(model["label"])
        fig.show()
