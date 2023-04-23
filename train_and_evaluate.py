# built-in
from typing import Tuple

# installed
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

# local
from utils.models import RNN


def get_loss_and_optimizer(model: RNN) -> Tuple[torch.nn.BCELoss, torch.optim.Adam]:
    """Set your loss function and optimizer

    Parameters
    ----------
    model: RNN
        PyTorch model

    Returns
    -------
    loss_fn : torch.nn.BCELoss
        PyTorch loss function
    optimizer : torch.optim.Adam
        PyTorch optimizer
    """
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)  # 0.001

    return loss_fn, optimizer


def train_model_for_one_epoch(train_dataloader: DataLoader, model: RNN, loss_fn: torch.nn.BCELoss,
                              optimizer: torch.optim.Adam) -> Tuple[np.float, float, np.float]:
    """Training routine for one epoch

    Parameters
    ----------
    train_dataloader : DataLoader
    model : RNN
    loss_fn : torch.nn.BCELoss
    optimizer : torch.optim.Adam

    Returns
    -------
    f1_batch : np.float
        Considering the loss of every single sample
    total_loss : float
        Considering the loss of every single sample
    batch_losses : np.float
        Considering the loss of every single batch
    """
    model.train()
    f1_batch, total_loss, batch_losses = 0, 0, 0

    for sentences_batch, label_batch, lengths in train_dataloader:
        optimizer.zero_grad()
        pred = model(sentences_batch, lengths)[:, 0]
        loss = loss_fn(pred, label_batch)
        loss.backward()
        optimizer.step()

        # computing f1_score and loss
        pred = (pred >= 0.5).float().cpu().detach().numpy()
        y_true = label_batch.cpu().detach().numpy()
        total_loss += loss.item() * label_batch.size(0)
        f1_batch += f1_score(y_true, pred)

        # collect losses for plotting
        loss_np = loss.detach().numpy()
        batch_losses += loss_np

    return (f1_batch / len(train_dataloader)), (total_loss / len(train_dataloader.dataset)), \
           (batch_losses / len(train_dataloader))


def evaluate_model(test_dataloader, model):
    """Model evaluation using F1-Score and Confusion Matrix"""

    model.eval()
    all_pred = []
    all_y_true = []

    with torch.no_grad():
        for text_batch, label_batch, lengths in test_dataloader:
            pred = model(text_batch, lengths)[:, 0]

            # computing f1_score
            pred = (pred >= 0.5).float().cpu().detach().numpy()
            all_pred.extend(pred)
            y_true = label_batch.cpu().detach().numpy()
            all_y_true.extend(y_true)

    # plotting confusion matrix
    sns.heatmap(confusion_matrix(all_y_true, all_pred), annot=True, fmt='g')

    return f1_score(all_y_true, all_pred)
