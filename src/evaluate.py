"""Model evaluation on a given dataset."""

import torch


def evaluate(model, criterion, testloader, device):
    """Evaluate the model on the given dataset."""
    loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in testloader:
            data = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(data)
            loss += criterion(outputs.view(-1), labels).item()
    loss /= len(testloader.dataset)
    return loss
