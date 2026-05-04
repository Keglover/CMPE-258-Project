from Models.MalConv2 import MalConv
from torch.utils.data import DataLoader

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleTrainingLoop():

    def __init__(self, model : MalConv.MalConv, train_loader : DataLoader, test_loader : DataLoader, optim, max_epochs, criterion = nn.BCEWithLogitsLoss):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optim
        self.max_epochs = max_epochs

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, y in self.train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            self.optimizer.zero_grad()
            output = self.model(x)
            logits = output[0] if isinstance(output, tuple) else output

            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

        return total_loss / total_samples, total_correct / total_samples

    @torch.no_grad()
    def _run_eval(self):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, y in self.test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            output = self.model(x)
            logits = output[0] if isinstance(output, tuple) else output

            loss = self.criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

        return total_loss / total_samples, total_correct / total_samples


    def train(self):
        loss_prev = 99999999

        for epoch in range(self.max_epochs):
            train_loss, train_acc = self._train_one_epoch()
            test_loss, test_acc = self._run_eval()

            print(
                f"Epoch {epoch+1}/{self.max_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
            )

            if train_loss > loss_prev:
                print(f"Previous loss ({loss_prev}) is less than current epoch loss ({loss}). Stopping early to circumvent overfitting")
                break

            if train_loss <= 0.0001:
                print("Loss has effectively reached 0 - there is no point to continue training.")
                break

        #test_loss, test_acc = run_eval(model, test_loader)
        #print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
