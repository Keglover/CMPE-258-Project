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
                print(f"Previous loss ({loss_prev}) is less than current epoch loss ({train_loss}). Stopping early to circumvent overfitting")
                break

            if train_loss <= 0.0001:
                print("Loss has effectively reached 0 - there is no point to continue training.")
                break

        #test_loss, test_acc = run_eval(model, test_loader)
        #print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

class Adversarial_Loop:
    def __init__(self, defender, adversary, warm_start_epochs,
                 train_loader: DataLoader, test_loader: DataLoader,
                 max_epochs: int = 10, criterion = nn.BCEWithLogitsLoss()):
        self.defender = defender
        self.criterion = criterion
        self.adversary = adversary
        self.warm_start = warm_start_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.max_epochs = max_epochs
        self.defender_token = "def" # Just to help with disambiguation and magic strings
        self.adversary_token = "adv"

    def _train_defender(self, mode: str):
        self.defender.to(DEVICE)
        self.adversary.eval()

        total_loss = 0.0
        total_acc  = 0.0
        total_samples = 0
        total_tp = 0
        total_tn = 0
        total_fp = 0
        total_fn = 0

        loader = None

        if mode.lower() == 'train':
            loader = self.train_loader
        elif mode.lower() == 'test':
            loader = self.test_loader
        else:
            raise IndexError(f"No such mode {mode.lower()}")

        for x, y in loader:
            self.defender.zero_grad()

            batch_perf = None
            if mode.lower() == 'train':
                batch_perf = self.defender.batch_eval(x, y, self.criterion)
            elif mode.lower() == 'test':
                batch_perf = self.defender.pred(x, y, self.criterion)
            else:
                raise IndexError("How did you get here?")

            batch_size = batch_perf["total"]

            total_loss += batch_perf["loss"] * batch_size
            total_acc  += batch_perf["accuracy"] * batch_size
            total_tp += batch_perf["tp"] 
            total_tn += batch_perf["tn"] 
            total_fp += batch_perf["fp"] 
            total_fn += batch_perf["fn"] 
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_acc  = total_acc / total_samples
        rate_tp  = total_tp / total_samples
        rate_fp  = total_fp / total_samples
        rate_tn  = total_tn / total_samples
        rate_fn  = total_fn / total_samples

        return {
            "loss": avg_loss, "accuracy": avg_acc, 
            "tp_rate": rate_tp, "fp_rate": rate_fp, "tn_rate": rate_tn, "fn_rate": rate_fn
        }

    def _train_adversary(self):
        self.adversary.to(DEVICE)
        self.defender.to(DEVICE)
        self.defender.eval()

        total_loss = 0.0
        total_success = 0.0
        total_batches = 0
        total_mal_probs = 0.0
        total_reward = 0.0

        for x, y in self.train_loader:
            y = y.view(-1)  # Figure out whether or not you need this set dynamically

            mal_mask = y == 1
            if mal_mask.sum() == 0: # I probably need to change this!
                continue

            x_mal = x[mal_mask]

            batch_perf = self.adversary.batch_eval(x_mal=x_mal, edit_mask=None, defender=self.defender)

            total_loss      += batch_perf["loss"]
            total_success   += batch_perf["esr"]
            total_mal_probs += batch_perf["mean_defender_malware_prob"]
            total_reward    += batch_perf["reward"]
            total_batches += 1

        return {
            "loss":             total_loss      / max(total_batches, 1),
            "esr":              total_success   / max(total_batches, 1),
            "avg_malware_prob": total_mal_probs / max(total_batches, 1),
            "avg_reward":       total_reward    / max(total_batches, 1)
        }

    def _train_one_epoch(self, model: str, mode: str = "train"):  
        if model.lower() == self.defender_token:
            return self._train_defender(mode=mode)

        elif model.lower() == self.adversary_token:
            return self._train_adversary()

        else:   # Shouldn't get here, but sometimes weird shit happens
            raise IndexError(f"Adversarial_Loop._train_one_epoch got an unexpected token: {model.lower}")

    def train(self):
        
        for epoch in range(self.max_epochs):
            def_train_performance = self._train_one_epoch(model=self.defender_token, mode='train')
            def_test_performance  = self._train_one_epoch(model=self.defender_token, mode='test')
            adv_train_performance = None
            
            if epoch >= self.warm_start:
                adv_train_performance = self._train_one_epoch(self.adversary_token)

            # Defender metrics
            def_train_loss = def_train_performance["loss"]
            def_train_acc  = def_train_performance["accuracy"]
            def_train_tp_rate = def_train_performance["tp_rate"]
            def_train_tn_rate = def_train_performance["tn_rate"]
            def_train_fp_rate = def_train_performance["fp_rate"]
            def_train_fn_rate = def_train_performance["fn_rate"]

            def_test_loss = def_test_performance["loss"]
            def_test_acc  = def_test_performance["accuracy"]
            def_test_tp_rate = def_test_performance["tp_rate"]
            def_test_tn_rate = def_test_performance["tn_rate"]
            def_test_fp_rate = def_test_performance["fp_rate"]
            def_test_fn_rate = def_test_performance["fn_rate"]

            print(
                f"Epoch {epoch+1}/{self.max_epochs} | "
                f"\n"
                f"==== DEFENDER METRICS ===="
                f"\n"
                f"Train Loss: {def_train_loss:.4f}, Train Acc: {def_train_acc:.4f} | Train TP Rate: {def_train_tp_rate:.4f}, Train FP Rate: {def_train_fp_rate:.4f}, Train TN Rate: {def_train_tn_rate:.4f}, Train FN Rate: {def_train_fn_rate:.4f} |"
                f"\n"
                f"Test Loss: {def_test_loss:.4f}, Test Acc: {def_test_acc:.4f} | Test TP Rate: {def_test_tp_rate:.4f}, Test FP Rate: {def_test_fp_rate:.4f}, Test TN Rate: {def_test_tn_rate:.4f}, Test FN Rate: {def_test_fn_rate:.4f} |"
                f"\n"
            )

            if adv_train_performance is not None:

                adv_loss       = adv_train_performance["loss"]
                adv_esr        = adv_train_performance['esr']
                adv_avg_prob   = adv_train_performance["avg_malware_prob"]
                adv_avg_reward = adv_train_performance["avg_reward"]

                print(
                    f"==== ADVERSARY METRICS ===="
                    f"\n"
                    f"Loss: {adv_loss:.4f}, ASR: {adv_esr:.4f} |"
                    f"\n"
                    f"Average Malware Probability: {adv_avg_prob:.4f}, Average Reward: {adv_avg_reward:.4f} |"
                    f"\n"
                )
