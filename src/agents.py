import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = int(2.048 * 10**6)    # 2 MB max input or output size

'''
    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Saves model and optimizer state for resuming co-training rounds."""
        torch.save(
            {
                "model_state":     self.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "last_loss":       self.last_loss,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Restores model and optimizer state from a checkpoint."""
        checkpoint = torch.load(path, weights_only=True)
        self.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.last_loss = checkpoint.get("last_loss")
'''

class Defender(nn.Module):
    def __init__(self, optim_name: str = "Adam", lr=1e-4, weight_decay=0.0, emb_dim=32, hidden_dim=128):
        super(Defender, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay

        self._build_model(embed_dim=emb_dim, num_filters=128, kernel_size=128, fc_hidden_dim=hidden_dim)
        self.optim = self._build_optim(optimizer_name=optim_name)

    def _build_model(self, embed_dim: int, num_filters: int, kernel_size: int, fc_hidden_dim: int) -> None:
        """
        Constructs layers following a MalConv-style architecture:
            Embedding → Gated Conv1d → Global Max Pool → FC head

        Gating: two parallel convolutions produce a value stream and a gate
        stream; their elementwise product (with sigmoid gate) is the output.
        This is MalConv's mechanism for attending to relevant byte windows.
        """
        # Byte vocabulary is fixed: 256 possible values (0x00–0xFF), plus
        # index 256 reserved for the padding token.
        vocab_size = 257
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=256,
        )

        # Gated convolution: value and gate branches share the same config.
        # Input channels = embed_dim; Conv1d expects (batch, channels, length).
        self.conv_value = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=kernel_size,  # Non-overlapping windows, matching MalConv.
            bias=True,
        )

        self.conv_gate = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

        # Global temporal max pool collapses the sequence dimension entirely,
        # making the model input-length agnostic (critical for variable-size PE).
        # Implemented in forward() via torch.max rather than a layer.

        # Classifier head: gated conv output → hidden → binary logit.
        self.fc1 = nn.Linear(num_filters, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, 1)  # Single logit for BCEWithLogitsLoss.

    def _build_optim(self, optimizer_name):
        if optimizer_name == "Adam":
            return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def optimizer_step(self):
        self.optim.step()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over a batch of raw byte sequences.

        Args:
            x: Integer tensor of shape (batch_size, sequence_length),
               values in [0, 256] where 256 is the padding token.

        Returns:
            Logits tensor of shape (batch_size,). Pass through sigmoid
            for probabilities; use directly with BCEWithLogitsLoss.
        """
        if x.dtype is not torch.long:
            x = x.long()

        if x.device is not DEVICE:
            x = x.to(DEVICE)

        # (batch, seq_len) → (batch, seq_len, embed_dim)
        x = self.embedding(x)

        # Conv1d expects (batch, channels, length).
        x = x.transpose(1, 2)  # → (batch, embed_dim, seq_len)

        # Gated activation: value * sigmoid(gate)
        value = self.conv_value(x)           # (batch, num_filters, windows)
        gate  = torch.sigmoid(self.conv_gate(x))
        x = value * gate                     # (batch, num_filters, windows)

        # Global temporal max pool → (batch, num_filters)
        x, _ = torch.max(x, dim=2)

        # Classifier head
        x = F.relu(self.fc1(x))              # (batch, fc_hidden_dim)
        x = self.fc2(x)                      # (batch, 1)

        return x.squeeze(1)                  # (batch,)

    def batch_eval(self, x: torch.Tensor, labels: torch.Tensor, loss_fn: nn.Module) -> dict:
        """
        Trains and evaluates the defender on a batch.

        Args:
            x:        Byte sequence batch, shape (batch_size, seq_len).
            labels:   Binary labels, shape (batch_size,), dtype float.
            loss_fn:  Loss function; expected BCEWithLogitsLoss.

        Returns:
            Dict with keys: "loss", "accuracy", "tp", "fp", "tn", "fn".
        """

        self.train()

        if x.dtype is not torch.long:
            x = x.long()
        if x.device is not DEVICE:
            x = x.to(DEVICE)

        if labels.dtype is not torch.float:
            labels = labels.float().view(-1)
        if labels.device is not DEVICE:
            labels = labels.to(DEVICE)

        self.optim.zero_grad()

        logits = self.forward(x)
        loss = loss_fn(logits, labels)

        loss.backward()
        self.optim.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) >= 0.5).float()

            tp = ((preds == 1) & (labels == 1)).sum().item()
            tn = ((preds == 0) & (labels == 0)).sum().item()
            fp = ((preds == 1) & (labels == 0)).sum().item()
            fn = ((preds == 0) & (labels == 1)).sum().item()

            total    = labels.numel()
            accuracy = (tp + tn) / total if total > 0 else 0.0

        return {"total": total, "loss": loss.item(), "accuracy": accuracy, "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    
    def pred(self, x: torch.Tensor, labels: torch.Tensor, loss_fn: nn.Module) -> dict:
        """
        Runs batch_eval for running testing batches. Does not update weights. And needed to remove self.backward()

        Args:
            Same as self.batch_eval()

        Returns:
            Same as self.batch_eval()
        """

        with torch.no_grad():
            self.eval()

            if x.dtype is not torch.long:
                x = x.long()
            if x.device is not DEVICE:
                x = x.to(DEVICE)

            if labels.dtype is not torch.float:
                labels = labels.float().view(-1)
            if labels.device is not DEVICE:
                labels = labels.to(DEVICE)

            logits = self.forward(x)
            loss = loss_fn(logits, labels)
            preds = (torch.sigmoid(logits) >= 0.5).float()

            tp = ((preds == 1) & (labels == 1)).sum().item()
            tn = ((preds == 0) & (labels == 0)).sum().item()
            fp = ((preds == 1) & (labels == 0)).sum().item()
            fn = ((preds == 0) & (labels == 1)).sum().item()

            total    = labels.numel()
            accuracy = (tp + tn) / total if total > 0 else 0.0

            return {"total": total, "loss": loss.item(), "accuracy": accuracy, "tp": tp, "tn": tn, "fp": fp, "fn": fn}

class CNNAttacker(nn.Module):
    def __init__(self, optim_name: str ="Adam", lr=1e-4, weight_decay=0.0, vocab_size=257, emb_dim=8, hidden_dim=128, adv_len=256, output_vocab_size=256):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.adv_len = adv_len
        self.output_size = output_vocab_size
        self.byte_emb = nn.Embedding(vocab_size, emb_dim)

        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim + 1, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, adv_len * output_vocab_size)
        )

        self.optim = self._bulid_optim(optim_name)

    @torch.no_grad()
    def _apply_adv_bytes(self, x_mal, adv_bytes, editable_mask) -> torch.Tensor:
        """
        x_mal:      (B, T)
        adv_logits: (B, L_adv, 256)

        Returns:
            x_adv: (B, T + L_adv) or padded/truncated to defender input length
        """
        
        x_adv = torch.cat([x_mal, adv_bytes], dim=1)    # (B, L_adv)

        # Optional: truncate/pad if defender expects fixed length
        if x_adv.size(1) > MAX_LEN:
            x_adv = x_adv[:, : MAX_LEN]
        elif x_adv.size(1) < MAX_LEN:
            pad_len = MAX_LEN - x_adv.size(1)
            pad = torch.full(
                (x_adv.size(0), pad_len), 256,
                dtype=x_adv.dtype,
                device=DEVICE
            )
            x_adv = torch.cat([x_adv, pad], dim=1)

        return x_adv.long()

    def _bulid_optim(self, optimizer_name):
        if optimizer_name == "Adam":
            return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"The optimizer called {optimizer_name} is not supported.")
    
    def optimizer_step(self):
        self.optim.step()

    def zero_grad(self):
        self.optim.zero_grad()

    def forward(self, x_bytes, editable_mask=None) -> torch.Tensor:
        x_bytes = x_bytes.to(DEVICE).long()

        x_emb = self.byte_emb(x_bytes)  # (B, T, emb_dim)

        if editable_mask is None:
            editable_mask = torch.zeros(x_bytes.shape, 
                                             device=x_bytes.device,
                                             dtype=torch.float32)

        mask_feat = editable_mask.unsqueeze(-1)     # (B, T, 1)

        x = torch.cat([x_emb, mask_feat], dim=-1)   # (B, T, emb_dim+1)
        x = x.transpose(1, 2)                       # (B, emb_dim+1, T)

        h = self.conv(x).squeeze(-1)                # (B, 128)
        logits = self.decoder(h)                    # (B, adv_len * 256)
        logits = logits.view(
            x_bytes.size(0), 
            self.adv_len, 
            self.output_size
        )

        return logits
    
    def batch_eval(self, x_mal: torch.Tensor, edit_mask: torch.Tensor, defender: Defender, loss_func=nn.BCEWithLogitsLoss()) -> dict:
        self.train()
        defender.eval()

        if x_mal.dtype is not torch.long:
            x_mal = x_mal.long()
        if x_mal.device is not DEVICE:
            x_mal = x_mal.to(DEVICE)
        
        if edit_mask is not None:
            if edit_mask.device is not DEVICE:
                edit_mask = edit_mask.to(DEVICE)
            if edit_mask.dtype is not torch.float:
                edit_mask = edit_mask.float()

        self.optim.zero_grad()
        
        adv_logits = self.forward(x_mal, edit_mask)
        dist = torch.distributions.Categorical(logits=adv_logits)

        adv_bytes = dist.sample()
        log_probs = dist.log_prob(adv_bytes)

        x_adv = self._apply_adv_bytes(x_mal, adv_bytes, edit_mask)

        with torch.no_grad():
            def_logts = defender(x_adv)
            mal_probs = torch.sigmoid(def_logts)

            reward = 1.0 - mal_probs
            esr = (mal_probs < 0.5).float().mean().item()

        sequence_log_prob = log_probs.sum(dim=1)
        loss = -(reward.detach() * sequence_log_prob).mean()

        loss.backward()
        self.optim.step()
        self.eval()

        return{
            "loss": loss.item(),
            "reward": reward.mean().item(),
            "esr": esr,
            "mean_defender_malware_prob": mal_probs.mean().item()
        }
    