"""
    After some consideration, I may not need this file for the project. The reason being
    that, given the research-like nature of the project, experimenting with the training
    loop to better model adversarial environments, it would not make sense to create my
    own models, but rather use existing frameworks, putting them through the experimental
    training, and evaluating them against their traditionally trained counterparts. I may
    still need to make an adversary, but I do not need one for the defender.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from typing import Optional


class Classifier(nn.Module):
    
    """
    Defender agent: MalConv-inspired raw-byte malware classifier.

    Operates on raw PE byte sequences represented as integer token IDs (0-255),
    passing them through an embedding layer, gated 1D convolutions, global
    temporal max pooling, and fully connected classification layers.

    Encapsulates model construction, forward pass, single training step,
    and evaluation. The outer co-training loop lives in the pipeline, not here.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        embed_dim: int       = 8,
        num_filters: int     = 128,
        kernel_size: int     = 512,
        fc_hidden_dim: int   = 128,
        lr: float            = 1e-3,
        optimizer: str       = "Adam",
        weight_decay: float  = 0.0,
        grad_clip: float     = 1.0,
    ):
        """
        Args:
            embed_dim:      Dimensionality of byte embedding vectors.
            num_filters:    Number of convolutional filters (output channels).
            kernel_size:    Width of the 1D convolution kernel.
            fc_hidden_dim:  Hidden units in the fully connected classifier head.
            lr:             Learning rate.
            optimizer:      Optimizer name; one of {"Adam", "AdamW"}.
            weight_decay:   L2 regularization coefficient.
            grad_clip:      Max norm for gradient clipping (applied in train_step).
        """
        super(Classifier, self).__init__()

        self.lr          = lr
        self.weight_decay = weight_decay
        self.grad_clip   = grad_clip

        self._build_model(embed_dim, num_filters, kernel_size, fc_hidden_dim)

        # Optimizer is constructed after layers exist so parameters are available.
        self.optimizer = self._build_optimizer(optimizer)

        # Track current training loss for logging/pipeline access.
        self.last_loss: Optional[float] = None

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

        self.pooling = nn.MaxPool1d(
            in_channels=embed_dim,
            out_channels="",
            kernel_size=kernel_size,
            stride=kernel_size
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

    def _build_optimizer(self, optimizer_name: str) -> optim.Optimizer:
        """
        Instantiates the optimizer over all trainable parameters.

        Args:
            optimizer_name: One of {"Adam", "AdamW"}.

        Returns:
            Configured optimizer instance.

        Raises:
            ValueError: If an unsupported optimizer name is provided.
        """
        params = self.parameters()

        if optimizer_name == "Adam":
            return optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        if optimizer_name == "AdamW":
            return optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}'. "
            f"Expected one of: 'Adam', 'AdamW'."
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, x: torch.Tensor, labels: torch.Tensor, loss_fn: nn.Module) -> float:
        """
        Performs a single supervised training step: forward, loss, backward,
        gradient clip, optimizer step.

        The co-training loop controls when this is called vs. when the
        defender is frozen. requires_grad toggling lives in the pipeline.

        Args:
            x:        Byte sequence batch, shape (batch_size, seq_len).
            labels:   Binary labels, shape (batch_size,), dtype float.
            loss_fn:  Loss function; expected BCEWithLogitsLoss.

        Returns:
            Scalar loss value for logging.
        """
        self.train()
        self.optimizer.zero_grad()

        logits = self.forward(x)
        loss = loss_fn(logits, labels)
        loss.backward()

        # Gradient clipping guards against instability during co-training,
        # where loss surfaces can shift abruptly between rounds.
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)

        self.optimizer.step()

        self.last_loss = loss.item()
        return self.last_loss

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, x: torch.Tensor, labels: torch.Tensor, loss_fn: nn.Module) -> dict:
        """
        Evaluates the defender on a batch without updating weights.

        Args:
            x:        Byte sequence batch, shape (batch_size, seq_len).
            labels:   Binary labels, shape (batch_size,), dtype float.
            loss_fn:  Loss function; expected BCEWithLogitsLoss.

        Returns:
            Dict with keys: "loss", "accuracy", "tp", "fp", "tn", "fn".
        """
        self.eval()

        logits = self.forward(x)
        loss = loss_fn(logits, labels)

        preds = (torch.sigmoid(logits) >= 0.5).float()

        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        total    = labels.numel()
        accuracy = (tp + tn) / total if total > 0 else 0.0

        return {"loss": loss.item(), "accuracy": accuracy, "tp": tp, "tn": tn, "fp": fp, "fn": fn}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.Sigmoid(self.forward(x))

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path="../../Models/Checkpoints") -> None:
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

class Adversary(nn.Module):
    def __init__(        
        self,
        embed_dim: int       = 8,
        num_filters: int     = 128,
        kernel_size: int     = 512,
        fc_hidden_dim: int   = 128,
        lr: float            = 1e-3,
        optimizer: str       = "Adam",
        weight_decay: float  = 0.0,
        grad_clip: float     = 1.0
    ):

        super(Adversary, self).__init__()

        self.lr          = lr
        self.weight_decay = weight_decay
        self.grad_clip   = grad_clip

        self._build_model(embed_dim, num_filters, kernel_size, fc_hidden_dim)

        # Optimizer is constructed after layers exist so parameters are available.
        self.optimizer = self._build_optimizer(optimizer)

        # Track current training loss for logging/pipeline access.
        self.last_loss: Optional[float] = None

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

    def _build_optimizer(self, optimizer_name: str) -> optim.Optimizer:
        """
        Instantiates the optimizer over all trainable parameters.

        Args:
            optimizer_name: One of {"Adam", "AdamW"}.

        Returns:
            Configured optimizer instance.

        Raises:
            ValueError: If an unsupported optimizer name is provided.
        """
        params = self.parameters()

        if optimizer_name == "Adam":
            return optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        if optimizer_name == "AdamW":
            return optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}'. "
            f"Expected one of: 'Adam', 'AdamW'."
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    #TODO: Not using byte-valid perturbations. Find out how to do this.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates adversarial perturbations in embedding space.
        Note that this currently is NOT a byte-valid perturbation

        Args:
            x: Integer byte sequence, shape (batch_size, seq_len)

        Returns:
            Perturbed embedding, shape (batch_size, embed_dim, seq_len)
            — ready to pass directly into defender's post-embedding layers.
        """
        # Embed the input into continuous space, same as defender.
        embedded = self.embedding(x)           # (batch, seq_len, embed_dim)
        embedded = embedded.transpose(1, 2)    # (batch, embed_dim, seq_len)

        # Generate a perturbation delta of the same shape.
        delta = self.perturbation_network(embedded)

        # Bound the perturbation within [-ε, ε] via tanh scaling.
        # This replaces a hard clamp, which has zero gradient at the boundaries.
        delta = self.epsilon * torch.tanh(delta)

        return embedded + delta

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, x: torch.Tensor, labels: torch.Tensor, loss_fn: nn.Module, defender: Classifier) -> float:
        """
        Performs a single supervised training step: forward, loss, backward,
        gradient clip, optimizer step.

        The co-training loop controls when this is called vs. when the
        defender is frozen. requires_grad toggling lives in the pipeline.

        Args:
            x:        Byte sequence batch, shape (batch_size, seq_len).
            labels:   Binary labels, shape (batch_size,), dtype float.
            loss_fn:  Loss function; expected BCEWithLogitsLoss.

        Returns:
            Scalar loss value for logging.
        """
        self.train()
        self.optimizer.zero_grad()

        evasive_perturbation = self.forward(x)

        defender_logits = defender.forward(evasive_perturbation)
        loss_negated = -loss_fn(defender_logits, labels)   # We want to maximize loss, not minimize it. No label change either
        loss_negated.backward()

        # Gradient clipping guards against instability during co-training,
        # where loss surfaces can shift abruptly between rounds.
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)

        self.optimizer.step()

        self.last_loss = loss_negated.item()
        return self.last_loss

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    # TODO: Currently using defender evaluate() method - MUST MODIFY FOR ATTACKER
    # ASR = (# of successful attacks / # of attempted attacks) --> will need to track # of attacks some how, but easy to track # successful attacks after
    @torch.no_grad()
    def evaluate(self, x: torch.Tensor, labels: torch.Tensor, loss_fn: nn.Module) -> dict:
        """
        Evaluates the defender on a batch without updating weights.

        Args:
            x:        Byte sequence batch, shape (batch_size, seq_len).
            labels:   Binary labels, shape (batch_size,), dtype float.
            loss_fn:  Loss function; expected BCEWithLogitsLoss.

        Returns:
            Dict with keys: "loss", "accuracy", "tp", "fp", "tn", "fn".
        """
        self.eval()

        logits = self.forward(x)
        loss = loss_fn(logits, labels)

        preds = (torch.sigmoid(logits) >= 0.5).float()

        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        total    = labels.numel()
        accuracy = (tp + tn) / total if total > 0 else 0.0

        return {}

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
