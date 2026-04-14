"""
config.py
---------
Loads hyperparameters.json, validates all fields, and exposes a single
PipelineConfig dataclass used throughout the pipeline.

Corresponds to the "Verify hyperparameters" node in the system diagram.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class AgentModelsConfig:
    defender: str
    evader: str

    _VALID_DEFENDERS = {"malconv"}
    _VALID_EVADERS   = {"byte_appender"}

    def validate(self) -> None:
        if self.defender not in self._VALID_DEFENDERS:
            raise ValueError(
                f"Unknown defender model '{self.defender}'. "
                f"Valid options: {self._VALID_DEFENDERS}"
            )
        if self.evader not in self._VALID_EVADERS:
            raise ValueError(
                f"Unknown evader model '{self.evader}'. "
                f"Valid options: {self._VALID_EVADERS}"
            )


@dataclass
class DataConfig:
    input_size_mb:  float
    split_train:    float
    split_val:      float
    split_test:     float
    batch_size:     int

    # Derived — computed once during validation
    input_size_bytes: int = field(init=False, default=0)

    def validate(self) -> None:
        if self.input_size_mb <= 0:
            raise ValueError("input_size_mb must be > 0")

        split_sum = round(self.split_train + self.split_val + self.split_test, 8)
        if not math.isclose(split_sum, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"split_train + split_val + split_test must sum to 1.0, "
                f"got {split_sum}"
            )
        for name, val in [
            ("split_train", self.split_train),
            ("split_val",   self.split_val),
            ("split_test",  self.split_test),
        ]:
            if not (0.0 < val < 1.0):
                raise ValueError(f"{name} must be in (0, 1), got {val}")

        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        # Compute derived field
        self.input_size_bytes = int(self.input_size_mb * 1024 * 1024)


@dataclass
class TrainingConfig:
    epochs:            int
    warm_start_len:    int
    num_cotrain_rounds: int
    update_freq:       int

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.warm_start_len < 1:
            raise ValueError("warm_start_len must be >= 1")
        if self.warm_start_len >= self.epochs:
            raise ValueError(
                f"warm_start_len ({self.warm_start_len}) must be < "
                f"epochs ({self.epochs})"
            )
        if self.num_cotrain_rounds < 1:
            raise ValueError("num_cotrain_rounds must be >= 1")
        if self.update_freq < 1:
            raise ValueError("update_freq must be >= 1")


@dataclass
class OptimizerConfig:
    lr:         float
    optim:      str
    weight_reg: Optional[float]
    grad_clip:  Optional[float]

    _VALID_OPTIMS = {"adam", "sgd", "adamw"}

    def validate(self, label: str = "optimizer") -> None:
        if self.lr <= 0:
            raise ValueError(f"{label}: lr must be > 0, got {self.lr}")
        if self.optim not in self._VALID_OPTIMS:
            raise ValueError(
                f"{label}: unknown optimizer '{self.optim}'. "
                f"Valid: {self._VALID_OPTIMS}"
            )
        if self.weight_reg is not None and self.weight_reg < 0:
            raise ValueError(f"{label}: weight_reg must be >= 0")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError(f"{label}: grad_clip must be > 0")


@dataclass
class AdversarialConfig:
    perturb_budget: int
    perturb_mode:   str

    _VALID_MODES = {"append", "replace", "insert"}

    def validate(self) -> None:
        if self.perturb_budget < 1:
            raise ValueError("perturb_budget must be >= 1 byte")
        if self.perturb_mode not in self._VALID_MODES:
            raise ValueError(
                f"Unknown perturb_mode '{self.perturb_mode}'. "
                f"Valid: {self._VALID_MODES}"
            )


@dataclass
class EvaluationConfig:
    ASR_WL_thresh:     float
    eval_freq:         int
    checkpoint_metric: str

    _VALID_METRICS = {"robust_accuracy", "clean_accuracy", "asr", "f1"}

    def validate(self) -> None:
        if not (0.0 < self.ASR_WL_thresh < 1.0):
            raise ValueError("ASR_WL_thresh must be in (0, 1)")
        if self.eval_freq < 1:
            raise ValueError("eval_freq must be >= 1")
        if self.checkpoint_metric not in self._VALID_METRICS:
            raise ValueError(
                f"Unknown checkpoint_metric '{self.checkpoint_metric}'. "
                f"Valid: {self._VALID_METRICS}"
            )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Fully-validated pipeline configuration.

    Instantiate via PipelineConfig.from_json(path) rather than directly.
    """
    agents:           AgentModelsConfig
    data:             DataConfig
    training:         TrainingConfig
    defender_optim:   OptimizerConfig
    evader_optim:     OptimizerConfig
    adversarial:      AdversarialConfig
    evaluation:       EvaluationConfig

    # Populated by from_json for traceability
    source_path: Path = field(default=Path(""), compare=False)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        """
        Load and fully validate a hyperparameters.json file.

        Raises
        ------
        FileNotFoundError
            If the JSON file does not exist.
        KeyError
            If a required field is missing from the JSON.
        ValueError
            If any field fails semantic validation.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Hyperparameter file not found: {path}")

        with path.open("r") as fh:
            raw = json.load(fh)

        cfg = cls(
            agents=AgentModelsConfig(
                defender=raw["agent_models"]["defender"],
                evader=raw["agent_models"]["evader"],
            ),
            data=DataConfig(
                input_size_mb=raw["data"]["input_size_mb"],
                split_train=raw["data"]["split_train"],
                split_val=raw["data"]["split_val"],
                split_test=raw["data"]["split_test"],
                batch_size=raw["data"]["batch_size"],
            ),
            training=TrainingConfig(
                epochs=raw["training"]["epochs"],
                warm_start_len=raw["training"]["warm_start_len"],
                num_cotrain_rounds=raw["training"]["num_cotrain_rounds"],
                update_freq=raw["training"]["update_freq"],
            ),
            defender_optim=OptimizerConfig(
                lr=raw["defender_optimizer"]["lr"],
                optim=raw["defender_optimizer"]["optim"],
                weight_reg=raw["defender_optimizer"]["weight_reg"],
                grad_clip=raw["defender_optimizer"]["grad_clip"],
            ),
            evader_optim=OptimizerConfig(
                lr=raw["evader_optimizer"]["lr"],
                optim=raw["evader_optimizer"]["optim"],
                weight_reg=raw["evader_optimizer"]["weight_reg"],
                grad_clip=raw["evader_optimizer"]["grad_clip"],
            ),
            adversarial=AdversarialConfig(
                perturb_budget=raw["adversarial"]["perturb_budget"],
                perturb_mode=raw["adversarial"]["perturb_mode"],
            ),
            evaluation=EvaluationConfig(
                ASR_WL_thresh=raw["evaluation"]["ASR_WL_thresh"],
                eval_freq=raw["evaluation"]["eval_freq"],
                checkpoint_metric=raw["evaluation"]["checkpoint_metric"],
            ),
            source_path=path,
        )

        cfg._validate_all()
        return cfg

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_all(self) -> None:
        self.agents.validate()
        self.data.validate()           # also populates input_size_bytes
        self.training.validate()
        self.defender_optim.validate("defender_optimizer")
        self.evader_optim.validate("evader_optimizer")
        self.adversarial.validate()
        self.evaluation.validate()
        self._cross_validate()

    def _cross_validate(self) -> None:
        """Checks that span multiple sub-configs."""
        # Perturbation budget must fit inside the input window
        if self.adversarial.perturb_budget >= self.data.input_size_bytes:
            raise ValueError(
                f"perturb_budget ({self.adversarial.perturb_budget} bytes) "
                f"must be < input_size_bytes ({self.data.input_size_bytes} bytes)"
            )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def input_size_bytes(self) -> int:
        """Shortcut so call sites don't need to reach into .data."""
        return self.data.input_size_bytes

    def __repr__(self) -> str:  # pragma: no cover
        lines = [
            "PipelineConfig:",
            f"  defender       : {self.agents.defender}",
            f"  evader         : {self.agents.evader}",
            f"  input_size     : {self.data.input_size_mb} MB "
            f"({self.data.input_size_bytes:,} bytes)",
            f"  batch_size     : {self.data.batch_size}",
            f"  epochs         : {self.training.epochs}",
            f"  warm_start_len : {self.training.warm_start_len}",
            f"  cotrain_rounds : {self.training.num_cotrain_rounds}",
            f"  perturb_budget : {self.adversarial.perturb_budget} bytes "
            f"({self.adversarial.perturb_mode})",
            f"  ASR threshold  : {self.evaluation.ASR_WL_thresh}",
        ]
        return "\n".join(lines)
