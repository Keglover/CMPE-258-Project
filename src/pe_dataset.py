"""
pe_dataset.py
-------------
PyTorch Dataset for raw-byte PE malware classification.

Corresponds to the "Load datasets and apply padding/truncation to prep inputs"
node in the system diagram.

Design decisions:
  - Input representation: raw byte sequences (not pre-extracted features)
  - Fixed-length handling: truncation (head) + zero-padding (primary strategy)
  - Byte vocabulary: 0–255 (valid bytes) + 256 (PAD token)
  - Output dtype: torch.long — required for nn.Embedding lookup
  - MalConv comparability: default max_bytes == 2 MB (2,097,152)

Label convention:
  0 = benign
  1 = malicious
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


# Byte 256 is reserved as the padding token.
# Values 0–255 are real bytes; this index is out-of-range for uint8,
# so it can never appear in a real PE file and is unambiguous as padding.
PAD_TOKEN: int = 256
VOCAB_SIZE: int = 257  # 0–255 real bytes + 1 PAD token


# ---------------------------------------------------------------------------
# Core dataset
# ---------------------------------------------------------------------------

class PEBinaryDataset(Dataset):
    """
    Dataset that loads raw Windows PE files as fixed-length byte tensors.

    Parameters
    ----------
    file_paths : sequence of path-like
        Paths to PE binary files on disk.
    labels : sequence of int
        Parallel label sequence (0 = benign, 1 = malicious).
    max_bytes : int
        Maximum byte-sequence length. Files longer than this are
        truncated from the head; shorter files are right-padded with
        PAD_TOKEN (256).
    transform : callable, optional
        Optional transform applied to the raw byte Tensor before
        returning.  Receives a 1-D LongTensor of shape (max_bytes,).
    """

    def __init__(self, file_paths: Sequence[str | Path], labels: Sequence[int], max_bytes: int, transform: Optional[Callable[[Tensor], Tensor]] = None):
        if len(file_paths) != len(labels):
            raise ValueError(
                f"file_paths ({len(file_paths)}) and labels ({len(labels)}) "
                "must have the same length"
            )
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")

        self.file_paths: list[Path] = [Path(p) for p in file_paths]
        self.labels:     list[int]  = list(labels)
        self.max_bytes:  int        = max_bytes
        self.transform              = transform

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Returns
        -------
        byte_tensor : LongTensor of shape (max_bytes,)
            Byte sequence ready for nn.Embedding. Values in [0, 256].
        label_tensor : LongTensor scalar
            0 = benign, 1 = malicious.
        """
        byte_tensor = self._load_and_preprocess(self.file_paths[idx])
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform is not None:
            byte_tensor = self.transform(byte_tensor)

        return byte_tensor, label_tensor

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_and_preprocess(self, path: Path) -> Tensor:
        """
        Read raw bytes from disk, truncate or pad to max_bytes, and
        return a 1-D LongTensor.

        Truncation strategy: keep the first max_bytes bytes (head scan).
        This mirrors MalConv's original behaviour and preserves the PE
        header, which is the most semantically dense region.

        Padding strategy: right-pad with PAD_TOKEN (256). Using an
        out-of-range value rather than 0x00 lets the embedding layer
        learn a distinct padding representation and avoids conflating
        real null bytes with structural padding.
        """
        raw = _read_bytes(path, self.max_bytes)
        return _pad_or_truncate(raw, self.max_bytes)

    def filter_readable_labels(self) -> tuple[list[Path], list[int], list]:
        import csv

        good_paths, good_labels, bad_files = [], [], []

        for path, label in zip(self.file_paths, self.labels):
            path = Path(path)

            try:
                with open(path, 'rb') as fh:
                    fh.read(1)     # Only need to be able to open the file
                good_paths.append(path)
                good_labels.append(label)
            except OSError as e:
                bad_files.append(str(path), label, str(e))
                
            if len(bad_files) > 0:  # Write bad files to an output log
                with open("../Data/bad_files.csv", 'w', newline='', encoding='utf8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['path', 'label', 'error'])
                    writer.writerows(bad_files)

        return good_paths, good_labels, bad_files

# ---------------------------------------------------------------------------
# Utility functions (module-level so they can be unit-tested independently)
# ---------------------------------------------------------------------------

def _read_bytes(path: Path, max_bytes: int) -> bytes:
    """
    Read up to max_bytes bytes from a file.

    Reading only what we need avoids loading entire 10 MB binaries into
    memory when the effective input window is 2 MB.
    """
    try:
        with open(path, "rb") as fh:
            return fh.read(max_bytes)
    except OSError as exc:
        raise OSError(f"Failed to read PE file '{path}': {exc}") from exc


def _pad_or_truncate(raw: bytes, max_bytes: int) -> Tensor:
    """
    Convert raw bytes to a fixed-length LongTensor.

    - If len(raw) >= max_bytes: take the first max_bytes bytes (truncate).
    - If len(raw) <  max_bytes: right-pad with PAD_TOKEN (256).

    Returns a 1-D LongTensor of shape (max_bytes,).
    """
    n = len(raw)

    if n >= max_bytes:
        # Truncation: slice from head
        byte_list = list(raw[:max_bytes])
    else:
        # Padding: real bytes + PAD_TOKEN tail
        byte_list = list(raw) + [PAD_TOKEN] * (max_bytes - n)

    return torch.tensor(byte_list, dtype=torch.long)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_dataset_from_dir(
    benign_dir:  str | Path,
    malware_dir: str | Path,
    max_bytes:   int,
    extensions:  tuple[str, ...] = (".exe", ".dll", ".sys", ""),
    transform:   Optional[Callable[[Tensor], Tensor]] = None,
) -> PEBinaryDataset:
    """
    Build a PEBinaryDataset by scanning two directories:
    one for benign files (label=0) and one for malicious files (label=1).

    Parameters
    ----------
    benign_dir : path-like
        Directory containing benign PE files.
    malware_dir : path-like
        Directory containing malicious PE files.
    max_bytes : int
        Maximum input sequence length (bytes).
    extensions : tuple of str
        Only files with these extensions are included.
        Empty string "" matches files with no extension.
    transform : callable, optional
        Optional transform applied to each byte tensor.

    Returns
    -------
    PEBinaryDataset
    """
    file_paths: list[Path] = []
    labels:     list[int]  = []

    for directory, label in [(benign_dir, 0), (malware_dir, 1)]:
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        for entry in sorted(directory.iterdir()):
            if entry.is_file() and entry.suffix.lower() in extensions:
                file_paths.append(entry)
                labels.append(label)

    if not file_paths:
        raise ValueError(
            f"No PE files found in '{benign_dir}' or '{malware_dir}' "
            f"matching extensions {extensions}"
        )

    return PEBinaryDataset(
        file_paths=file_paths,
        labels=labels,
        max_bytes=max_bytes,
        transform=transform,
    )

def split(dataset: PEBinaryDataset, test_split=0.3, rand_state=42):
    from sklearn.model_selection import train_test_split

    data_labels = dataset.labels

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        dataset.file_paths, data_labels, test_size=test_split,
        random_state=rand_state, stratify=data_labels)

    return train_paths, test_paths, train_labels, test_labels