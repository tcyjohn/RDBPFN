from __future__ import annotations

import json
import logging

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils import get_default_device


logger = logging.getLogger(__name__)


def _extract_fk_values(
    X: h5py.Dataset,
    indices: np.ndarray,
    fk_indices_per_sample: list[list[int]],
    max_rows: int,
) -> torch.Tensor:
    """Extract FK column values from H5 X dataset for a batch.

    Returns (B, max_rows, max_fk_cols) LongTensor padded with -1 for
    samples with no FK columns or fewer FK columns than the max.
    """
    max_fk_cols = max((len(fi) for fi in fk_indices_per_sample), default=0)
    if max_fk_cols == 0:
        return torch.zeros(len(indices), max_rows, 0, dtype=torch.long)

    B = len(indices)
    fk_vals = torch.full((B, max_rows, max_fk_cols), -1, dtype=torch.long)
    for b, (ds_idx, fk_idx_list) in enumerate(zip(indices, fk_indices_per_sample)):
        if not fk_idx_list:
            continue
        vals = X[ds_idx, :max_rows, fk_idx_list].astype(np.int64)
        fk_vals[b, :, :len(fk_idx_list)] = torch.from_numpy(vals)
    return fk_vals


class PriorDumpDataLoader(DataLoader):
    def __init__(
        self, filename, num_steps=None, batch_size=32, device=None, start_index=0
    ):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.device = device or get_default_device()
        self._file_handle = None
        with h5py.File(self.filename, "r") as f:
            self.max_num_classes = f["max_num_classes"][0]
            self.dataset_size = f["X"].shape[0]
            self.has_category_mask = "feature_is_categorical" in f
            self.has_available_features = "num_available_features" in f
            fk_raw = f.attrs.get("fk_column_indices", "[]")
            self.fk_column_indices: list[list[int]] = json.loads(fk_raw) if isinstance(fk_raw, str) else list(fk_raw)
        self.pointer = start_index % self.dataset_size
        if start_index > 0:
            logger.info("Starting dataset iteration from index %d", self.pointer)

    def _ensure_file_open(self):
        """Keep file handle open to avoid repeated open/close overhead."""
        if self._file_handle is None:
            self._file_handle = h5py.File(self.filename, "r")
        return self._file_handle

    def close(self):
        """Close the file handle if open."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __del__(self):
        self.close()

    def __iter__(self):
        step_counter = 0
        f = self._ensure_file_open()
        while True:
            if self.num_steps is not None and step_counter >= self.num_steps:
                break
            indices = (
                np.arange(self.pointer, self.pointer + self.batch_size)
                % self.dataset_size
            ).astype(np.int64)
            num_features_batch = f["num_features"][indices]
            num_datapoints_batch = f["num_datapoints"][indices]
            train_test_split_index = f["single_eval_pos"][indices]
            available_features_batch = (
                f["num_available_features"][indices]
                if self.has_available_features
                else num_features_batch
            )
            num_features_max = (
                int(num_features_batch.max()) if num_features_batch.size else 0
            )
            max_seq_in_batch = (
                int(num_datapoints_batch.max()) if num_datapoints_batch.size else 0
            )
            x = torch.from_numpy(
                f["X"][indices, :max_seq_in_batch, :num_features_max]
            )
            y = torch.from_numpy(f["y"][indices, :max_seq_in_batch])
            category_mask = None
            if self.has_category_mask:
                category_mask = torch.from_numpy(
                    f["feature_is_categorical"][indices, :num_features_max]
                ).to(torch.float32)
            available_features_batch = (
                available_features_batch
                if isinstance(available_features_batch, np.ndarray)
                else available_features_batch
            )
            num_features_tensor = torch.from_numpy(num_features_batch).to(self.device)
            available_features_tensor = torch.from_numpy(
                available_features_batch
            ).to(self.device)
            prev_pointer = self.pointer
            self.pointer = (self.pointer + self.batch_size) % self.dataset_size
            if self.pointer <= prev_pointer:
                logger.info("Finished iteration over all stored datasets!")

            # Extract FK values for FK attention bias
            fk_indices_for_batch = [
                self.fk_column_indices[i] if i < len(self.fk_column_indices) else []
                for i in indices
            ]
            fk_values = _extract_fk_values(
                f["X"], indices, fk_indices_for_batch, max_seq_in_batch,
            )

            batch = dict(
                x=x.to(self.device),
                y=y.to(self.device),
                train_test_split_index=int(train_test_split_index[0]),
                num_features=num_features_tensor,
                num_available_features=available_features_tensor,
            )
            if category_mask is not None:
                batch["category_mask"] = category_mask.to(self.device)
            if any(fk_indices_for_batch):
                batch["fk_values"] = fk_values.to(self.device)
            yield batch
            step_counter += 1

    def __len__(self):
        return self.num_steps if self.num_steps is not None else self.dataset_size


class JointPriorLoader:
    def __init__(
        self,
        loaders: list[PriorDumpDataLoader],
        steps_per_epoch: int,
        weights: list[float] | None = None,
        seed: int = 0,
        shard_id: int = 0,
        num_shards: int = 1,
    ):
        if not loaders:
            raise ValueError("JointPriorLoader requires at least one loader.")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive.")
        self.loaders = loaders
        self.global_steps = steps_per_epoch
        self.num_shards = max(1, num_shards)
        self.shard_id = int(shard_id) % self.num_shards
        self.steps_per_epoch = (self.global_steps + self.num_shards - 1) // self.num_shards
        self.seed = seed
        self.epoch = 0
        if weights is None:
            probs = np.ones(len(loaders), dtype=np.float64)
        else:
            probs = np.asarray(weights, dtype=np.float64)
            if probs.shape != (len(loaders),):
                raise ValueError("loader_weights must match number of loaders.")
            probs = np.clip(probs, 0.0, None)
            if probs.sum() == 0:
                probs[:] = 1.0
        self.probs = probs / probs.sum()
        self._rng = np.random.default_rng(self.seed)
        self._iterators: list | None = None
        self._epoch_choices: np.ndarray | None = None

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._rng = np.random.default_rng(self.seed + epoch)
        self._iterators = None
        self._epoch_choices = None

    def _ensure_iterators(self):
        if self._iterators is None:
            self._iterators = [iter(loader) for loader in self.loaders]

    def __iter__(self):
        self._ensure_iterators()
        num_loaders = len(self.loaders)
        if self._epoch_choices is None or len(self._epoch_choices) != self.global_steps:
            self._epoch_choices = self._rng.choice(
                num_loaders, size=self.global_steps, p=self.probs
            )
        yielded = 0
        positions = range(self.shard_id, self.global_steps, self.num_shards)
        for pos in positions:
            if yielded >= self.steps_per_epoch:
                break
            idx = int(self._epoch_choices[pos])
            iterator = self._iterators[idx]
            try:
                batch = next(iterator)
            except StopIteration:
                self._iterators[idx] = iter(self.loaders[idx])
                batch = next(self._iterators[idx])
            batch["source_id"] = idx
            yield batch
            yielded += 1
