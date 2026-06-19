from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.nn.modules.transformer import LayerNorm, Linear, MultiheadAttention
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

logger = logging.getLogger(__name__)


class FKAttentionBias(nn.Module):
    """Learnable FK-aware attention bias for between-datapoints attention.

    For each FK column j, adds λ_j * I[fk_i == fk_k] to attention scores,
    letting rows that share the same FK value attend more strongly to each other.

    λ_j is softplus-constrained to stay positive.
    """

    def __init__(self, init_lambda: float = 0.1):
        super().__init__()
        raw_init = float(np.log(np.exp(init_lambda) - 1))  # invert softplus
        self.raw_lambdas = nn.Parameter(torch.empty(0))

    def _lambdas_for(self, K: int) -> torch.Tensor:
        """Return softplus lambdas for K FK columns, expanding the parameter if needed."""
        current = self.raw_lambdas.shape[0]
        if K <= current:
            return F.softplus(self.raw_lambdas[:K])
        # Expand and replace the parameter to accommodate more FK columns
        raw_init = float(np.log(np.exp(0.1) - 1))
        pad = torch.full((K - current,), raw_init, device=self.raw_lambdas.device)
        self.raw_lambdas = nn.Parameter(torch.cat([self.raw_lambdas.data, pad]))
        return F.softplus(self.raw_lambdas[:K])

    def forward(
        self,
        fk_values: torch.Tensor,
        train_rows: int,
        parent_entity_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute FK attention bias matrices.

        Args:
            fk_values: (B, total_rows, num_fk_cols) long tensor, -1 for null FK.
            train_rows: number of support rows.
            parent_entity_ids: (B, total_rows, num_fk_cols) long tensor, -1 for null.
                When provided, use entity-level keys for matching instead of raw FK values.

        Returns:
            (bias_left, bias_right) — each is (B, tr, tr) or (B, te, tr), or
            None when fk_values has no FK columns.
        """
        if fk_values is None or fk_values.shape[-1] == 0:
            return None, None

        B, total_rows, K = fk_values.shape
        test_rows = total_rows - train_rows
        lambdas = self._lambdas_for(K)

        # Use entity-level keys when available, raw FK otherwise
        match_keys = parent_entity_ids if parent_entity_ids is not None else fk_values

        keys_train = match_keys[:, :train_rows, :]  # (B, tr, K)
        keys_test = match_keys[:, train_rows:, :]    # (B, te, K)

        # Mask: only match VALID (non-negative) keys
        valid_train = (keys_train >= 0)  # (B, tr, K)
        valid_test = (keys_test >= 0)    # (B, te, K)

        # bias_left: (B, tr, tr) — support↔support
        match_left = (
            keys_train.unsqueeze(2) == keys_train.unsqueeze(1)
        ).float()  # (B, tr, tr, K)
        both_valid_left = (
            valid_train.unsqueeze(2) & valid_train.unsqueeze(1)
        ).float()
        bias_left = ((match_left * both_valid_left) * lambdas).sum(dim=-1)

        # bias_right: (B, te, tr) — query→support
        match_right = (
            keys_test.unsqueeze(2) == keys_train.unsqueeze(1)
        ).float()  # (B, te, tr, K)
        both_valid_right = (
            valid_test.unsqueeze(2) & valid_train.unsqueeze(1)
        ).float()
        bias_right = ((match_right * both_valid_right) * lambdas).sum(dim=-1)

        return bias_left, bias_right


class EntityAttentionBias(nn.Module):
    """Learnable same-entity attention bias for between-datapoints attention.

    Adds λ_se * I[entity_id[i] == entity_id[j]] to attention scores, letting
    rows that belong to the same entity (temporal snapshots) attend more
    strongly to each other.

    λ_se is softplus-constrained to stay positive.
    """

    def __init__(self, init_lambda: float = 0.1):
        super().__init__()
        raw_init = float(np.log(np.exp(init_lambda) - 1))
        self.raw_lambda = nn.Parameter(torch.tensor(raw_init))

    def forward(
        self,
        entity_ids: torch.Tensor,
        train_rows: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute same-entity attention bias matrices.

        Args:
            entity_ids: (B, total_rows) long tensor, -1 for non-entity rows.
            train_rows: number of support rows.

        Returns:
            (bias_left, bias_right) — each is (B, tr, tr) or (B, te, tr), or
            None when entity_ids is all -1 (no entity table).
        """
        if entity_ids is None:
            return None, None

        B = entity_ids.shape[0]
        total_rows = entity_ids.shape[1]
        test_rows = total_rows - train_rows
        lam = F.softplus(self.raw_lambda)

        eid_train = entity_ids[:, :train_rows]  # (B, tr)
        eid_test = entity_ids[:, train_rows:]    # (B, te)

        valid_train = (eid_train >= 0)  # (B, tr)
        valid_test = (eid_test >= 0)    # (B, te)

        # bias_left: (B, tr, tr) — support↔support
        match_left = (eid_train.unsqueeze(2) == eid_train.unsqueeze(1)).float()
        both_valid_left = (
            valid_train.unsqueeze(2) & valid_train.unsqueeze(1)
        ).float()
        bias_left = match_left * both_valid_left * lam

        # bias_right: (B, te, tr) — query→support
        match_right = (eid_test.unsqueeze(2) == eid_train.unsqueeze(1)).float()
        both_valid_right = (
            valid_test.unsqueeze(2) & valid_train.unsqueeze(1)
        ).float()
        bias_right = match_right * both_valid_right * lam

        return bias_left, bias_right


@dataclass
class ModelConfig:
    type: Literal["base", "categorical"] = "base"
    embedding_size: int = 96
    num_attention_heads: int = 4
    mlp_hidden_size: int = 192
    num_layers: int = 3
    num_outputs: int = 2
    num_category_buckets: int = 10
    per_column_embeddings: bool = False
    sort_category_embeddings: bool = False
    invariant_noise_encoder: bool = False
    dual_feature_attention: bool = False
    category_as_numeric: bool = False
    use_fk_bias: bool = False
    use_entity_bias: bool = False


class FeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        x = x.unsqueeze(-1)
        train_slice = x[:, :train_test_split_index]
        valid_mask = ~torch.isnan(train_slice)
        valid_count = valid_mask.sum(dim=1, keepdims=True).clamp(min=1)
        train_filled = torch.where(
            valid_mask, train_slice, torch.zeros_like(train_slice)
        )
        mean = train_filled.sum(dim=1, keepdims=True) / valid_count
        diff = torch.where(
            valid_mask, train_slice - mean, torch.zeros_like(train_slice)
        )
        var = (diff**2).sum(dim=1, keepdims=True) / valid_count
        std = torch.sqrt(var + 1e-20)
        x = torch.where(torch.isnan(x), mean, x)
        x = (x - mean) / std
        x = torch.clip(x, min=-100, max=100)
        return self.linear_layer(x)


class TargetEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        mean = torch.mean(y_train.to(torch.float), dim=1, keepdim=True)
        padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.linear_layer(y)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        mlp_hidden_size: int,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        device=None,
        dtype=None,
        dual_feature_attention: bool = False,
        use_fk_bias: bool = False,
        use_entity_bias: bool = False,
    ):
        super().__init__()
        self.self_attention_between_datapoints = MultiheadAttention(
            embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
        )
        self.use_dual_feature_attention = dual_feature_attention
        if self.use_dual_feature_attention:
            self.numeric_feature_attention = MultiheadAttention(
                embedding_size,
                nhead,
                batch_first=batch_first,
                device=device,
                dtype=dtype,
            )
            self.categorical_feature_attention = MultiheadAttention(
                embedding_size,
                nhead,
                batch_first=batch_first,
                device=device,
                dtype=dtype,
            )
        else:
            self.self_attention_between_features = MultiheadAttention(
                embedding_size,
                nhead,
                batch_first=batch_first,
                device=device,
                dtype=dtype,
            )
        self.linear1 = Linear(
            embedding_size, mlp_hidden_size, device=device, dtype=dtype
        )
        self.linear2 = Linear(
            mlp_hidden_size, embedding_size, device=device, dtype=dtype
        )
        self.norm1 = LayerNorm(
            embedding_size, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.norm2 = LayerNorm(
            embedding_size, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.norm3 = LayerNorm(
            embedding_size, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.fk_bias = FKAttentionBias() if use_fk_bias else None
        self.entity_bias = EntityAttentionBias() if use_entity_bias else None

    def forward(
        self,
        src: torch.Tensor,
        train_test_split_index: int,
        category_mask: torch.Tensor | None = None,
        fk_values: torch.Tensor | None = None,
        entity_ids: torch.Tensor | None = None,
        parent_entity_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = src.shape
        src = src.reshape(batch_size * rows_size, col_size, embedding_size)
        if self.use_dual_feature_attention:
            if category_mask is None:
                raise ValueError(
                    "Dual feature attention requires a category mask to be provided."
                )
            mask = category_mask
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).expand(batch_size, rows_size, col_size)
            mask = mask.reshape(batch_size * rows_size, col_size)
            mask = mask.unsqueeze(-1).to(dtype=torch.bool, device=src.device)
            numeric_out = self.numeric_feature_attention(src, src, src)[0]
            categorical_out = self.categorical_feature_attention(src, src, src)[0]
            src = torch.where(mask, categorical_out, numeric_out) + src
        else:
            src = self.self_attention_between_features(src, src, src)[0] + src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        src = src.transpose(1, 2)
        src = src.reshape(batch_size * col_size, rows_size, embedding_size)

        # FK attention bias (row-level → replicated across feature columns)
        fk_bias_left, fk_bias_right = None, None
        if self.fk_bias is not None and fk_values is not None:
            fb_left, fb_right = self.fk_bias(fk_values, train_test_split_index, parent_entity_ids=parent_entity_ids)
            if fb_left is not None:
                fk_b = fb_left.shape[0]
                if fk_b < batch_size:
                    fb_left = fb_left.repeat(batch_size // fk_b, 1, 1)
                    fb_right = fb_right.repeat(batch_size // fk_b, 1, 1)
                fk_bias_left = fb_left
                fk_bias_right = fb_right

        # Same-entity bias
        ent_bias_left, ent_bias_right = None, None
        if self.entity_bias is not None and entity_ids is not None:
            eb_left, eb_right = self.entity_bias(entity_ids, train_test_split_index)
            if eb_left is not None:
                eb = eb_left.shape[0]
                if eb < batch_size:
                    eb_left = eb_left.repeat(batch_size // eb, 1, 1)
                    eb_right = eb_right.repeat(batch_size // eb, 1, 1)
                ent_bias_left = eb_left
                ent_bias_right = eb_right

        # Combine FK bias + entity bias → expand to (B*C*nhead, L, S)
        combined_left, combined_right = None, None
        if fk_bias_left is not None or ent_bias_left is not None:
            test_rows = rows_size - train_test_split_index
            combined_left = torch.zeros(batch_size, train_test_split_index, train_test_split_index, device=src.device)
            combined_right = torch.zeros(batch_size, test_rows, train_test_split_index, device=src.device)
            if fk_bias_left is not None:
                combined_left = combined_left + fk_bias_left
                combined_right = combined_right + fk_bias_right
            if ent_bias_left is not None:
                combined_left = combined_left + ent_bias_left
                combined_right = combined_right + ent_bias_right
            nhead = self.self_attention_between_datapoints.num_heads
            combined_left = combined_left.repeat_interleave(col_size * nhead, dim=0)
            combined_right = combined_right.repeat_interleave(col_size * nhead, dim=0)

        src_left = self.self_attention_between_datapoints(
            src[:, :train_test_split_index],
            src[:, :train_test_split_index],
            src[:, :train_test_split_index],
            attn_mask=combined_left,
        )[0]
        src_right = self.self_attention_between_datapoints(
            src[:, train_test_split_index:],
            src[:, :train_test_split_index],
            src[:, :train_test_split_index],
            attn_mask=combined_right,
        )[0]
        src = torch.cat([src_left, src_right], dim=1) + src
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)
        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = self.norm3(src)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(x)))


class NanoTabPFNModel(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        num_outputs: int,
        dual_feature_attention: bool = False,
        use_fk_bias: bool = False,
        use_entity_bias: bool = False,
    ):
        super().__init__()
        self.dual_feature_attention = dual_feature_attention
        self.use_fk_bias = use_fk_bias
        self.use_entity_bias = use_entity_bias
        self.use_category_mask = False
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embedding_size,
                    num_attention_heads,
                    mlp_hidden_size,
                    dual_feature_attention=dual_feature_attention,
                    use_fk_bias=use_fk_bias,
                    use_entity_bias=use_entity_bias,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)

    def forward(
        self,
        src: tuple[torch.Tensor, torch.Tensor],
        train_test_split_index: int,
        fk_values: torch.Tensor | None = None,
        entity_ids: torch.Tensor | None = None,
        parent_entity_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_src, y_src = src
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        x_src = self.feature_encoder(x_src, train_test_split_index)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)
        src = torch.cat([x_src, y_src], 2)
        category_mask = None
        for block in self.transformer_blocks:
            src = block(
                src,
                train_test_split_index=train_test_split_index,
                category_mask=category_mask,
                fk_values=fk_values,
                entity_ids=entity_ids,
                parent_entity_ids=parent_entity_ids,
            )
        output = src[:, train_test_split_index:, -1, :]
        output = self.decoder(output)
        return output


class FeatureEncoderWithCategories(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        num_category_buckets: int = 10,
        per_column_embeddings: bool = False,
        sort_embeddings: bool = False,
        invariant_noise_encoder: bool = False,
    ):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)
        self.num_category_buckets = num_category_buckets
        self.per_column_embeddings = per_column_embeddings
        self.invariant_noise_encoder = invariant_noise_encoder
        if self.invariant_noise_encoder and not self.per_column_embeddings:
            raise ValueError(
                "Invariant noise encoder requires per-column embeddings to be enabled."
            )
        self.sort_embeddings = sort_embeddings and not self.invariant_noise_encoder
        if not per_column_embeddings:
            self.category_embedding = nn.Embedding(
                num_embeddings=num_category_buckets, embedding_dim=embedding_size
            )
        else:
            self.category_embedding = None
        self.embedding_size = embedding_size
        self.column_embeddings: dict[int, nn.Embedding] = {}
        self.invariant_noise_hidden_dim = embedding_size // 2
        self.noise_size = 32
        if self.invariant_noise_encoder:
            self.noise_encoder_mlp1 = nn.Sequential(
                nn.Linear(1, self.invariant_noise_hidden_dim // 2),
                nn.GELU(),
                nn.Linear(
                    self.invariant_noise_hidden_dim // 2,
                    self.invariant_noise_hidden_dim,
                ),
            )
            self.noise_encoder_mlp2 = nn.Linear(
                self.invariant_noise_hidden_dim, self.embedding_size
            )
            # self.noise_size = 32
        else:
            self.category_encoder = nn.Sequential(
                nn.Linear(self.noise_size, self.embedding_size // 2),
                nn.GELU(),
                nn.Linear(self.embedding_size // 2, self.embedding_size),
            )

    def _apply_invariant_noise_encoder(self, noise: torch.Tensor) -> torch.Tensor:
        encoded = self.noise_encoder_mlp1(noise.unsqueeze(-1))
        encoded = encoded.sum(dim=-2)
        return self.noise_encoder_mlp2(encoded)

    def forward(
        self,
        x: torch.Tensor,
        train_test_split_index: int,
        category_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x.unsqueeze(-1)
        mean = torch.mean(x[:, :train_test_split_index], dim=1, keepdims=True)
        std = torch.std(x[:, :train_test_split_index], dim=1, keepdims=True) + 1e-20
        normalized = torch.clip((x - mean) / std, min=-100, max=100)
        numeric_emb = self.linear_layer(normalized)

        if category_mask is None:
            return numeric_emb

        mask = (
            category_mask.unsqueeze(1).unsqueeze(-1).to(dtype=x.dtype, device=x.device)
        )
        if not torch.any(mask.bool()):
            return numeric_emb

        cat_values = torch.nan_to_num(x.squeeze(-1), nan=0.0)
        cat_values = cat_values.round().long()

        if self.per_column_embeddings:
            cat_emb_list = []
            for col_idx in range(cat_values.shape[-1]):
                emb = self.column_embeddings.get(col_idx)
                if emb is None:
                    emb = nn.Embedding(
                        num_embeddings=self.num_category_buckets,
                        embedding_dim=self.noise_size,
                    )
                    self.column_embeddings[col_idx] = emb.to(x.device)
                values = cat_values[..., col_idx]
                values = torch.clamp(values, min=0, max=self.num_category_buckets - 1)
                temp = emb(values)
                if self.invariant_noise_encoder:
                    temp = self._apply_invariant_noise_encoder(temp)
                else:
                    if self.sort_embeddings:
                        temp, _ = torch.sort(temp, dim=-1)
                    temp = self.category_encoder(temp)
                cat_emb_list.append(temp)
            cat_emb = torch.stack(cat_emb_list, dim=-2)
        else:
            cat_values = torch.clamp(
                cat_values, min=0, max=self.category_embedding.num_embeddings - 1
            )
            cat_emb = self.category_embedding(cat_values)
        return torch.where(mask.bool(), cat_emb, numeric_emb)


class NanoTabPFNModelCategorical(NanoTabPFNModel):
    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        num_outputs: int,
        num_category_buckets: int = 10,
        per_column_embeddings: bool = False,
        sort_embeddings: bool = False,
        invariant_noise_encoder: bool = False,
        dual_feature_attention: bool = False,
        category_as_numeric: bool = False,
        use_fk_bias: bool = False,
        use_entity_bias: bool = False,
    ):
        super().__init__(
            embedding_size,
            num_attention_heads,
            mlp_hidden_size,
            num_layers,
            num_outputs,
            dual_feature_attention=dual_feature_attention,
            use_fk_bias=use_fk_bias,
            use_entity_bias=use_entity_bias,
        )
        self.category_as_numeric = category_as_numeric
        if self.category_as_numeric:
            self.feature_encoder = FeatureEncoder(embedding_size)
            pass
        else:
            self.feature_encoder = FeatureEncoderWithCategories(
                embedding_size,
                num_category_buckets=num_category_buckets,
                per_column_embeddings=per_column_embeddings,
                sort_embeddings=sort_embeddings,
                invariant_noise_encoder=invariant_noise_encoder,
            )
        self.use_category_mask = (
            not self.category_as_numeric
        ) or dual_feature_attention

    def forward(
        self,
        src: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        train_test_split_index: int,
        fk_values: torch.Tensor | None = None,
        entity_ids: torch.Tensor | None = None,
        parent_entity_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(src) == 2:
            if not self.category_as_numeric or self.dual_feature_attention:
                raise ValueError(
                    "Category mask is required unless using category-as-numeric without dual attention."
                )
            x_src, y_src = src
            category_mask = None
        elif len(src) == 3:
            x_src, y_src, category_mask = src
        else:
            raise ValueError(
                "NanoTabPFNModelCategorical expects (x, y, [category_mask]) tuple."
            )
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        if self.category_as_numeric:
            x_src = self.feature_encoder(x_src, train_test_split_index)
        else:
            x_src = self.feature_encoder(x_src, train_test_split_index, category_mask)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)
        src = torch.cat([x_src, y_src], 2)
        feature_category_mask = None
        if category_mask is not None:
            mask = category_mask.to(dtype=torch.bool, device=src.device)
            mask = mask.unsqueeze(1).expand(-1, num_rows, -1)
            target_mask = torch.zeros(
                mask.shape[0], num_rows, 1, dtype=torch.bool, device=mask.device
            )
            feature_category_mask = torch.cat([mask, target_mask], dim=2)
        for block in self.transformer_blocks:
            src = block(
                src,
                train_test_split_index=train_test_split_index,
                category_mask=feature_category_mask,
                fk_values=fk_values,
                entity_ids=entity_ids,
                parent_entity_ids=parent_entity_ids,
            )
        output = src[:, train_test_split_index:, -1, :]
        output = self.decoder(output)
        return output


class NanoTabPFNClassifier:
    def __init__(self, model: NanoTabPFNModel, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.fk_values: torch.Tensor | None = None
        self.entity_ids: torch.Tensor | None = None
        self.parent_entity_ids: torch.Tensor | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        fk_values: np.ndarray | None = None,
        entity_ids: np.ndarray | None = None,
        parent_entity_ids: np.ndarray | None = None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.num_classes = max(set(y_train)) + 1
        if fk_values is not None:
            self.fk_values = torch.from_numpy(fk_values).long().to(self.device)
        else:
            self.fk_values = None
        if entity_ids is not None:
            self.entity_ids = torch.from_numpy(entity_ids).long().to(self.device)
        else:
            self.entity_ids = None
        if parent_entity_ids is not None:
            self.parent_entity_ids = torch.from_numpy(parent_entity_ids).long().to(self.device)
        else:
            self.parent_entity_ids = None

    def predict_proba(
        self,
        X_test: np.ndarray,
        fk_values_test: np.ndarray | None = None,
        entity_ids_test: np.ndarray | None = None,
        parent_entity_ids_test: np.ndarray | None = None,
    ) -> np.ndarray:
        x = np.concatenate((self.X_train, X_test))
        y = self.y_train
        n_train = len(self.X_train)
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)

            if self.fk_values is not None:
                if fk_values_test is not None:
                    fk_test = torch.from_numpy(fk_values_test).long().to(self.device)
                    fk = torch.cat([self.fk_values, fk_test], dim=0).unsqueeze(0)
                else:
                    pad = torch.full(
                        (len(X_test), self.fk_values.shape[1]),
                        -1,
                        dtype=torch.long,
                        device=self.device,
                    )
                    fk = torch.cat([self.fk_values, pad], dim=0).unsqueeze(0)
            else:
                fk = None

            if self.entity_ids is not None:
                if entity_ids_test is not None:
                    eid_test = torch.from_numpy(entity_ids_test).long().to(self.device)
                    eid = torch.cat([self.entity_ids, eid_test], dim=0).unsqueeze(0)
                else:
                    pad = torch.full(
                        (len(X_test),),
                        -1,
                        dtype=torch.long,
                        device=self.device,
                    )
                    eid = torch.cat([self.entity_ids, pad], dim=0).unsqueeze(0)
            else:
                eid = None

            if self.parent_entity_ids is not None:
                if parent_entity_ids_test is not None:
                    peid_test = torch.from_numpy(parent_entity_ids_test).long().to(self.device)
                    peid = torch.cat([self.parent_entity_ids, peid_test], dim=0).unsqueeze(0)
                else:
                    pad = torch.full(
                        (len(X_test), self.parent_entity_ids.shape[1]),
                        -1,
                        dtype=torch.long,
                        device=self.device,
                    )
                    peid = torch.cat([self.parent_entity_ids, pad], dim=0).unsqueeze(0)
            else:
                peid = None

            out = self.model(
                (x, y),
                train_test_split_index=len(self.X_train),
                fk_values=fk,
                entity_ids=eid,
                parent_entity_ids=peid,
            ).squeeze(0)
            out = out[:, : self.num_classes]
            probabilities = F.softmax(out, dim=1)
            return probabilities.to("cpu").numpy()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.predict_proba(X_test).argmax(axis=1)


class NanoTabPFNClassifierCategorical(NanoTabPFNClassifier):
    def __init__(
        self,
        model: NanoTabPFNModel,
        device: torch.device,
        max_category_cardinality: int = 50,
    ):
        super().__init__(model, device)
        self.max_category_cardinality = max_category_cardinality
        self.category_mask: np.ndarray | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        fk_values: np.ndarray | None = None,
        entity_ids: np.ndarray | None = None,
        parent_entity_ids: np.ndarray | None = None,
    ):
        super().fit(X_train, y_train, fk_values, entity_ids, parent_entity_ids)
        num_features = X_train.shape[1]
        mask = np.zeros(num_features, dtype=np.uint8)
        for idx in range(num_features):
            vals = X_train[:, idx]
            if np.unique(vals).size <= self.max_category_cardinality:
                mask[idx] = 1
        self.category_mask = mask

    def predict_proba(
        self,
        X_test: np.ndarray,
        fk_values_test: np.ndarray | None = None,
        entity_ids_test: np.ndarray | None = None,
        parent_entity_ids_test: np.ndarray | None = None,
    ) -> np.ndarray:
        if not getattr(self.model, "use_category_mask", False):
            return super().predict_proba(
                X_test,
                fk_values_test=fk_values_test,
                entity_ids_test=entity_ids_test,
                parent_entity_ids_test=parent_entity_ids_test,
            )
        if self.category_mask is None:
            raise RuntimeError("Category mask not set; call fit first.")
        x = np.concatenate((self.X_train, X_test))
        y = self.y_train
        category_mask = (
            torch.from_numpy(self.category_mask).unsqueeze(0).to(torch.float32)
        )
        n_train = len(self.X_train)
        with torch.no_grad():
            x_tensor = (
                torch.from_numpy(x).unsqueeze(0).to(torch.float32).to(self.device)
            )
            y_tensor = (
                torch.from_numpy(y).unsqueeze(0).to(torch.float32).to(self.device)
            )

            if self.fk_values is not None:
                if fk_values_test is not None:
                    fk_test = torch.from_numpy(fk_values_test).long().to(self.device)
                    fk = torch.cat([self.fk_values, fk_test], dim=0).unsqueeze(0)
                else:
                    pad = torch.full(
                        (len(X_test), self.fk_values.shape[1]),
                        -1,
                        dtype=torch.long,
                        device=self.device,
                    )
                    fk = torch.cat([self.fk_values, pad], dim=0).unsqueeze(0)
            else:
                fk = None

            if self.entity_ids is not None:
                if entity_ids_test is not None:
                    eid_test = torch.from_numpy(entity_ids_test).long().to(self.device)
                    eid = torch.cat([self.entity_ids, eid_test], dim=0).unsqueeze(0)
                else:
                    pad = torch.full(
                        (len(X_test),),
                        -1,
                        dtype=torch.long,
                        device=self.device,
                    )
                    eid = torch.cat([self.entity_ids, pad], dim=0).unsqueeze(0)
            else:
                eid = None

            if self.parent_entity_ids is not None:
                if parent_entity_ids_test is not None:
                    peid_test = torch.from_numpy(parent_entity_ids_test).long().to(self.device)
                    peid = torch.cat([self.parent_entity_ids, peid_test], dim=0).unsqueeze(0)
                else:
                    pad = torch.full(
                        (len(X_test), self.parent_entity_ids.shape[1]),
                        -1,
                        dtype=torch.long,
                        device=self.device,
                    )
                    peid = torch.cat([self.parent_entity_ids, pad], dim=0).unsqueeze(0)
            else:
                peid = None

            out = self.model(
                (x_tensor, y_tensor, category_mask.to(self.device)),
                train_test_split_index=len(self.X_train),
                fk_values=fk,
                entity_ids=eid,
                parent_entity_ids=peid,
            ).squeeze(0)
            out = out[:, : self.num_classes]
            probabilities = F.softmax(out, dim=1)
            return probabilities.to("cpu").numpy()


def load_checkpoint(model: torch.nn.Module, path: Path, device: str, output_log: bool = False):
    """Load checkpoint and return full state dict for resuming training."""
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    consume_prefix_in_state_dict_if_present(state_dict, "module.")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if output_log:
        if missing:
            logger.warning("Missing keys in checkpoint (will use init values): %s", missing)
        if unexpected:
            logger.warning("Unexpected keys in checkpoint (ignored): %s", unexpected)
    has_optim = "optimizer_state_dict" in checkpoint
    if output_log:
        if has_optim:
            logger.info("Loaded checkpoint from %s (with optimizer state)", path)
        else:
            logger.info("Loaded checkpoint from %s (model only)", path)
    return checkpoint


def build_model(model_cfg: ModelConfig):
    use_fk_bias = getattr(model_cfg, "use_fk_bias", False)
    use_entity_bias = getattr(model_cfg, "use_entity_bias", False)
    if model_cfg.type == "categorical":
        return NanoTabPFNModelCategorical(
            embedding_size=model_cfg.embedding_size,
            num_attention_heads=model_cfg.num_attention_heads,
            mlp_hidden_size=model_cfg.mlp_hidden_size,
            num_layers=model_cfg.num_layers,
            num_outputs=model_cfg.num_outputs,
            num_category_buckets=model_cfg.num_category_buckets,
            per_column_embeddings=model_cfg.per_column_embeddings,
            sort_embeddings=model_cfg.sort_category_embeddings,
            invariant_noise_encoder=model_cfg.invariant_noise_encoder,
            dual_feature_attention=model_cfg.dual_feature_attention,
            category_as_numeric=model_cfg.category_as_numeric,
            use_fk_bias=use_fk_bias,
            use_entity_bias=use_entity_bias,
        )
    return NanoTabPFNModel(
        embedding_size=model_cfg.embedding_size,
        num_attention_heads=model_cfg.num_attention_heads,
        mlp_hidden_size=model_cfg.mlp_hidden_size,
        num_layers=model_cfg.num_layers,
        num_outputs=model_cfg.num_outputs,
        use_fk_bias=use_fk_bias,
        use_entity_bias=use_entity_bias,
    )


def build_classifier(model, device, model_cfg: ModelConfig):
    if model_cfg.type == "categorical":
        return NanoTabPFNClassifierCategorical(model, device)
    return NanoTabPFNClassifier(model, device)


def count_model_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return the number of model parameters."""
    if trainable_only:
        return sum(param.numel() for param in model.parameters() if param.requires_grad)
    return sum(param.numel() for param in model.parameters())


def model_size_mb(model: nn.Module, trainable_only: bool = True) -> float:
    """Return the parameter size in megabytes (MB, base-2)."""
    if trainable_only:
        params = (param for param in model.parameters() if param.requires_grad)
    else:
        params = model.parameters()
    total_bytes = sum(param.numel() * param.element_size() for param in params)
    return total_bytes / (1024**2)


def _load_model_config_from_yaml(config_path: Path) -> ModelConfig:
    with config_path.open("r") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "model" not in data:
        raise ValueError(f"Missing 'model' section in config: {config_path}")
    model_data = data["model"]
    if not isinstance(model_data, dict):
        raise ValueError(f"Invalid 'model' section in config: {config_path}")
    return ModelConfig(**model_data)


def _resolve_model_path(config_data: dict, override: str | None) -> Path | None:
    if override:
        return Path(override)
    train_cfg = config_data.get("train", {}) if isinstance(config_data, dict) else {}
    for key in ("load_model_path", "save_model_path"):
        candidate = train_cfg.get(key)
        if candidate:
            return Path(candidate)
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute NanoTabPFN model parameter counts from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config containing a 'model' section.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional checkpoint path; overrides train.load_model_path/save_model_path.",
    )
    parser.add_argument(
        "--trainable-only",
        action="store_true",
        help="Count only trainable parameters.",
    )
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    with config_path.open("r") as handle:
        config_data = yaml.safe_load(handle)
    model_cfg = ModelConfig(**config_data.get("model", {}))
    model = build_model(model_cfg)
    model_path = _resolve_model_path(config_data, args.model_path)
    if model_path and model_path.exists():
        load_checkpoint(model, model_path, device="cpu", output_log=True)
    size_mb = model_size_mb(model, trainable_only=args.trainable_only)
    print(f"parameters_mb: {size_mb:.2f}")


if __name__ == "__main__":
    _main()
