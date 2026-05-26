#!/usr/bin/env python3
"""Fast low-rank delta model for ProteinTalk.

The model keeps the full protein expression vector as input/output.  It avoids
full protein-token self-attention and full graph message passing in the hot
training path, which is the main speed bottleneck in the legacy graph
Transformer.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    *,
    dropout: float,
    layers: int = 2,
    use_layer_norm: bool = True,
) -> nn.Sequential:
    blocks: list[nn.Module] = []
    last_dim = in_dim
    for _ in range(max(1, layers - 1)):
        blocks.append(nn.Linear(last_dim, hidden_dim))
        if use_layer_norm:
            blocks.append(nn.LayerNorm(hidden_dim))
        blocks.append(nn.GELU())
        blocks.append(nn.Dropout(dropout))
        last_dim = hidden_dim
    blocks.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*blocks)


def _random_projection(in_dim: int, out_dim: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(max(1, in_dim))
    return rng.normal(0.0, scale, size=(in_dim, out_dim)).astype(np.float32)


def _zero_init_last_linear(module: nn.Module) -> None:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            return


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparse probability projection used for hop selection gates."""

    shifted = logits - logits.max(dim=dim, keepdim=True).values
    sorted_logits, _ = torch.sort(shifted, descending=True, dim=dim)
    range_values = torch.arange(1, sorted_logits.shape[dim] + 1, device=logits.device, dtype=logits.dtype)
    view_shape = [1] * logits.dim()
    view_shape[dim] = -1
    range_values = range_values.view(view_shape)
    support = 1 + range_values * sorted_logits > torch.cumsum(sorted_logits, dim=dim)
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (torch.gather(torch.cumsum(sorted_logits, dim=dim), dim, support_size - 1) - 1) / support_size.to(logits.dtype)
    return torch.clamp(shifted - tau, min=0.0)


class FastDeltaDrugResponseModel(nn.Module):
    """Low-rank control-conditioned drug response model.

    Inputs:
    - full control expression vector ``[B, G]``
    - two Morgan drug embeddings ``[B, 2, D]``
    - target protein index list, aggregated through fixed ESM embeddings
    - tokenized batch/cell covariates
    - optional compact DDI scalar

    Outputs follow the existing contract:
    ``(perturbed_expression, response_logits, synergy_logits)``.
    """

    def __init__(
        self,
        *,
        n_genes: int,
        drug_embedding_dim: int,
        protein_embedding: np.ndarray,
        ordered_protein_index: list[int] | None = None,
        covariate_sizes: list[int],
        hidden_dim: int = 512,
        expression_latent_dim: int = 768,
        covariate_embedding_dim: int = 96,
        dropout: float = 0.15,
        control_layers: int = 2,
        fusion_layers: int = 3,
        target_layers: int = 2,
        graph_feature_dim: int = 0,
        graph_layers: int = 2,
        graph_init_scale: float = 0.1,
        graph_drug_concat: bool = False,
        graph_pair_add_scale: float = 0.0,
        graph_logit_scale: float = 0.0,
        graph_feature_blocks: list[dict[str, int | str]] | None = None,
        graph_jump_fusion: str = "concat",
        graph_jump_gate: str = "softmax",
        graph_jump_temperature: float = 1.0,
        pair_fusion_mode: str = "symmetric",
        pair_type_features: bool = False,
        cell_pair_film_scale: float = 0.0,
        target_expression_mode: str = "off",
        target_expression_weight_matrix: np.ndarray | None = None,
        target_expression_dim: int = 64,
        target_expression_init_scale: float = 0.1,
        target_expression_seed: int = 29,
        target_expression_fusion_mode: str = "piece",
        target_expression_cell_gate_mode: str = "off",
        target_expression_cell_gate_scale: float = 0.0,
        target_expression_cell_gate_temperature: float = 1.0,
        protein_concat_mode: str = "off",
        protein_concat_dim: int = 64,
        protein_concat_topk: int = 512,
        protein_concat_init_scale: float = 0.1,
        protein_concat_seed: int = 23,
        protein_concat_score_mode: str = "multiply",
        protein_concat_expr_scale: float = 1.0,
        control_logit_scale: float = 0.0,
        pair_logit_scale: float = 0.0,
        pair_logit_gate: bool = False,
        target_logit_scale: float = 0.0,
        covariate_logit_scale: float = 0.0,
        response_delta_mode: str = "off",
        response_delta_dim: int = 64,
        response_delta_seed: int = 31,
        response_delta_detach: bool = False,
        delta_logit_scale: float = 0.0,
        aux_covariate_sizes: list[int] | None = None,
        prior_feature_dim: int = 0,
        prior_logit_scale: float = 0.0,
        prior_fixed_logit_scale: float = 0.0,
        use_ddi: bool = False,
        residual_expression: bool = True,
        init_delta_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.hidden_dim = int(hidden_dim)
        self.use_ddi = bool(use_ddi)
        self.residual_expression = bool(residual_expression)
        self.graph_feature_dim = int(graph_feature_dim)
        self.graph_drug_concat = bool(graph_drug_concat and self.graph_feature_dim > 0)
        self.graph_pair_add_scale = float(graph_pair_add_scale)
        self.graph_logit_scale = float(graph_logit_scale)
        self.graph_jump_fusion = str(graph_jump_fusion).lower()
        if self.graph_jump_fusion not in {"concat", "selective"}:
            raise ValueError("graph_jump_fusion must be concat or selective")
        self.graph_jump_gate = str(graph_jump_gate).lower()
        if self.graph_jump_gate not in {"softmax", "sparsemax"}:
            raise ValueError("graph_jump_gate must be softmax or sparsemax")
        self.graph_jump_temperature = float(graph_jump_temperature)
        if self.graph_jump_temperature <= 0:
            raise ValueError("graph_jump_temperature must be positive")
        self.pair_fusion_mode = str(pair_fusion_mode).lower()
        if self.pair_fusion_mode not in {"symmetric", "rich_symmetric", "ordered_concat", "dual"}:
            raise ValueError("pair_fusion_mode must be symmetric, rich_symmetric, ordered_concat, or dual")
        self.pair_type_features = bool(pair_type_features)
        self.cell_pair_film_scale = float(cell_pair_film_scale)
        if self.cell_pair_film_scale < 0.0:
            raise ValueError("cell_pair_film_scale must be non-negative")
        self.target_expression_mode = str(target_expression_mode).lower()
        if self.target_expression_mode not in {"off", "pdi", "pdi_ppi"}:
            raise ValueError("target_expression_mode must be off, pdi, or pdi_ppi")
        self.target_expression_enabled = self.target_expression_mode != "off"
        self.target_expression_dim = int(target_expression_dim)
        if self.target_expression_enabled and self.target_expression_dim <= 0:
            raise ValueError("target_expression_dim must be positive when target expression is enabled")
        self.target_expression_fusion_mode = str(target_expression_fusion_mode).lower()
        if self.target_expression_fusion_mode not in {"piece", "control_add", "pair_add"}:
            raise ValueError("target_expression_fusion_mode must be piece, control_add, or pair_add")
        self.target_expression_cell_gate_mode = str(target_expression_cell_gate_mode).lower()
        if self.target_expression_cell_gate_mode not in {"off", "magnitude", "signed"}:
            raise ValueError("target_expression_cell_gate_mode must be off, magnitude, or signed")
        self.target_expression_cell_gate_scale = float(target_expression_cell_gate_scale)
        if self.target_expression_cell_gate_scale < 0.0:
            raise ValueError("target_expression_cell_gate_scale must be non-negative")
        self.target_expression_cell_gate_temperature = float(target_expression_cell_gate_temperature)
        if self.target_expression_cell_gate_temperature <= 0.0:
            raise ValueError("target_expression_cell_gate_temperature must be positive")
        self.protein_concat_mode = str(protein_concat_mode).lower()
        if self.protein_concat_mode not in {"off", "pcep", "pcep_cell", "pcep_dual"}:
            raise ValueError("protein_concat_mode must be off, pcep, pcep_cell, or pcep_dual")
        self.protein_concat_dim = int(protein_concat_dim)
        self.protein_concat_topk = int(protein_concat_topk)
        self.protein_concat_score_mode = str(protein_concat_score_mode).lower()
        if self.protein_concat_score_mode not in {"multiply", "additive", "magnitude"}:
            raise ValueError("protein_concat_score_mode must be multiply, additive, or magnitude")
        self.protein_concat_expr_scale = float(protein_concat_expr_scale)
        self.control_logit_scale = float(control_logit_scale)
        self.pair_logit_scale = float(pair_logit_scale)
        self.pair_logit_gate_enabled = bool(pair_logit_gate)
        self.target_logit_scale = float(target_logit_scale)
        self.covariate_logit_scale = float(covariate_logit_scale)
        self.response_delta_mode = str(response_delta_mode).lower()
        if self.response_delta_mode not in {"off", "summary", "gate"}:
            raise ValueError("response_delta_mode must be off, summary, or gate")
        self.response_delta_dim = int(response_delta_dim)
        if self.response_delta_mode != "off" and self.response_delta_dim <= 0:
            raise ValueError("response_delta_dim must be positive when response_delta_mode is enabled")
        self.response_delta_detach = bool(response_delta_detach)
        self.delta_logit_scale = float(delta_logit_scale)
        if self.delta_logit_scale < 0.0:
            raise ValueError("delta_logit_scale must be non-negative")
        if self.response_delta_mode == "off" and self.delta_logit_scale:
            raise ValueError("delta_logit_scale requires response_delta_mode to be summary or gate")
        self.prior_feature_dim = int(prior_feature_dim)
        self.prior_logit_scale = float(prior_logit_scale)
        self.prior_fixed_logit_scale = float(prior_fixed_logit_scale)
        if self.prior_feature_dim < 0:
            raise ValueError("prior_feature_dim must be non-negative")
        if self.prior_fixed_logit_scale < 0.0:
            raise ValueError("prior_fixed_logit_scale must be non-negative")

        protein_embedding = np.asarray(protein_embedding, dtype=np.float32)
        self.register_buffer(
            "protein_embedding",
            torch.tensor(protein_embedding, dtype=torch.float32),
            persistent=False,
        )
        self.protein_embedding_dim = int(protein_embedding.shape[1])
        self.target_pad_index = int(protein_embedding.shape[0])
        if ordered_protein_index is None:
            ordered_protein_index = list(range(self.n_genes)) if self.n_genes <= protein_embedding.shape[0] else None
        if self.target_expression_enabled:
            if ordered_protein_index is None or len(ordered_protein_index) != self.n_genes:
                raise ValueError("target expression context requires ordered_protein_index with one entry per expression gene")
            if target_expression_weight_matrix is None:
                raise ValueError("target_expression_weight_matrix is required when target expression context is enabled")
            target_expression_weight_matrix = np.asarray(target_expression_weight_matrix)
            if target_expression_weight_matrix.ndim != 2 or target_expression_weight_matrix.shape[1] != self.n_genes:
                raise ValueError(
                    "target_expression_weight_matrix must have shape [n_drugs, n_genes]; "
                    f"got {target_expression_weight_matrix.shape}, expected second dim {self.n_genes}"
                )
            self.target_expression_drug_count = int(target_expression_weight_matrix.shape[0])
            ordered = np.asarray(ordered_protein_index, dtype=np.int64)
            if ordered.min(initial=0) < 0 or ordered.max(initial=0) >= protein_embedding.shape[0]:
                raise ValueError("ordered_protein_index contains values outside protein_embedding")
            target_projection = _random_projection(
                self.protein_embedding_dim,
                self.target_expression_dim,
                seed=target_expression_seed,
            )
            target_expression_protein_features = np.nan_to_num(
                protein_embedding[ordered] @ target_projection,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)
            target_expression_weight_matrix = np.nan_to_num(
                target_expression_weight_matrix.astype(np.float32, copy=False),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            positive_counts = np.count_nonzero(target_expression_weight_matrix > 0.0, axis=1)
            max_positive_count = int(positive_counts.max(initial=0))
            if 0 < max_positive_count < self.n_genes:
                top_idx = np.argpartition(-target_expression_weight_matrix, kth=max_positive_count - 1, axis=1)[
                    :, :max_positive_count
                ]
                top_values = np.take_along_axis(target_expression_weight_matrix, top_idx, axis=1)
                top_values = np.where(top_values > 0.0, top_values, 0.0)
                self.register_buffer(
                    "target_expression_indices",
                    torch.tensor(top_idx.astype(np.int64, copy=False), dtype=torch.long),
                    persistent=False,
                )
                self.register_buffer(
                    "target_expression_values",
                    torch.tensor(top_values.astype(np.float16, copy=False), dtype=torch.float16),
                    persistent=False,
                )
                self.register_buffer("target_expression_weights", torch.empty(0), persistent=False)
            else:
                self.register_buffer(
                    "target_expression_weights",
                    torch.tensor(target_expression_weight_matrix.astype(np.float16, copy=False), dtype=torch.float16),
                    persistent=False,
                )
                self.register_buffer("target_expression_indices", torch.empty(0, dtype=torch.long), persistent=False)
                self.register_buffer("target_expression_values", torch.empty(0), persistent=False)
            self.register_buffer(
                "target_expression_protein_features",
                torch.tensor(target_expression_protein_features, dtype=torch.float32),
                persistent=False,
            )
            self.target_expression_encoder = make_mlp(
                self.target_expression_dim * 3,
                hidden_dim,
                hidden_dim,
                dropout=dropout,
                layers=2,
            )
            _zero_init_last_linear(self.target_expression_encoder)
            self.target_expression_scale = nn.Parameter(
                torch.tensor(float(target_expression_init_scale), dtype=torch.float32)
            )
        else:
            self.target_expression_drug_count = 0
            self.register_buffer("target_expression_weights", torch.empty(0), persistent=False)
            self.register_buffer("target_expression_indices", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("target_expression_values", torch.empty(0), persistent=False)
            self.register_buffer("target_expression_protein_features", torch.empty(0), persistent=False)
            self.target_expression_encoder = None
            self.target_expression_scale = None
        if self.protein_concat_mode != "off":
            if ordered_protein_index is None or len(ordered_protein_index) != self.n_genes:
                raise ValueError("PCEP requires ordered_protein_index with one entry per expression gene")
            ordered = np.asarray(ordered_protein_index, dtype=np.int64)
            if ordered.min(initial=0) < 0 or ordered.max(initial=0) >= protein_embedding.shape[0]:
                raise ValueError("ordered_protein_index contains values outside protein_embedding")
            protein_projection = _random_projection(self.protein_embedding_dim, self.protein_concat_dim, seed=protein_concat_seed)
            expression_protein_features = np.nan_to_num(
                protein_embedding[ordered] @ protein_projection,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)
            self.register_buffer(
                "expression_protein_features",
                torch.tensor(expression_protein_features, dtype=torch.float32),
                persistent=False,
            )
            self.pcep_context_proj = (
                nn.Linear(hidden_dim * 3, self.protein_concat_dim)
                if self.protein_concat_mode in {"pcep", "pcep_dual"}
                else None
            )
            self.pcep_cell_proj = (
                nn.Linear(hidden_dim, self.protein_concat_dim)
                if self.protein_concat_mode in {"pcep_cell", "pcep_dual"}
                else None
            )
            pcep_input_dim = self.protein_concat_dim * (2 if self.protein_concat_mode == "pcep_dual" else 1)
            self.pcep_norm = nn.LayerNorm(pcep_input_dim)
            self.pcep_out = make_mlp(
                pcep_input_dim,
                hidden_dim,
                hidden_dim,
                dropout=dropout,
                layers=2,
            )
            self.pcep_scale = nn.Parameter(torch.tensor(float(protein_concat_init_scale), dtype=torch.float32))
            self.pcep_score_bias = nn.Parameter(torch.zeros(self.n_genes, dtype=torch.float32))
        else:
            self.register_buffer("expression_protein_features", torch.empty(0), persistent=False)
            self.pcep_context_proj = None
            self.pcep_cell_proj = None
            self.pcep_norm = None
            self.pcep_out = None
            self.pcep_scale = None
            self.pcep_score_bias = None

        self.control_norm = nn.LayerNorm(self.n_genes)
        self.control_encoder = make_mlp(
            self.n_genes,
            expression_latent_dim,
            hidden_dim,
            dropout=dropout,
            layers=control_layers,
        )
        drug_encoder_input_dim = drug_embedding_dim + (self.graph_feature_dim if self.graph_drug_concat else 0)
        self.drug_encoder = make_mlp(
            drug_encoder_input_dim,
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            layers=2,
        )
        pair_encoder_input_dim = self._pair_encoder_input_dim(hidden_dim)
        self.pair_type_encoder = make_mlp(2, hidden_dim, hidden_dim, dropout=dropout, layers=2) if self.pair_type_features else None
        self.pair_encoder = make_mlp(
            pair_encoder_input_dim,
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            layers=2,
        )
        self.cell_pair_film = (
            make_mlp(hidden_dim, hidden_dim, hidden_dim * 2, dropout=dropout, layers=2)
            if self.cell_pair_film_scale > 0.0
            else None
        )
        if self.cell_pair_film is not None:
            _zero_init_last_linear(self.cell_pair_film)
        self.target_encoder = make_mlp(
            self.protein_embedding_dim,
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            layers=target_layers,
        )
        self.covariate_embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=max(1, int(size)), embedding_dim=covariate_embedding_dim)
                for size in covariate_sizes
            ]
        )
        cov_input_dim = max(1, len(covariate_sizes)) * covariate_embedding_dim
        self.covariate_encoder = make_mlp(
            cov_input_dim,
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            layers=2,
        )
        self.graph_feature_blocks = self._normalize_graph_feature_blocks(graph_feature_blocks)
        if self.graph_feature_dim > 0 and self.graph_jump_fusion == "selective":
            if not self.graph_feature_blocks:
                self.graph_feature_blocks = [{"name": "graph_all", "start": 0, "end": self.graph_feature_dim}]
            self.graph_block_encoders = nn.ModuleList(
                [
                    make_mlp(
                        int(block["end"]) - int(block["start"]),
                        hidden_dim,
                        hidden_dim,
                        dropout=dropout,
                        layers=graph_layers,
                    )
                    for block in self._pair_block_dims()
                ]
            )
            self.graph_gate = make_mlp(
                hidden_dim * 3,
                hidden_dim,
                len(self.graph_feature_blocks),
                dropout=dropout,
                layers=2,
                use_layer_norm=True,
            )
            self.graph_encoder = None
        else:
            self.graph_encoder = (
                make_mlp(
                    self.graph_feature_dim * 3,
                    hidden_dim,
                    hidden_dim,
                    dropout=dropout,
                    layers=graph_layers,
                )
                if self.graph_feature_dim > 0
                else None
            )
            self.graph_block_encoders = None
            self.graph_gate = None
        self.graph_scale = (
            nn.Parameter(torch.tensor(float(graph_init_scale), dtype=torch.float32))
            if self.graph_feature_dim > 0
            else None
        )
        self.ddi_encoder = make_mlp(1, hidden_dim, hidden_dim, dropout=dropout, layers=2) if self.use_ddi else None
        self.prior_encoder = (
            make_mlp(self.prior_feature_dim, hidden_dim, hidden_dim, dropout=dropout, layers=2)
            if self.prior_feature_dim > 0
            else None
        )

        fusion_input_dim = hidden_dim * (
            4
            + int(self.target_expression_enabled and self.target_expression_fusion_mode == "piece")
            + int(self.use_ddi)
            + int(self.graph_feature_dim > 0)
            + int(self.prior_feature_dim > 0)
        )
        self.fusion = make_mlp(
            fusion_input_dim,
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            layers=fusion_layers,
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, expression_latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expression_latent_dim, self.n_genes),
        )
        self.delta_scale = nn.Parameter(torch.tensor(float(init_delta_scale), dtype=torch.float32))
        head_hidden = max(64, hidden_dim // 2)
        self.response_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.synergy_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        if self.response_delta_mode != "off":
            response_delta_projection = _random_projection(
                self.n_genes,
                self.response_delta_dim,
                seed=response_delta_seed,
            )
            self.register_buffer(
                "response_delta_projection",
                torch.tensor(response_delta_projection, dtype=torch.float32),
                persistent=False,
            )
            self.response_delta_encoder = make_mlp(
                self.response_delta_dim,
                hidden_dim,
                hidden_dim,
                dropout=dropout,
                layers=2,
            )
            self.delta_response_head, self.delta_synergy_head = self._make_aux_logit_heads(
                hidden_dim,
                head_hidden,
                dropout,
                self.delta_logit_scale,
            )
            self.delta_logit_gate_head = (
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, head_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden, 1),
                )
                if self.delta_logit_scale and self.response_delta_mode == "gate"
                else None
            )
        else:
            self.register_buffer("response_delta_projection", torch.empty(0), persistent=False)
            self.response_delta_encoder = None
            self.delta_response_head = None
            self.delta_synergy_head = None
            self.delta_logit_gate_head = None
        self.control_response_head, self.control_synergy_head = self._make_aux_logit_heads(
            hidden_dim,
            head_hidden,
            dropout,
            self.control_logit_scale,
        )
        self.pair_response_head, self.pair_synergy_head = self._make_aux_logit_heads(
            hidden_dim,
            head_hidden,
            dropout,
            self.pair_logit_scale,
        )
        self.pair_logit_gate_head = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )
            if self.pair_logit_scale and self.pair_logit_gate_enabled
            else None
        )
        self.target_response_head, self.target_synergy_head = self._make_aux_logit_heads(
            hidden_dim,
            head_hidden,
            dropout,
            self.target_logit_scale,
        )
        self.covariate_response_head, self.covariate_synergy_head = self._make_aux_logit_heads(
            hidden_dim,
            head_hidden,
            dropout,
            self.covariate_logit_scale,
        )
        self.prior_response_head, self.prior_synergy_head = self._make_aux_logit_heads(
            hidden_dim,
            head_hidden,
            dropout,
            self.prior_logit_scale if self.prior_feature_dim > 0 else 0.0,
        )
        self.aux_covariate_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, head_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden, int(size)),
                )
                for size in (aux_covariate_sizes or [])
            ]
        )
        self.graph_response_head = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )
            if self.graph_feature_dim > 0 and self.graph_logit_scale
            else None
        )
        self.graph_synergy_head = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )
            if self.graph_feature_dim > 0 and self.graph_logit_scale
            else None
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
        control_expression = torch.nan_to_num(batch["control_expression"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        normalized_control = self.control_norm(control_expression)
        control_hidden = self.control_encoder(normalized_control)
        expression_hidden = control_hidden
        graph_features = batch["graph_features"].float() if self.graph_feature_dim > 0 else None
        graph_feature_mask = batch["graph_feature_mask"].float() if self.graph_feature_dim > 0 else None
        pair_hidden = self._encode_drug_pair(batch["drug_embeddings"].float(), graph_features, batch.get("drug_indices"))
        target_hidden = self._encode_targets(batch["target_indices"].long(), batch["target_mask"].float())
        covariate_hidden = self._encode_covariates(batch["covariates"].long(), device=control_expression.device)
        if self.cell_pair_film is not None and self.cell_pair_film_scale > 0.0:
            pair_hidden = self._apply_cell_pair_film(pair_hidden, control_hidden)
        if self.protein_concat_mode != "off":
            pcep_hidden = self._encode_protein_concat(
                normalized_control,
                expression_hidden,
                pair_hidden,
                target_hidden,
                covariate_hidden,
            )
            control_hidden = control_hidden + pcep_hidden
        target_expression_hidden = None
        if self.target_expression_enabled:
            target_expression_hidden = self._encode_target_expression_context(normalized_control, batch.get("drug_indices"))
            if self.target_expression_fusion_mode == "control_add":
                control_hidden = control_hidden + target_expression_hidden
            elif self.target_expression_fusion_mode == "pair_add":
                pair_hidden = pair_hidden + target_expression_hidden
        pieces = [control_hidden, pair_hidden, target_hidden, covariate_hidden]
        if target_expression_hidden is not None and self.target_expression_fusion_mode == "piece":
            pieces.append(target_expression_hidden)
        graph_hidden = None
        if self.graph_feature_dim > 0:
            if graph_features is None or graph_feature_mask is None:
                raise RuntimeError("graph feature tensors are required when graph_encoder is initialized")
            if self.graph_jump_fusion == "selective":
                graph_hidden = self._encode_selective_graph_pair(
                    graph_features,
                    graph_feature_mask,
                    pair_hidden,
                    target_hidden,
                    covariate_hidden,
                )
            else:
                graph_hidden = self._encode_graph_pair(graph_features, graph_feature_mask)
            if self.graph_pair_add_scale:
                pair_hidden = pair_hidden + self.graph_pair_add_scale * graph_hidden
                pieces[1] = pair_hidden
            pieces.append(graph_hidden)
        if self.use_ddi:
            if self.ddi_encoder is None:
                raise RuntimeError("DDI encoder was not initialized")
            ddi = batch["ddi_value"].float().reshape(-1, 1)
            pieces.append(self.ddi_encoder(ddi))
        prior_hidden = None
        if self.prior_feature_dim > 0:
            prior_hidden = self._encode_prior_features(
                batch.get("prior_features"),
                control_expression.device,
                batch_size=control_expression.shape[0],
            )
            pieces.append(prior_hidden)
        hidden = self.fusion(torch.cat(pieces, dim=-1))
        delta = self.delta_head(hidden)
        if self.residual_expression:
            expression_pred = control_expression + self.delta_scale * delta
        else:
            expression_pred = delta
        delta_hidden = None
        if self.response_delta_mode != "off":
            delta_signal = expression_pred - control_expression if self.residual_expression else expression_pred
            delta_hidden = self._encode_response_delta(delta_signal)
        response_logits = self.response_head(hidden)
        synergy_logits = self.synergy_head(hidden)
        response_logits, synergy_logits = self._add_delta_logits(
            response_logits,
            synergy_logits,
            delta_hidden,
            hidden,
        )
        response_logits, synergy_logits = self._add_aux_logits(
            response_logits,
            synergy_logits,
            control_hidden,
            self.control_logit_scale,
            self.control_response_head,
            self.control_synergy_head,
        )
        response_logits, synergy_logits = self._add_pair_logits(
            response_logits,
            synergy_logits,
            pair_hidden,
            control_hidden,
        )
        response_logits, synergy_logits = self._add_aux_logits(
            response_logits,
            synergy_logits,
            target_hidden,
            self.target_logit_scale,
            self.target_response_head,
            self.target_synergy_head,
        )
        response_logits, synergy_logits = self._add_aux_logits(
            response_logits,
            synergy_logits,
            covariate_hidden,
            self.covariate_logit_scale,
            self.covariate_response_head,
            self.covariate_synergy_head,
        )
        if graph_hidden is not None and self.graph_logit_scale:
            if self.graph_response_head is None or self.graph_synergy_head is None:
                raise RuntimeError("graph logit heads were not initialized")
            response_logits = response_logits + self.graph_logit_scale * self.graph_response_head(graph_hidden)
            synergy_logits = synergy_logits + self.graph_logit_scale * self.graph_synergy_head(graph_hidden)
        if prior_hidden is not None and self.prior_logit_scale:
            response_logits, synergy_logits = self._add_aux_logits(
                response_logits,
                synergy_logits,
                prior_hidden,
                self.prior_logit_scale,
                self.prior_response_head,
                self.prior_synergy_head,
            )
        if self.prior_fixed_logit_scale and self.prior_feature_dim > 0 and not self.training:
            fixed_prior_logit = self._fixed_prior_logit(batch.get("prior_features"), control_expression.device)
            response_logits = response_logits + self.prior_fixed_logit_scale * fixed_prior_logit
            synergy_logits = synergy_logits + self.prior_fixed_logit_scale * fixed_prior_logit
        aux_outputs = [head(expression_hidden) for head in self.aux_covariate_heads]
        return expression_pred, response_logits, synergy_logits, aux_outputs, expression_hidden

    def _make_aux_logit_heads(
        self,
        hidden_dim: int,
        head_hidden: int,
        dropout: float,
        scale: float,
    ) -> tuple[nn.Module | None, nn.Module | None]:
        if not scale:
            return None, None

        def make_head() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )

        return make_head(), make_head()

    def _add_aux_logits(
        self,
        response_logits: torch.Tensor,
        synergy_logits: torch.Tensor,
        hidden: torch.Tensor,
        scale: float,
        response_head: nn.Module | None,
        synergy_head: nn.Module | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not scale:
            return response_logits, synergy_logits
        if response_head is None or synergy_head is None:
            raise RuntimeError("auxiliary logit heads were not initialized")
        return (
            response_logits + float(scale) * response_head(hidden),
            synergy_logits + float(scale) * synergy_head(hidden),
        )

    def _add_pair_logits(
        self,
        response_logits: torch.Tensor,
        synergy_logits: torch.Tensor,
        pair_hidden: torch.Tensor,
        control_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.pair_logit_scale:
            return response_logits, synergy_logits
        if self.pair_response_head is None or self.pair_synergy_head is None:
            raise RuntimeError("pair auxiliary logit heads were not initialized")
        gate = 1.0
        if self.pair_logit_gate_head is not None:
            gate = torch.sigmoid(self.pair_logit_gate_head(control_hidden))
        scale = float(self.pair_logit_scale)
        return (
            response_logits + scale * gate * self.pair_response_head(pair_hidden),
            synergy_logits + scale * gate * self.pair_synergy_head(pair_hidden),
        )

    def _encode_response_delta(self, delta_signal: torch.Tensor) -> torch.Tensor:
        if self.response_delta_encoder is None or self.response_delta_projection.numel() == 0:
            raise RuntimeError("response delta encoder was not initialized")
        delta_signal = torch.nan_to_num(delta_signal.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if self.response_delta_detach:
            delta_signal = delta_signal.detach()
        projection = self.response_delta_projection.to(device=delta_signal.device, dtype=delta_signal.dtype)
        projected = delta_signal @ projection
        return self.response_delta_encoder(projected)

    def _add_delta_logits(
        self,
        response_logits: torch.Tensor,
        synergy_logits: torch.Tensor,
        delta_hidden: torch.Tensor | None,
        fusion_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.delta_logit_scale:
            return response_logits, synergy_logits
        if delta_hidden is None or self.delta_response_head is None or self.delta_synergy_head is None:
            raise RuntimeError("delta logit heads were not initialized")
        gate = 1.0
        if self.delta_logit_gate_head is not None:
            gate = torch.sigmoid(self.delta_logit_gate_head(fusion_hidden))
        scale = float(self.delta_logit_scale)
        return (
            response_logits + scale * gate * self.delta_response_head(delta_hidden),
            synergy_logits + scale * gate * self.delta_synergy_head(delta_hidden),
        )

    def _pair_encoder_input_dim(self, hidden_dim: int) -> int:
        if self.pair_fusion_mode == "symmetric":
            base = hidden_dim * 3
        elif self.pair_fusion_mode == "rich_symmetric":
            base = hidden_dim * 5
        elif self.pair_fusion_mode == "ordered_concat":
            base = hidden_dim * 4
        elif self.pair_fusion_mode == "dual":
            base = hidden_dim * 5
        else:
            raise RuntimeError(f"unsupported pair_fusion_mode: {self.pair_fusion_mode!r}")
        return base + (hidden_dim if self.pair_type_features else 0)

    def _encode_drug_pair(
        self,
        drug_embeddings: torch.Tensor,
        graph_features: torch.Tensor | None = None,
        drug_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        drug_embeddings = torch.nan_to_num(drug_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        if self.graph_drug_concat:
            if graph_features is None:
                raise RuntimeError("graph features are required for graph_drug_concat")
            graph_features = torch.nan_to_num(graph_features, nan=0.0, posinf=0.0, neginf=0.0)
            drug_embeddings = torch.cat([drug_embeddings, graph_features.to(drug_embeddings.dtype)], dim=-1)
        encoded = self.drug_encoder(drug_embeddings)
        drug1 = encoded[:, 0, :]
        drug2 = encoded[:, 1, :]
        pair_features = self._drug_pair_features(drug1, drug2)
        if self.pair_type_features:
            if self.pair_type_encoder is None:
                raise RuntimeError("pair_type_encoder was not initialized")
            pair_features = torch.cat([pair_features, self._encode_pair_type(drug_indices, drug1)], dim=-1)
        return self.pair_encoder(pair_features)

    def _drug_pair_features(self, drug1: torch.Tensor, drug2: torch.Tensor) -> torch.Tensor:
        mean = 0.5 * (drug1 + drug2)
        abs_diff = torch.abs(drug1 - drug2)
        product = drug1 * drug2
        if self.pair_fusion_mode == "symmetric":
            pieces = [mean, abs_diff, product]
        elif self.pair_fusion_mode == "rich_symmetric":
            pieces = [mean, abs_diff, product, torch.maximum(drug1, drug2), torch.minimum(drug1, drug2)]
        elif self.pair_fusion_mode == "ordered_concat":
            pieces = [drug1, drug2, abs_diff, product]
        elif self.pair_fusion_mode == "dual":
            pieces = [mean, abs_diff, product, drug1, drug2]
        else:
            raise RuntimeError(f"unsupported pair_fusion_mode: {self.pair_fusion_mode!r}")
        return torch.cat(pieces, dim=-1)

    def _encode_pair_type(self, drug_indices: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
        if drug_indices is None:
            type_features = reference.new_zeros((reference.shape[0], 2))
        else:
            indices = drug_indices.to(reference.device)
            same_slot = (indices[:, 0] == indices[:, 1]).to(reference.dtype).reshape(-1, 1)
            type_features = torch.cat([same_slot, 1.0 - same_slot], dim=-1)
        return self.pair_type_encoder(type_features)

    def _apply_cell_pair_film(self, pair_hidden: torch.Tensor, control_hidden: torch.Tensor) -> torch.Tensor:
        if self.cell_pair_film is None:
            return pair_hidden
        gamma_beta = self.cell_pair_film(control_hidden)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        scale = float(self.cell_pair_film_scale)
        return pair_hidden * (1.0 + scale * torch.tanh(gamma)) + scale * beta

    def _encode_target_expression_context(
        self,
        normalized_expression: torch.Tensor,
        drug_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.target_expression_encoder is None or self.target_expression_scale is None:
            raise RuntimeError("target expression context was not initialized")
        if drug_indices is None:
            batch_size = normalized_expression.shape[0]
            return normalized_expression.new_zeros(batch_size, self.hidden_dim)
        indices = drug_indices.to(normalized_expression.device).long()
        protein_features = self.target_expression_protein_features.to(
            device=normalized_expression.device,
            dtype=normalized_expression.dtype,
        )
        indices = indices.clamp(min=0, max=self.target_expression_drug_count - 1)
        if self.target_expression_indices.numel() > 0:
            gene_indices = self.target_expression_indices[indices]
            weights = self.target_expression_values[indices].to(
                device=normalized_expression.device,
                dtype=normalized_expression.dtype,
            )
            expanded_expression = normalized_expression.unsqueeze(1).expand(-1, 2, -1)
            selected_expression = torch.gather(expanded_expression, dim=2, index=gene_indices)
            weights = self._target_expression_cell_gated_weights(weights, selected_expression)
            selected_features = protein_features[gene_indices]
            pooled = (selected_features * (weights * selected_expression).unsqueeze(-1)).sum(dim=2)
        else:
            weights = self.target_expression_weights[indices].to(
                device=normalized_expression.device,
                dtype=normalized_expression.dtype,
            )
            weights = self._target_expression_cell_gated_weights(weights, normalized_expression.unsqueeze(1))
            weighted_expression = weights * normalized_expression.unsqueeze(1)
            pooled = torch.matmul(weighted_expression, protein_features)
        context1 = pooled[:, 0, :]
        context2 = pooled[:, 1, :]
        pair_context = torch.cat(
            [
                0.5 * (context1 + context2),
                torch.abs(context1 - context2),
                context1 * context2,
            ],
            dim=-1,
        )
        encoded = self.target_expression_encoder(pair_context)
        return self.target_expression_scale.to(encoded.dtype) * encoded

    def _target_expression_cell_gated_weights(
        self,
        weights: torch.Tensor,
        selected_expression: torch.Tensor,
    ) -> torch.Tensor:
        if self.target_expression_cell_gate_mode == "off" or self.target_expression_cell_gate_scale <= 0.0:
            return weights
        valid = weights > 0.0
        if self.target_expression_cell_gate_mode == "magnitude":
            cell_signal = selected_expression.abs()
        elif self.target_expression_cell_gate_mode == "signed":
            cell_signal = selected_expression
        else:
            raise RuntimeError(f"unsupported target expression cell gate: {self.target_expression_cell_gate_mode!r}")
        logits = torch.log(weights.clamp_min(1e-8)) + float(self.target_expression_cell_gate_scale) * cell_signal
        logits = logits / float(self.target_expression_cell_gate_temperature)
        logits = logits.masked_fill(~valid, -torch.inf)
        gated = torch.softmax(logits, dim=-1)
        return torch.nan_to_num(gated, nan=0.0, posinf=0.0, neginf=0.0)

    def _encode_prior_features(
        self,
        prior_features: torch.Tensor | None,
        device: torch.device,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        if self.prior_encoder is None:
            raise RuntimeError("prior_encoder is not initialized")
        if prior_features is None:
            prior_features = torch.zeros(batch_size, self.prior_feature_dim, device=device)
        prior_features = prior_features.to(device=device, dtype=torch.float32)
        if prior_features.shape[-1] != self.prior_feature_dim:
            raise ValueError(
                f"prior_features last dimension {prior_features.shape[-1]} != expected {self.prior_feature_dim}"
            )
        prior_features = torch.nan_to_num(prior_features, nan=0.0, posinf=0.0, neginf=0.0)
        return self.prior_encoder(prior_features)

    def _fixed_prior_logit(self, prior_features: torch.Tensor | None, device: torch.device) -> torch.Tensor:
        if prior_features is None:
            return torch.zeros(0, 1, device=device)
        prior_features = prior_features.to(device=device, dtype=torch.float32)
        if prior_features.shape[-1] < 1:
            raise ValueError("fixed prior logit requires prior_features with at least one column")
        prior_prob = torch.nan_to_num(prior_features[:, 0], nan=0.5, posinf=0.5, neginf=0.5).clamp(1e-4, 1.0 - 1e-4)
        return torch.logit(prior_prob).reshape(-1, 1)

    def _encode_graph_pair(self, graph_features: torch.Tensor, graph_feature_mask: torch.Tensor) -> torch.Tensor:
        if self.graph_encoder is None:
            raise RuntimeError("graph_encoder is not initialized")
        graph_features = torch.nan_to_num(graph_features, nan=0.0, posinf=0.0, neginf=0.0)
        graph1 = graph_features[:, 0, :]
        graph2 = graph_features[:, 1, :]
        pair_features = torch.cat(
            [
                0.5 * (graph1 + graph2),
                torch.abs(graph1 - graph2),
                graph1 * graph2,
            ],
            dim=-1,
        )
        encoded = self.graph_encoder(pair_features)
        mask = graph_feature_mask.reshape(-1, 1).to(encoded.dtype)
        return self.graph_scale.to(encoded.dtype) * encoded * mask

    def _encode_selective_graph_pair(
        self,
        graph_features: torch.Tensor,
        graph_feature_mask: torch.Tensor,
        pair_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        covariate_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if self.graph_block_encoders is None or self.graph_gate is None:
            raise RuntimeError("selective graph fusion was not initialized")
        graph_features = torch.nan_to_num(graph_features, nan=0.0, posinf=0.0, neginf=0.0)
        block_hiddens = []
        for block, encoder in zip(self.graph_feature_blocks, self.graph_block_encoders, strict=True):
            start = int(block["start"])
            end = int(block["end"])
            block_features = graph_features[:, :, start:end]
            graph1 = block_features[:, 0, :]
            graph2 = block_features[:, 1, :]
            pair_features = torch.cat(
                [
                    0.5 * (graph1 + graph2),
                    torch.abs(graph1 - graph2),
                    graph1 * graph2,
                ],
                dim=-1,
            )
            block_hiddens.append(encoder(pair_features))
        stacked = torch.stack(block_hiddens, dim=1)
        gate_context = torch.cat([pair_hidden, target_hidden, covariate_hidden], dim=-1)
        gate_logits = self.graph_gate(gate_context) / self.graph_jump_temperature
        if self.graph_jump_gate == "sparsemax":
            gate = sparsemax(gate_logits, dim=-1)
        else:
            gate = torch.softmax(gate_logits, dim=-1)
        encoded = (stacked * gate.unsqueeze(-1).to(stacked.dtype)).sum(dim=1)
        mask = graph_feature_mask.reshape(-1, 1).to(encoded.dtype)
        return self.graph_scale.to(encoded.dtype) * encoded * mask

    def _encode_protein_concat(
        self,
        normalized_expression: torch.Tensor,
        expression_hidden: torch.Tensor,
        pair_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        covariate_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if self.pcep_norm is None or self.pcep_out is None or self.pcep_scale is None:
            raise RuntimeError("PCEP was not initialized")
        protein_features = self.expression_protein_features.to(normalized_expression.device, dtype=normalized_expression.dtype)
        pooled_parts = []
        if self.protein_concat_mode in {"pcep", "pcep_dual"}:
            if self.pcep_context_proj is None:
                raise RuntimeError("PCEP context projection was not initialized")
            context = torch.cat([pair_hidden, target_hidden, covariate_hidden], dim=-1)
            query = self.pcep_context_proj(context).to(normalized_expression.dtype)
            pooled_parts.append(self._pool_expression_proteins(query, normalized_expression, protein_features))
        if self.protein_concat_mode in {"pcep_cell", "pcep_dual"}:
            if self.pcep_cell_proj is None:
                raise RuntimeError("PCEP cell projection was not initialized")
            query = self.pcep_cell_proj(expression_hidden).to(normalized_expression.dtype)
            pooled_parts.append(self._pool_expression_proteins(query, normalized_expression, protein_features))
        pooled = torch.cat(pooled_parts, dim=-1) if len(pooled_parts) > 1 else pooled_parts[0]
        pooled = self.pcep_norm(pooled)
        return self.pcep_scale.to(pooled.dtype) * self.pcep_out(pooled)

    def _pool_expression_proteins(
        self,
        query: torch.Tensor,
        normalized_expression: torch.Tensor,
        protein_features: torch.Tensor,
    ) -> torch.Tensor:
        scores = (query @ protein_features.t()) / math.sqrt(max(1, protein_features.shape[1]))
        if self.protein_concat_score_mode == "multiply":
            scores = scores * normalized_expression
        elif self.protein_concat_score_mode == "additive":
            scores = scores + self.protein_concat_expr_scale * normalized_expression
        elif self.protein_concat_score_mode == "magnitude":
            scores = scores + self.protein_concat_expr_scale * normalized_expression.abs()
        else:
            raise RuntimeError(f"unsupported protein_concat_score_mode: {self.protein_concat_score_mode!r}")
        scores = scores + self.pcep_score_bias.to(scores.device, scores.dtype)
        if self.protein_concat_topk > 0 and self.protein_concat_topk < scores.shape[1]:
            top_scores, top_idx = torch.topk(scores, k=self.protein_concat_topk, dim=1)
            weights = torch.softmax(top_scores, dim=1)
            top_expression = torch.gather(normalized_expression, dim=1, index=top_idx)
            top_features = protein_features[top_idx]
            pooled = (weights * top_expression).unsqueeze(1) @ top_features
            pooled = pooled.squeeze(1)
        else:
            weights = torch.softmax(scores, dim=1)
            pooled = (weights * normalized_expression) @ protein_features
        return pooled

    def _encode_targets(self, target_indices: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        protein_embedding = self.protein_embedding.to(target_indices.device)
        zero_row = protein_embedding.new_zeros(1, protein_embedding.shape[1])
        lookup = torch.cat([protein_embedding, zero_row], dim=0)
        target_indices = target_indices.clamp(min=0, max=lookup.shape[0] - 1)
        target_embedding = lookup[target_indices]
        mask = target_mask.unsqueeze(-1).to(target_embedding.dtype)
        pooled = (target_embedding * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.target_encoder(pooled)

    def _encode_covariates(self, covariates: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        if not self.covariate_embeddings:
            return torch.zeros(covariates.shape[0], self.hidden_dim, device=device)
        parts = []
        for col_idx, embedding in enumerate(self.covariate_embeddings):
            value = covariates[:, col_idx].clamp(min=0, max=embedding.num_embeddings - 1)
            parts.append(embedding(value.to(device)))
        return self.covariate_encoder(torch.cat(parts, dim=-1))

    def _normalize_graph_feature_blocks(self, blocks: list[dict[str, int | str]] | None) -> list[dict[str, int | str]]:
        if not blocks or self.graph_feature_dim <= 0:
            return []
        normalized = []
        for block in blocks:
            start = int(block["start"])
            end = int(block["end"])
            if start < 0 or end <= start or end > self.graph_feature_dim:
                raise ValueError(f"invalid graph feature block: {block}")
            normalized.append({"name": str(block.get("name", f"block_{len(normalized)}")), "start": start, "end": end})
        normalized.sort(key=lambda item: int(item["start"]))
        return normalized

    def _pair_block_dims(self) -> list[dict[str, int | str]]:
        return [
            {"name": str(block["name"]), "start": 0, "end": 3 * (int(block["end"]) - int(block["start"]))}
            for block in self.graph_feature_blocks
        ]
