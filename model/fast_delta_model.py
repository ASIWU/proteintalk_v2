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
        hidden_dim: int = 384,
        expression_latent_dim: int = 512,
        covariate_embedding_dim: int = 64,
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
        protein_concat_mode: str = "off",
        protein_concat_dim: int = 64,
        protein_concat_topk: int = 512,
        protein_concat_init_scale: float = 0.1,
        protein_concat_seed: int = 23,
        control_logit_scale: float = 0.0,
        pair_logit_scale: float = 0.0,
        target_logit_scale: float = 0.0,
        covariate_logit_scale: float = 0.0,
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
        self.protein_concat_mode = str(protein_concat_mode).lower()
        if self.protein_concat_mode not in {"off", "pcep"}:
            raise ValueError("protein_concat_mode must be off or pcep")
        self.protein_concat_dim = int(protein_concat_dim)
        self.protein_concat_topk = int(protein_concat_topk)
        self.control_logit_scale = float(control_logit_scale)
        self.pair_logit_scale = float(pair_logit_scale)
        self.target_logit_scale = float(target_logit_scale)
        self.covariate_logit_scale = float(covariate_logit_scale)

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
        if self.protein_concat_mode == "pcep":
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
            self.pcep_context_proj = nn.Linear(hidden_dim * 3, self.protein_concat_dim)
            self.pcep_norm = nn.LayerNorm(self.protein_concat_dim)
            self.pcep_out = make_mlp(
                self.protein_concat_dim,
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

        fusion_input_dim = hidden_dim * (4 + int(self.use_ddi) + int(self.graph_feature_dim > 0))
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

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        control_expression = torch.nan_to_num(batch["control_expression"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        normalized_control = self.control_norm(control_expression)
        control_hidden = self.control_encoder(normalized_control)
        graph_features = batch["graph_features"].float() if self.graph_feature_dim > 0 else None
        graph_feature_mask = batch["graph_feature_mask"].float() if self.graph_feature_dim > 0 else None
        pair_hidden = self._encode_drug_pair(batch["drug_embeddings"].float(), graph_features, batch.get("drug_indices"))
        target_hidden = self._encode_targets(batch["target_indices"].long(), batch["target_mask"].float())
        covariate_hidden = self._encode_covariates(batch["covariates"].long(), device=control_expression.device)
        if self.protein_concat_mode == "pcep":
            pcep_hidden = self._encode_protein_concat(normalized_control, pair_hidden, target_hidden, covariate_hidden)
            control_hidden = control_hidden + pcep_hidden
        pieces = [control_hidden, pair_hidden, target_hidden, covariate_hidden]
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
        hidden = self.fusion(torch.cat(pieces, dim=-1))
        delta = self.delta_head(hidden)
        if self.residual_expression:
            expression_pred = control_expression + self.delta_scale * delta
        else:
            expression_pred = delta
        response_logits = self.response_head(hidden)
        synergy_logits = self.synergy_head(hidden)
        response_logits, synergy_logits = self._add_aux_logits(
            response_logits,
            synergy_logits,
            control_hidden,
            self.control_logit_scale,
            self.control_response_head,
            self.control_synergy_head,
        )
        response_logits, synergy_logits = self._add_aux_logits(
            response_logits,
            synergy_logits,
            pair_hidden,
            self.pair_logit_scale,
            self.pair_response_head,
            self.pair_synergy_head,
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
        return expression_pred, response_logits, synergy_logits

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
        pair_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        covariate_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if self.pcep_context_proj is None or self.pcep_norm is None or self.pcep_out is None or self.pcep_scale is None:
            raise RuntimeError("PCEP was not initialized")
        protein_features = self.expression_protein_features.to(normalized_expression.device, dtype=normalized_expression.dtype)
        context = torch.cat([pair_hidden, target_hidden, covariate_hidden], dim=-1)
        query = self.pcep_context_proj(context).to(normalized_expression.dtype)
        scores = (query @ protein_features.t()) / math.sqrt(max(1, protein_features.shape[1]))
        scores = scores * normalized_expression + self.pcep_score_bias.to(scores.device, scores.dtype)
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
        pooled = self.pcep_norm(pooled)
        return self.pcep_scale.to(pooled.dtype) * self.pcep_out(pooled)

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
