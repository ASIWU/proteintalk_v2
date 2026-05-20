#!/usr/bin/env python3
"""Legacy-style ProteinTalk models for the training-ready data pipeline.

The public model names match ``docs/Data_Process_4.md``.  The current data
pipeline always supplies a double-drug perturbation tensor, while the internals
below preserve the legacy token/CLS Transformer structure and the legacy PyG
PDI graph implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, MessagePassing
except Exception:  # pragma: no cover - dependency is validated in flow_v2.
    HeteroData = None
    HeteroConv = None
    MessagePassing = None


PDI_HETERO_MODEL_NAME = "attention_v10_hetero_cls_ee"

GRAPH_MODEL_NAMES = {
    PDI_HETERO_MODEL_NAME,
}

SELECTED_MODEL_NAMES = {
    PDI_HETERO_MODEL_NAME,
    "baseline_emb_v3",
}


@dataclass(frozen=True)
class ModelArtifacts:
    protein_embedding: np.ndarray
    drug_embedding: np.ndarray
    ordered_protein_index: list[int]
    pdi_matrix: np.ndarray | None = None


class ValueEmbedding(nn.Module):
    """Legacy continuous-value embedding with a dedicated NaN embedding."""

    def __init__(
        self,
        hidden_dim: int,
        num_bins: int = 50,
        mlp_hidden: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        self.value_mlp = nn.Sequential(
            nn.Linear(1, mlp_hidden),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(mlp_hidden, num_bins),
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.T = nn.Parameter(torch.randn(num_bins, hidden_dim) * 0.02)
        self.nan_embedding = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.dim() == 1:
            x = values.unsqueeze(-1)
        elif values.size(-1) != 1:
            x = values.unsqueeze(-1)
        else:
            x = values
        nan_mask = torch.isnan(x)
        x_safe = torch.where(nan_mask, torch.zeros_like(x), x)
        scores = self.value_mlp(x_safe) * self.alpha
        attn = F.softmax(scores, dim=-1)
        emb = attn @ self.T
        emb = torch.where(nan_mask, self.nan_embedding, emb)
        return self.dropout(emb)


class MultiCategoryEmbeddingStack(nn.Module):
    """Legacy stacked categorical covariate embedding."""

    def __init__(self, category_voc_sizes: list[int], embedding_dim: int) -> None:
        super().__init__()
        self.K = len(category_voc_sizes)
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_embeddings=max(1, int(size)), embedding_dim=embedding_dim) for size in category_voc_sizes]
        )
        self.output_dim = self.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.K:
            raise ValueError(f"Input shape must be (batch_size, {self.K}), but got {tuple(x.shape)}")
        embedded_features = []
        for idx, embedder in enumerate(self.embeddings):
            feature_column = x[:, idx].long().clamp(min=0, max=embedder.num_embeddings - 1)
            embedded_features.append(embedder(feature_column))
        return torch.stack(embedded_features, dim=1)


if MessagePassing is not None:

    class WeightedSAGEConvHetero(MessagePassing):
        """Port of legacy ``WeightedSAGEConv_hetero``."""

        def __init__(self, in_channels, out_channels: int, aggr: str = "mean", bias: bool = True) -> None:
            super().__init__(aggr=aggr, node_dim=0)
            if isinstance(in_channels, int):
                in_channels = (in_channels, in_channels)
            if not isinstance(in_channels, (tuple, list)) or len(in_channels) != 2:
                raise ValueError("in_channels must be an int or a (src_dim, dst_dim) pair")
            self.in_channels = (int(in_channels[0]), int(in_channels[1]))
            self.out_channels = int(out_channels)
            self.lin_neigh = nn.Linear(self.in_channels[0], self.out_channels, bias=bias)
            self.lin_update = nn.Linear(self.in_channels[1] + self.out_channels, self.out_channels, bias=bias)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            self.lin_neigh.reset_parameters()
            self.lin_update.reset_parameters()

        def forward(self, x, edge_index, edge_weight=None, size=None):
            if isinstance(x, tuple):
                x_src, x_dst = x
            else:
                x_src = x_dst = x
            if x_src is None or x_dst is None:
                raise ValueError("x must provide both source and destination node features")
            if size is None:
                size = (x_src.size(0), x_dst.size(0))
            if edge_weight is None:
                edge_weight = x_src.new_ones(edge_index.size(1))
            else:
                edge_weight = edge_weight.to(dtype=x_src.dtype, device=x_src.device)
            x_src_transformed = self.lin_neigh(x_src)
            out = self.propagate(edge_index=edge_index, x=x_src_transformed, edge_weight=edge_weight, size=size)
            if out.size(0) != x_dst.size(0):
                raise RuntimeError(
                    f"aggregated node count {out.size(0)} does not match destination count {x_dst.size(0)}"
                )
            return self.lin_update(torch.cat([x_dst, out], dim=-1))

        def message(self, x_j, edge_weight):
            return edge_weight.unsqueeze(-1) * x_j

else:
    WeightedSAGEConvHetero = None


class PDIOnlyProteinDrugNet(nn.Module):
    """Port of legacy ``PDIOnlyProteinDrugNet``."""

    def __init__(
        self,
        metadata,
        *,
        protein_in_dim: int,
        drug_in_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if HeteroConv is None or WeightedSAGEConvHetero is None:
            raise ImportError("torch_geometric is required for PDI graph models")
        self.dropout = dropout
        self.conv1 = HeteroConv(
            {
                ("protein", "binds", "drug"): WeightedSAGEConvHetero((protein_in_dim, drug_in_dim), hidden_dim),
                ("drug", "rev_binds", "protein"): WeightedSAGEConvHetero((drug_in_dim, protein_in_dim), hidden_dim),
            },
            aggr="sum",
        )
        self.conv2 = HeteroConv(
            {
                ("protein", "binds", "drug"): WeightedSAGEConvHetero((hidden_dim, hidden_dim), out_dim),
                ("drug", "rev_binds", "protein"): WeightedSAGEConvHetero((hidden_dim, hidden_dim), out_dim),
            },
            aggr="sum",
        )
        self.drug_mlp = nn.Sequential(nn.Linear(out_dim, out_dim), nn.ReLU())

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_weight_dict = {
            edge_type: data[edge_type].edge_weight
            for edge_type in edge_index_dict
            if hasattr(data[edge_type], "edge_weight")
        }
        h_dict = self.conv1(x_dict, edge_index_dict, edge_weight_dict)
        h_dict = {key: F.dropout(F.relu(value), p=self.dropout, training=self.training) for key, value in h_dict.items()}
        final_dict = self.conv2(h_dict, edge_index_dict, edge_weight_dict)
        return {
            "protein": F.relu(final_dict["protein"]),
            "drug": F.relu(self.drug_mlp(final_dict["drug"])),
        }


def create_pdi_only_graph(
    protein_embedding: np.ndarray,
    drug_embedding: np.ndarray,
    pdi_matrix: np.ndarray,
    *,
    reverse_pdi: bool = True,
):
    """Create the legacy protein->drug PDI graph.

    Current training-ready PDI artifacts are stored as ``[drug, protein]``.
    ``reverse_pdi=True`` is therefore the default and transposes that matrix
    before constructing the legacy ``protein -> drug`` and ``drug -> protein``
    edge types.
    """

    if HeteroData is None:
        raise ImportError("torch_geometric is required for PDI graph models")
    protein_embedding = np.asarray(protein_embedding, dtype=np.float32)
    drug_embedding = np.asarray(drug_embedding, dtype=np.float32)
    pdi_matrix = np.asarray(pdi_matrix, dtype=np.float32)
    expected_current = (drug_embedding.shape[0], protein_embedding.shape[0])
    expected_legacy = (protein_embedding.shape[0], drug_embedding.shape[0])
    if reverse_pdi:
        if pdi_matrix.shape == expected_current:
            pdi_for_graph = pdi_matrix.T
        elif pdi_matrix.shape == expected_legacy:
            pdi_for_graph = pdi_matrix
        else:
            raise ValueError(
                "pdi_matrix shape is incompatible with drug/protein embeddings; "
                f"got {pdi_matrix.shape}, expected {expected_current} or {expected_legacy}"
            )
    else:
        if pdi_matrix.shape != expected_legacy:
            raise ValueError(f"legacy PDI orientation requires {expected_legacy}, got {pdi_matrix.shape}")
        pdi_for_graph = pdi_matrix

    data = HeteroData()
    data["protein"].x = torch.tensor(protein_embedding, dtype=torch.float32)
    data["drug"].x = torch.tensor(drug_embedding, dtype=torch.float32)
    pdi = torch.tensor(pdi_for_graph, dtype=torch.float32)
    edge_index = pdi.nonzero(as_tuple=False).t()
    edge_weight = pdi[edge_index[0], edge_index[1]]
    data["protein", "binds", "drug"].edge_index = edge_index
    data["protein", "binds", "drug"].edge_weight = edge_weight
    data["drug", "rev_binds", "protein"].edge_index = edge_index.flip(0)
    data["drug", "rev_binds", "protein"].edge_weight = edge_weight
    return data


class LegacyDoubleDrugTransformer(nn.Module):
    """Shared implementation of the selected legacy Transformer variants."""

    def __init__(
        self,
        *,
        model_name: str,
        topk_genes: int,
        batch_cov_list: list[str],
        batch_cov_category_sizes: list[int],
        hidden_dim: int,
        perturb_fusion_mode: str,
        protein_embedding: np.ndarray,
        drug_embedding: np.ndarray,
        ordered_protein_index: list[int],
        pdi_matrix: np.ndarray | None,
        dropout: float,
        fusion_mode: str = "concat",
        num_heads: int = 8,
        num_layers: int = 4,
        cls_type: str = "all_1",
        graph_dropout: bool = False,
        use_target: bool = True,
        target_protein_fusion_model: str = "concat",
        gate_weight: float = 1.0,
        reverse_pdi: bool = True,
    ) -> None:
        super().__init__()
        if model_name not in SELECTED_MODEL_NAMES - {"baseline_emb_v3"}:
            raise ValueError(f"unsupported transformer model_name={model_name!r}")
        if perturb_fusion_mode not in {"add", "concat", "mlp"}:
            raise ValueError("perturb_fusion_mode must be one of: add, concat, mlp")
        if fusion_mode not in {"concat", "add"}:
            raise ValueError("fusion_mode must be one of: concat, add")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        protein_embedding = np.asarray(protein_embedding, dtype=np.float32)
        drug_embedding = np.asarray(drug_embedding, dtype=np.float32)
        ordered_protein_index = [int(idx) for idx in ordered_protein_index]
        if len(ordered_protein_index) != int(topk_genes):
            raise ValueError(
                f"ordered_protein_index length {len(ordered_protein_index)} does not match topk_genes {topk_genes}"
            )

        self.model_name = model_name
        self.topk_genes = int(topk_genes)
        self.batch_cov_list = list(batch_cov_list)
        self.hidden_dim = int(hidden_dim)
        self.perturb_fusion_mode = perturb_fusion_mode
        self.fusion_mode = fusion_mode
        self.is_graph = model_name in GRAPH_MODEL_NAMES
        self.include_target = bool(use_target)
        self.static_gene_gate = False
        self.graph_gene_gate = self.is_graph and target_protein_fusion_model == "gate"
        self.graph_dropout = bool(graph_dropout)
        self.gate_weight = float(gate_weight)

        self.register_buffer(
            "ordered_protein_index",
            torch.tensor(ordered_protein_index, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "protein_embedding_features",
            torch.tensor(protein_embedding, dtype=torch.float32),
            persistent=False,
        )
        if self.static_gene_gate:
            self.register_buffer(
                "input_protein_emb",
                torch.tensor(protein_embedding[ordered_protein_index], dtype=torch.float32),
                persistent=False,
            )

        self.gene_proj = ValueEmbedding(hidden_dim=hidden_dim, num_bins=50, mlp_hidden=128, dropout=dropout)
        pert_input_dim = hidden_dim if self.is_graph else int(drug_embedding.shape[1])
        self.pert_proj = nn.Sequential(
            nn.Linear(pert_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        target_input_dim = hidden_dim if self.is_graph else int(protein_embedding.shape[1])
        self.target_proj = nn.Sequential(
            nn.Linear(target_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.batch_proj = MultiCategoryEmbeddingStack(batch_cov_category_sizes, hidden_dim)

        token_type_count = 4 if self.include_target else 3
        self.token_type_emb = nn.Embedding(token_type_count, hidden_dim)
        self.cls_token = self._build_cls_token(cls_type, hidden_dim)

        if self.graph_dropout:
            self.emb_dropout = nn.Dropout(0.2)
        if self.static_gene_gate:
            self.fusion_norm = nn.LayerNorm(hidden_dim * 2)
            self.fusion_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            nn.init.constant_(self.fusion_gate[0].bias, -2.0)
            self.fusion_proj = nn.Linear(int(protein_embedding.shape[1]), hidden_dim)
        elif self.is_graph and self.graph_gene_gate:
            self.fusion_norm = nn.LayerNorm(hidden_dim * 2)
            self.fusion_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            nn.init.constant_(self.fusion_gate[0].bias, -2.0)
            self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)

        if self.is_graph:
            if pdi_matrix is None:
                raise ValueError(f"{model_name} requires a PDI matrix")
            self.hetero_graph = create_pdi_only_graph(
                protein_embedding,
                drug_embedding,
                pdi_matrix,
                reverse_pdi=reverse_pdi,
            )
            self.embedding_proj = PDIOnlyProteinDrugNet(
                self.hetero_graph.metadata(),
                protein_in_dim=int(protein_embedding.shape[1]),
                drug_in_dim=int(drug_embedding.shape[1]),
                out_dim=hidden_dim,
            )
        else:
            self.hetero_graph = None
            self.embedding_proj = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        head_dim = max(1, hidden_dim // 2)
        self.expression_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )
        self.synergy_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    @staticmethod
    def _build_cls_token(cls_type: str, hidden_dim: int) -> nn.Parameter:
        if "0" in cls_type and "1" not in cls_type:
            token = torch.zeros(1, 1, hidden_dim)
        else:
            token = torch.ones(1, 1, hidden_dim)
        param = nn.Parameter(token)
        if "random" in cls_type:
            nn.init.trunc_normal_(param, std=0.02)
        return param

    def _dict_to_tensor(self, cov_dict: dict[str, torch.Tensor], feature_list: list[str]) -> torch.Tensor:
        vals = []
        for key in feature_list:
            if key not in cov_dict:
                raise KeyError(f"Feature {key!r} not found in covariates_dict")
            vals.append(cov_dict[key])
        return torch.stack(vals, dim=-1)

    def forward(self, batch: dict[str, dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        control = batch["control"]
        perturb = batch["perturb"]
        gene_expr = control["expressions_hvg"].float()
        bs, topk = gene_expr.shape
        if topk != self.topk_genes:
            raise ValueError(f"model was built for {self.topk_genes} proteins, got {topk}")

        embedding_features = None
        drug_lookup = None
        protein_lookup = self.protein_embedding_features.to(gene_expr.device)
        if self.is_graph:
            assert self.hetero_graph is not None and self.embedding_proj is not None
            embedding_features = self.embedding_proj(self.hetero_graph.to(gene_expr.device))
            drug_lookup = embedding_features["drug"]
            protein_lookup = embedding_features["protein"]

        gene_feat = self.gene_proj(gene_expr)
        gene_feat = self._apply_gene_protein_fusion(gene_feat, protein_lookup)
        pert_feat, pert_token_count = self._perturbation_tokens(perturb, drug_lookup)
        batch_feat = self._batch_tokens(perturb, bs)
        target_feat, target_token_count = self._target_tokens(perturb, protein_lookup)

        if self.fusion_mode == "concat":
            token_parts = [gene_feat, pert_feat, batch_feat]
            if target_feat is not None:
                token_parts.append(target_feat)
            fused = torch.cat(token_parts, dim=-2)
        else:
            context = pert_feat.mean(-2)
            if batch_feat.shape[-2]:
                context = context + batch_feat.mean(-2)
            if target_feat is not None and self.is_graph:
                context = context + target_feat.mean(-2)
            fused = gene_feat + context.unsqueeze(-2)

        _, n_tokens, hid = fused.shape
        token_type = self._token_type(
            bs=bs,
            topk=topk,
            n_tokens=n_tokens,
            pert_token_count=pert_token_count,
            batch_token_count=len(self.batch_cov_list),
            target_token_count=target_token_count,
            device=fused.device,
        )
        fused = fused + self.token_type_emb(token_type)
        cls_tok = self.cls_token.expand(bs, -1, -1)
        fused = torch.cat([cls_tok, fused], dim=1)
        out = self.transformer(fused)
        cls_output = out[:, 0, :]
        gene_output = out[:, 1 : topk + 1, :]
        perturbed_expr = self.expression_head(gene_output).squeeze(-1)
        logits = self.classification_head(cls_output)
        synergy_logits = self.synergy_head(cls_output)
        return perturbed_expr, logits, synergy_logits

    def _apply_gene_protein_fusion(self, gene_feat: torch.Tensor, protein_lookup: torch.Tensor) -> torch.Tensor:
        bs, _, _ = gene_feat.shape
        if self.static_gene_gate:
            protein_feat = self.input_protein_emb.to(gene_feat.device)
            protein_feat_aligned = self.fusion_proj(protein_feat).unsqueeze(0).expand(bs, -1, -1)
            combined = self.fusion_norm(torch.cat([gene_feat, protein_feat_aligned], dim=-1))
            gate = self.fusion_gate(combined)
            return gene_feat + gate * protein_feat_aligned * self.gate_weight
        if not self.is_graph:
            return gene_feat
        axis_index = self.ordered_protein_index.to(protein_lookup.device)
        protein_embedding = protein_lookup[axis_index].unsqueeze(0).expand(bs, -1, -1)
        if self.graph_dropout:
            protein_embedding = self.emb_dropout(protein_embedding)
        if self.graph_gene_gate:
            protein_feat_aligned = self.fusion_proj(protein_embedding)
            combined = self.fusion_norm(torch.cat([gene_feat, protein_feat_aligned], dim=-1))
            gate = self.fusion_gate(combined)
            return gene_feat + gate * protein_feat_aligned * self.gate_weight
        return gene_feat + protein_embedding

    def _perturbation_tokens(
        self,
        perturb: dict[str, torch.Tensor],
        drug_lookup: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int]:
        pert = perturb["pert_id"]
        if self.is_graph:
            if drug_lookup is None:
                raise ValueError("graph model expected drug_lookup")
            indices = pert.long().to(drug_lookup.device).clamp(min=0, max=drug_lookup.shape[0] - 1)
            pert_covariates = drug_lookup[indices]
            if not self.include_target:
                if self.perturb_fusion_mode == "mlp":
                    return self.pert_proj(pert_covariates), int(pert_covariates.shape[-2])
                return pert_covariates, int(pert_covariates.shape[-2])
        else:
            pert_covariates = pert.float()
        if self.perturb_fusion_mode == "add":
            pert_feat = self.pert_proj(pert_covariates.sum(dim=-2, keepdim=True))
            return pert_feat, 1
        pert_feat = self.pert_proj(pert_covariates)
        return pert_feat, int(pert_covariates.shape[-2])

    def _batch_tokens(self, perturb: dict[str, torch.Tensor], bs: int) -> torch.Tensor:
        if not self.batch_cov_list:
            return torch.zeros(bs, 0, self.hidden_dim, device=next(self.parameters()).device)
        batch_cov = self._dict_to_tensor(perturb, self.batch_cov_list)
        return self.batch_proj(batch_cov.long())

    def _target_tokens(
        self,
        perturb: dict[str, torch.Tensor],
        protein_lookup: torch.Tensor,
    ) -> tuple[torch.Tensor | None, int]:
        if not self.include_target:
            return None, 0
        target_indices = perturb["target_protein_list"].long().to(protein_lookup.device)
        target_n = int(target_indices.shape[-1])
        zero_row = protein_lookup.new_zeros(1, protein_lookup.shape[1])
        target_embedding = torch.cat([protein_lookup, zero_row], dim=0)
        target_indices = target_indices.clamp(min=0, max=target_embedding.shape[0] - 1)
        return self.target_proj(target_embedding[target_indices]), target_n

    def _token_type(
        self,
        *,
        bs: int,
        topk: int,
        n_tokens: int,
        pert_token_count: int,
        batch_token_count: int,
        target_token_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.fusion_mode != "concat":
            return torch.zeros(bs, n_tokens, dtype=torch.long, device=device)
        pieces = [
            torch.zeros(bs, topk, dtype=torch.long, device=device),
            torch.ones(bs, pert_token_count, dtype=torch.long, device=device),
            2 * torch.ones(bs, batch_token_count, dtype=torch.long, device=device),
        ]
        if target_token_count:
            pieces.append(3 * torch.ones(bs, target_token_count, dtype=torch.long, device=device))
        return torch.cat(pieces, dim=-1)


class BaselineEmbV3(nn.Module):
    """Legacy ``Baseline_emb_v3`` adapted to the double-drug perturbation tensor."""

    def __init__(
        self,
        *,
        topk_genes: int,
        gene_emb_dim: int,
        drug_embedding_dim: int,
        hidden_dim: int,
        perturb_fusion_mode: str,
        dropout: float,
        fusion_mode: str,
        emb_dataset_path: str | None,
    ) -> None:
        super().__init__()
        if not emb_dataset_path:
            raise ValueError("baseline_emb_v3 requires --emb-dataset-path")
        if not Path(emb_dataset_path).exists():
            raise FileNotFoundError(f"baseline_emb_v3 emb dataset not found: {emb_dataset_path}")
        if perturb_fusion_mode not in {"add", "concat"}:
            raise ValueError("baseline_emb_v3 supports perturb_fusion_mode add or concat")
        self.topk_genes = int(topk_genes)
        self.gene_emb_dim = int(gene_emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.perturb_fusion_mode = perturb_fusion_mode
        self.fusion_mode = fusion_mode
        emb_dataset = np.load(emb_dataset_path).astype(np.float32, copy=False)
        if emb_dataset.shape[1] != gene_emb_dim:
            raise ValueError(
                f"emb_dataset second dimension {emb_dataset.shape[1]} does not match gene_emb_dim {gene_emb_dim}"
            )
        self.register_buffer("emb_dataset", torch.tensor(emb_dataset, dtype=torch.float32), persistent=False)
        pert_cov_dim = drug_embedding_dim * 2 if perturb_fusion_mode == "concat" else drug_embedding_dim
        self.gene_proj = nn.Sequential(
            nn.Linear(gene_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pert_covariate_proj = nn.Sequential(
            nn.Linear(pert_cov_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        fusion_input_dim = hidden_dim * 2 if fusion_mode == "concat" else hidden_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        head_dim = max(1, hidden_dim // 2)
        self.expression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, topk_genes),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )
        self.synergy_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    def forward(self, batch: dict[str, dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        control = batch["control"]
        perturb = batch["perturb"]
        index = control["index"].long().to(self.emb_dataset.device)
        if torch.any(index < 0) or torch.any(index >= self.emb_dataset.shape[0]):
            raise ValueError(
                "baseline_emb_v3 emb dataset row count does not cover feature row indices; "
                f"max index={int(index.max().item())}, emb rows={self.emb_dataset.shape[0]}"
            )
        gene_expression = self.emb_dataset[index].float().to(index.device)
        pert_covariates = perturb["pert_id"].float()
        if pert_covariates.dim() == 3:
            if self.perturb_fusion_mode == "concat":
                pert_covariates = pert_covariates.reshape(pert_covariates.shape[0], -1)
            else:
                pert_covariates = pert_covariates.sum(dim=-2)
        gene_features = self.gene_proj(gene_expression)
        pert_features = self.pert_covariate_proj(pert_covariates)
        if self.fusion_mode == "concat":
            fused = torch.cat([gene_features, pert_features], dim=-1)
        else:
            fused = gene_features + pert_features
        hidden = self.fusion_mlp(fused)
        return self.expression_head(hidden), self.classification_head(hidden), self.synergy_head(hidden)


def build_model(
    model_name: str,
    *,
    artifacts: ModelArtifacts,
    topk_genes: int,
    batch_cov_list: list[str],
    batch_cov_category_sizes: list[int],
    hidden_dim: int,
    perturb_fusion_mode: str,
    target_protein_max_length: int,
    dropout: float = 0.1,
    fusion_mode: str = "concat",
    num_heads: int = 8,
    num_layers: int = 4,
    cls_type: str = "all_1",
    graph_dropout: bool = False,
    use_target: bool = True,
    target_protein_fusion_model: str = "concat",
    gate_weight: float = 1.0,
    reverse_pdi: bool = True,
    emb_dataset_path: str | None = None,
    gene_emb_dim: int = 768,
) -> nn.Module:
    if model_name not in SELECTED_MODEL_NAMES:
        raise ValueError(f"model_name must be one of {sorted(SELECTED_MODEL_NAMES)}")
    del target_protein_max_length
    if model_name == "baseline_emb_v3":
        return BaselineEmbV3(
            topk_genes=topk_genes,
            gene_emb_dim=gene_emb_dim,
            drug_embedding_dim=int(artifacts.drug_embedding.shape[1]),
            hidden_dim=hidden_dim,
            perturb_fusion_mode=perturb_fusion_mode,
            dropout=dropout,
            fusion_mode=fusion_mode,
            emb_dataset_path=emb_dataset_path,
        )
    return LegacyDoubleDrugTransformer(
        model_name=model_name,
        topk_genes=topk_genes,
        batch_cov_list=batch_cov_list,
        batch_cov_category_sizes=batch_cov_category_sizes,
        hidden_dim=hidden_dim,
        perturb_fusion_mode=perturb_fusion_mode,
        protein_embedding=artifacts.protein_embedding,
        drug_embedding=artifacts.drug_embedding,
        ordered_protein_index=artifacts.ordered_protein_index,
        pdi_matrix=artifacts.pdi_matrix,
        dropout=dropout,
        fusion_mode=fusion_mode,
        num_heads=num_heads,
        num_layers=num_layers,
        cls_type=cls_type,
        graph_dropout=graph_dropout,
        use_target=use_target,
        target_protein_fusion_model=target_protein_fusion_model,
        gate_weight=gate_weight,
        reverse_pdi=reverse_pdi,
    )
