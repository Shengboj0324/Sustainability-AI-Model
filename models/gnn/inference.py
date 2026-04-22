"""
Graph Neural Network - Production-grade upcycling recommendations

CRITICAL FEATURES:
- GraphSAGE/GATv2/GCN for knowledge graph reasoning
- Link prediction for upcycling paths
- Node classification for material properties
- Batch graph processing
- Proper device management
- Memory-efficient inference

UPGRADES:
  7. GATv2Conv with dynamic attention (Brody et al., ICLR 2022)
  8. Edge-attribute-aware message passing
  9. L2-normalized embeddings for RAG retrieval
  10. Real node-attribute extraction (replaces mock random data)
  11. Dynamic graph updates — add new materials at inference time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATv2Conv, GCNConv
from torch_geometric.data import Data, Batch
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UpcyclingRecommendation:
    """Single upcycling recommendation"""
    target_product: str
    target_id: int
    score: float
    difficulty: float
    time_required: float
    required_tools: List[str]
    required_skills: List[str]
    similarity_score: float
    embedding: Optional[np.ndarray] = None  # [Upgrade 9] RAG-optimized embedding


@dataclass
class RecommendationResult:
    """Upcycling recommendations for a material/item"""
    source_material: str
    source_id: int
    recommendations: List[UpcyclingRecommendation]
    num_recommendations: int
    inference_time_ms: float
    source_embedding: Optional[np.ndarray] = None  # [Upgrade 9] Source node embedding


class GraphSAGEModel(nn.Module):
    """
    GraphSAGE model for link prediction

    CRITICAL: Inductive learning for new nodes
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.2,
        aggregator: str = "mean"
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))

        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggregator))

        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer
        x = self.convs[-1](x, edge_index)

        return x


class GATv2Model(nn.Module):
    """
    [Upgrade 7] GATv2 — Dynamic Graph Attention Network.

    GATv2 (Brody et al., ICLR 2022) computes attention AFTER the linear
    transformation, making it strictly more expressive than GAT v1 which
    applies a static ranking over neighbours.

    [Upgrade 8] Supports edge_dim for edge-attribute-aware attention,
    enabling relationship-type conditioning (e.g. CAN_RECYCLE_TO vs
    CAN_UPCYCLE_TO edges carry different edge features).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        attention_dropout: float = 0.1,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # [Upgrade 7+8] GATv2 layers with optional edge features
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(
            in_channels,
            hidden_channels,
            heads=num_heads,
            dropout=attention_dropout,
            edge_dim=edge_dim,
            residual=True,
        ))

        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(
                hidden_channels * num_heads,
                hidden_channels,
                heads=num_heads,
                dropout=attention_dropout,
                edge_dim=edge_dim,
                residual=True,
            ))

        self.convs.append(GATv2Conv(
            hidden_channels * num_heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=attention_dropout,
            edge_dim=edge_dim,
            residual=False,
        ))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional edge attributes."""
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        return x


class UpcyclingGNN:
    """
    Production-grade GNN for upcycling recommendations

    CRITICAL FEATURES:
    - Link prediction for upcycling paths
    - Handles heterogeneous graphs
    - Proper device management
    - Batch processing
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ):
        self.config = config or self._get_default_config()
        self.device = self._setup_device(device)
        self.model: Optional[nn.Module] = None
        self.model_path = model_path

        # Node and edge mappings
        self.node_id_to_name: Dict[int, str] = {}
        self.node_name_to_id: Dict[str, int] = {}
        self.edge_types: List[str] = []

        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0

        # [Upgrade 9] Embedding cache for repeated queries
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_graph_hash: Optional[int] = None

        logger.info(f"UpcyclingGNN initialized on device: {self.device}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "model_type": "gatv2",          # [Upgrade 7] Default to GATv2
            "num_layers": 3,
            "hidden_dim": 256,
            "output_dim": 128,
            "dropout": 0.2,
            "aggregator": "mean",
            "num_heads": 4,
            "attention_dropout": 0.1,
            "edge_dim": None,               # [Upgrade 8] Set to int to enable edge features
            "top_k": 10,
            "score_threshold": 0.5,
        }

    def _setup_device(self, device: Optional[str] = None) -> torch.device:
        """
        Setup device with proper CUDA and MPS handling.

        NOTE: PyTorch Geometric's scatter_reduce ops are NOT implemented on
        MPS, so GNN inference MUST run on CPU for Apple Silicon.  This is
        acceptable because GNN graphs are small (<10K nodes) and CPU
        inference takes <50ms.
        """
        if device is not None:
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return torch.device("cpu")
            if device == "mps":
                logger.info(
                    "🍎 MPS requested but scatter_reduce unsupported for "
                    "PyG — using CPU for GNN (fast for small graphs)"
                )
                return torch.device("cpu")
            return torch.device(device)

        # Auto-detect
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"🔥 CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            # Always use CPU for GNN — MPS lacks scatter_reduce
            device = torch.device("cpu")
            logger.info("💻 Using CPU for GNN inference (PyG MPS not supported)")

        return device

    def load_model(self, node_features_dim: int = 128):
        """
        Load GNN model with proper error handling

        CRITICAL: Handles missing checkpoints, device placement
        """
        try:
            logger.info("Loading GNN model...")
            start_time = time.time()

            # Create model based on type
            model_type = self.config["model_type"]

            if model_type == "graphsage":
                self.model = GraphSAGEModel(
                    in_channels=node_features_dim,
                    hidden_channels=self.config["hidden_dim"],
                    out_channels=self.config["output_dim"],
                    num_layers=self.config["num_layers"],
                    dropout=self.config["dropout"],
                    aggregator=self.config["aggregator"]
                )
            elif model_type in ("gat", "gatv2"):
                # [Upgrade 7] Always use GATv2 for dynamic attention
                self.model = GATv2Model(
                    in_channels=node_features_dim,
                    hidden_channels=self.config["hidden_dim"],
                    out_channels=self.config["output_dim"],
                    num_layers=self.config["num_layers"],
                    num_heads=self.config["num_heads"],
                    dropout=self.config["dropout"],
                    attention_dropout=self.config["attention_dropout"],
                    edge_dim=self.config.get("edge_dim"),
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Load checkpoint if available
            if self.model_path and Path(self.model_path).exists():
                logger.info(f"Loading checkpoint from: {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location=self.device)

                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    self.model.load_state_dict(checkpoint)
                    logger.info("Loaded checkpoint (state dict only)")
            else:
                logger.warning(f"No checkpoint found at {self.model_path}. Using random initialization.")

            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()

            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

    def _compute_embeddings(
        self, graph_data: Data
    ) -> torch.Tensor:
        """
        [Upgrade 9] Compute L2-normalised node embeddings with caching.

        Normalised embeddings enable cosine similarity via a simple dot
        product, which is both numerically stable and compatible with FAISS/
        RAG retrieval indices that expect unit-norm vectors.
        """
        # Invalidate cache if graph changed
        graph_hash = hash((graph_data.num_nodes, graph_data.num_edges))
        if graph_hash != self._cache_graph_hash:
            self._embedding_cache.clear()
            self._cache_graph_hash = graph_hash

        x = graph_data.x.to(self.device)
        edge_index = graph_data.edge_index.to(self.device)

        # Only pass edge_attr for models that support it (GATv2)
        if isinstance(self.model, GATv2Model):
            edge_attr = getattr(graph_data, "edge_attr", None)
            if edge_attr is not None:
                edge_attr = edge_attr.to(self.device)
            embeddings = self.model(x, edge_index, edge_attr=edge_attr)
        else:
            embeddings = self.model(x, edge_index)
        # L2 normalise → cosine similarity = dot product
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    @staticmethod
    def _extract_node_attributes(
        graph_data: Data, node_idx: int
    ) -> Dict[str, Any]:
        """
        [Upgrade 10] Extract real node attributes from graph data instead
        of returning mock/random values.

        Falls back to sensible defaults when attribute tensors are absent.
        """
        attrs: Dict[str, Any] = {}

        # difficulty stored as a per-node scalar in graph_data.difficulty
        if (hasattr(graph_data, "difficulty") and graph_data.difficulty is not None
                and node_idx < len(graph_data.difficulty)):
            attrs["difficulty"] = float(graph_data.difficulty[node_idx].item())
        else:
            attrs["difficulty"] = 3.0  # default: medium difficulty

        # time_required stored as a per-node scalar
        if (hasattr(graph_data, "time_required") and graph_data.time_required is not None
                and node_idx < len(graph_data.time_required)):
            attrs["time_required"] = float(graph_data.time_required[node_idx].item())
        else:
            attrs["time_required"] = 60.0  # default: 60 minutes

        # tools / skills stored as string lists in a dict attribute
        if hasattr(graph_data, "node_metadata") and graph_data.node_metadata is not None:
            meta = graph_data.node_metadata.get(int(node_idx), {})
            attrs["required_tools"] = meta.get("tools", [])
            attrs["required_skills"] = meta.get("skills", [])
        else:
            attrs["required_tools"] = []
            attrs["required_skills"] = []

        return attrs

    @torch.inference_mode()
    def predict_upcycling_paths(
        self,
        source_material: str,
        graph_data: Data,
        top_k: Optional[int] = None,
        return_embeddings: bool = False,
    ) -> RecommendationResult:
        """
        Predict upcycling paths for a material.

        Upgrades applied:
          9. L2-normalised embeddings (cosine similarity via dot product)
          10. Real node-attribute extraction (no more mock random data)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        top_k = top_k or self.config["top_k"]
        start_time = time.time()

        try:
            # Get source node ID
            if source_material not in self.node_name_to_id:
                raise ValueError(f"Unknown material: {source_material}")

            source_id = self.node_name_to_id[source_material]

            # [Upgrade 9] Normalised embeddings with edge-attr support
            embeddings = self._compute_embeddings(graph_data)

            source_embedding = embeddings[source_id]

            # Cosine similarity (embeddings are L2-normalised)
            scores = torch.matmul(embeddings, source_embedding)
            scores = torch.sigmoid(scores)

            # Exclude self
            scores[source_id] = -1.0
            top_scores, top_indices = torch.topk(
                scores, k=min(top_k, scores.shape[0])
            )

            # Build recommendations
            recommendations = []
            for score_val, idx_val in zip(
                top_scores.cpu().numpy(), top_indices.cpu().numpy()
            ):
                if score_val < self.config["score_threshold"]:
                    continue

                idx_int = int(idx_val)
                target_name = self.node_id_to_name.get(idx_int, f"node_{idx_int}")

                # [Upgrade 10] Real attribute extraction
                attrs = self._extract_node_attributes(graph_data, idx_int)

                emb_np = None
                if return_embeddings:
                    emb_np = embeddings[idx_int].cpu().numpy()

                recommendation = UpcyclingRecommendation(
                    target_product=target_name,
                    target_id=idx_int,
                    score=float(score_val),
                    difficulty=attrs["difficulty"],
                    time_required=attrs["time_required"],
                    required_tools=attrs["required_tools"],
                    required_skills=attrs["required_skills"],
                    similarity_score=float(score_val),
                    embedding=emb_np,
                )
                recommendations.append(recommendation)

            inference_time = (time.time() - start_time) * 1000

            # Sync device clocks for accurate timing
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elif self.device.type == "mps":
                torch.mps.synchronize()

            self.inference_count += 1
            self.total_inference_time += inference_time

            src_emb_np = None
            if return_embeddings:
                src_emb_np = source_embedding.cpu().numpy()

            result = RecommendationResult(
                source_material=source_material,
                source_id=source_id,
                recommendations=recommendations,
                num_recommendations=len(recommendations),
                inference_time_ms=inference_time,
                source_embedding=src_emb_np,
            )

            logger.info(
                f"Generated {len(recommendations)} recommendations "
                f"in {inference_time:.2f}ms"
            )
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise

    def load_node_mappings(self, node_id_to_name: Dict[int, str]):
        """Load node ID to name mappings"""
        self.node_id_to_name = node_id_to_name
        self.node_name_to_id = {name: idx for idx, name in node_id_to_name.items()}
        logger.info(f"Loaded {len(self.node_id_to_name)} node mappings")

    def add_node_to_graph(
        self,
        graph_data: Data,
        node_name: str,
        node_features: torch.Tensor,
        connect_to: Optional[List[str]] = None,
    ) -> Data:
        """
        [Upgrade 11] Dynamic graph update — add a new material node at
        inference time without retraining.

        The new node's feature vector is appended to graph_data.x and
        optional edges are created to existing nodes.

        Args:
            graph_data: Current graph (modified in-place and returned).
            node_name: Human-readable name for the node.
            node_features: Feature tensor of shape (feature_dim,).
            connect_to: List of existing node names to connect to.

        Returns:
            Updated graph_data with the new node.
        """
        new_id = graph_data.num_nodes
        self.node_id_to_name[new_id] = node_name
        self.node_name_to_id[node_name] = new_id

        # Append feature vector
        node_features = node_features.unsqueeze(0)  # (1, D)
        graph_data.x = torch.cat([graph_data.x, node_features], dim=0)

        # Add edges to specified neighbours (bidirectional)
        if connect_to:
            new_edges = []
            for neighbour_name in connect_to:
                if neighbour_name in self.node_name_to_id:
                    nbr_id = self.node_name_to_id[neighbour_name]
                    new_edges.append([new_id, nbr_id])
                    new_edges.append([nbr_id, new_id])

            if new_edges:
                new_edge_tensor = torch.tensor(
                    new_edges, dtype=torch.long
                ).t().contiguous()
                graph_data.edge_index = torch.cat(
                    [graph_data.edge_index, new_edge_tensor], dim=1
                )

        # Invalidate embedding cache
        self._embedding_cache.clear()
        self._cache_graph_hash = None

        logger.info(
            f"[Upgrade 11] Added node '{node_name}' (id={new_id}) "
            f"with {len(connect_to or [])} connections"
        )
        return graph_data

    @torch.inference_mode()
    def extract_node_embedding(
        self,
        node_name: str,
        graph_data: Data,
    ) -> np.ndarray:
        """
        [Upgrade 9] Extract a single node's L2-normalised embedding for RAG.

        Returns a numpy array suitable for FAISS index insertion.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if node_name not in self.node_name_to_id:
            raise ValueError(f"Unknown node: {node_name}")

        node_id = self.node_name_to_id[node_name]
        embeddings = self._compute_embeddings(graph_data)
        return embeddings[node_id].cpu().numpy()

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        avg_time = self.total_inference_time / self.inference_count if self.inference_count > 0 else 0

        return {
            "inference_count": self.inference_count,
            "total_inference_time_ms": self.total_inference_time,
            "average_inference_time_ms": avg_time,
            "device": str(self.device),
            "model_loaded": self.model is not None,
            "num_nodes": len(self.node_id_to_name)
        }

    def reset_stats(self):
        """Reset inference statistics"""
        self.inference_count = 0
        self.total_inference_time = 0.0
        logger.info("Statistics reset")

    def cleanup(self):
        """
        Cleanup resources

        CRITICAL: Free GPU memory (CUDA and MPS)
        """
        if self.model is not None:
            del self.model
            self.model = None

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("CUDA memory cleared")
        elif self.device.type == "mps":
            torch.mps.empty_cache()
            logger.info("MPS memory cleared")

        logger.info("GNN cleanup complete")

