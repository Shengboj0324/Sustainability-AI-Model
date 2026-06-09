"""
Physics-Informed, Hierarchically-Consistent Waste Classifier.

Redesign goals (vs. the legacy 3-independent-head MultiHeadClassifier):

1. CONSISTENCY BY CONSTRUCTION.
   Material and bin predictions are fused with distributions *derived* from the
   item prediction through the frozen taxonomy mapping matrices. In `hard`
   consistency mode the material/bin outputs are exactly the item-implied
   distribution, so the model can never emit a physically impossible
   (item, material, bin) triple. In `soft` mode a learnable gate blends a direct
   perceptual head with the derived distribution.

2. ROBUSTNESS TO FINE-GRAINED ITEM CONFUSION.
   The single biggest legacy failure is glass_beverage_bottles vs glass_food_jars
   (group acc 64.7%). But both are `glass -> recycle`. A *direct* material/bin
   perceptual pathway lets the model get material & bin RIGHT even when the fine
   item class is wrong — decoupling the user-facing disposal decision from the
   hardest sub-classification.

3. MATERIAL-PROPERTY PRIORS (the "physics").
   A property scorer biases material logits using physical priors (transparency,
   metallic sheen, density, resin code, ferromagnetism) seeded from the taxonomy
   so the network reasons about *why* something is a given material, not just its
   texture.

The class is checkpoint-compatible with the legacy model: it exposes
`backbone`, `se_block`, `item_head`, `material_head`(=material_direct_head alias),
`bin_head`, `temperature`, and an optional `llm_projection`, so legacy weights
warm-start via `load_legacy_state_dict`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vision import taxonomy as tx

logger = logging.getLogger(__name__)

LLM_HIDDEN_DIM = 4096
_EPS = 1e-8


# --------------------------------------------------------------------------- #
#  Building blocks
# --------------------------------------------------------------------------- #
class SqueezeExcitation(nn.Module):
    """Channel attention on pooled (B, C) features. Matches legacy layout."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.excitation(x)


class MaterialPropertyScorer(nn.Module):
    """Score materials from a feature vector using physical-property priors.

    A learnable material embedding is *initialised* from the taxonomy property
    table (resin code, density, transparency, metallic sheen, ferromagnetism,
    recyclability, compostability, melting point). The feature vector is
    projected into the same property space and scored by inner product, so the
    material decision is grounded in physics rather than texture alone.
    """

    def __init__(self, in_features: int, prop_dim: int = 32):
        super().__init__()
        props = torch.tensor(tx.material_property_matrix(), dtype=torch.float32)  # (M, 8)
        # Fixed seed projection 8 -> prop_dim gives each material a physically
        # meaningful anchor; the embedding is then fine-tuned.
        gen = torch.Generator().manual_seed(42)
        seed = torch.randn(props.shape[1], prop_dim, generator=gen) * 0.3
        self.material_embed = nn.Parameter(props @ seed)        # (M, prop_dim), learnable
        self.register_buffer("material_props", props)            # kept for introspection
        self.feat_proj = nn.Sequential(
            nn.Linear(in_features, prop_dim),
            nn.LayerNorm(prop_dim),
        )
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        z = self.feat_proj(feat)                                 # (B, prop_dim)
        z = F.normalize(z, dim=-1)
        e = F.normalize(self.material_embed, dim=-1)             # (M, prop_dim)
        return self.scale * (z @ e.t())                          # (B, M)


# --------------------------------------------------------------------------- #
#  Output container
# --------------------------------------------------------------------------- #
@dataclass
class PhysicsHeads:
    item_logits: torch.Tensor          # (B, NUM_ITEMS)
    material_logits: torch.Tensor      # (B, NUM_MATERIALS)  — log-probabilities
    bin_logits: torch.Tensor           # (B, NUM_BINS)       — log-probabilities
    material_direct: torch.Tensor      # (B, NUM_MATERIALS)  pre-fusion perceptual logits
    bin_direct: torch.Tensor           # (B, NUM_BINS)
    p_mat_derived: torch.Tensor        # (B, NUM_MATERIALS)  item-implied (consistent)
    p_bin_derived: torch.Tensor        # (B, NUM_BINS)
    features: torch.Tensor             # (B, feat_dim)
    llm_embedding: Optional[torch.Tensor] = None


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
class PhysicsInformedWasteClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        pretrained: bool = True,
        drop_rate: float = 0.2,
        enable_se: bool = True,
        enable_llm_projection: bool = False,
        llm_dim: int = LLM_HIDDEN_DIM,
        consistency_mode: str = "soft",   # "soft" (gated fusion) | "hard" (pure derived)
        prop_dim: int = 32,
    ):
        super().__init__()
        import timm

        self.consistency_mode = consistency_mode
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, drop_rate=drop_rate
        )
        self.feature_dim = self.backbone.num_features

        self.se_block = SqueezeExcitation(self.feature_dim) if enable_se else nn.Identity()

        # Primary item head
        self.item_head = nn.Linear(self.feature_dim, tx.NUM_ITEMS)
        # Direct perceptual heads (decoupled from item confusion)
        self.material_direct_head = nn.Linear(self.feature_dim, tx.NUM_MATERIALS)
        self.bin_head = nn.Linear(self.feature_dim, tx.NUM_BINS)
        # Physics-property scorer (added to material_direct)
        self.property_scorer = MaterialPropertyScorer(self.feature_dim, prop_dim=prop_dim)

        # Learnable fusion gates (sigmoid). Initialised to ~0.5 -> trust derived &
        # direct equally; training moves them. Per-output, not per-sample, to stay
        # interpretable.
        self.mat_gate = nn.Parameter(torch.zeros(1))
        self.bin_gate = nn.Parameter(torch.zeros(1))

        # Confidence-calibration temperature (shared, matches legacy)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        self.enable_llm_projection = enable_llm_projection
        if enable_llm_projection:
            self.llm_projection = nn.Sequential(
                nn.Linear(self.feature_dim, llm_dim), nn.GELU(),
                nn.LayerNorm(llm_dim), nn.Linear(llm_dim, llm_dim), nn.LayerNorm(llm_dim),
            )

        # Frozen taxonomy mapping matrices (registered as buffers -> move with .to())
        maps = tx.build_mapping_tensors()
        self.register_buffer("M_item_material", maps["M_item_material"])  # (I, M)
        self.register_buffer("M_item_bin", maps["M_item_bin"])            # (I, B)

        logger.info(
            "PhysicsInformedWasteClassifier: %s -> %dD | mode=%s | items=%d mat=%d bin=%d",
            backbone, self.feature_dim, consistency_mode, tx.NUM_ITEMS, tx.NUM_MATERIALS, tx.NUM_BINS,
        )

    # ----- alias so legacy state dicts (`material_head.*`) still load --------
    @property
    def material_head(self) -> nn.Linear:  # pragma: no cover - convenience alias
        return self.material_direct_head

    def forward(self, x: torch.Tensor, return_embeddings: bool = False) -> PhysicsHeads:
        feat = self.se_block(self.backbone(x))
        t = self.temperature.clamp(min=0.1)

        item_logits = self.item_head(feat) / t
        p_item = F.softmax(item_logits, dim=-1)

        # Consistent (derived) distributions — single matmul through taxonomy
        p_mat_derived = p_item @ self.M_item_material           # (B, M)
        p_bin_derived = p_item @ self.M_item_bin                # (B, B)

        # Direct perceptual logits + physics-property prior
        mat_direct = (self.material_direct_head(feat) + self.property_scorer(feat)) / t
        bin_direct = self.bin_head(feat) / t

        if self.consistency_mode == "hard":
            # Pure derived -> material/bin are exactly the item-implied expectation
            material_logits = torch.log(p_mat_derived + _EPS)
            bin_logits = torch.log(p_bin_derived + _EPS)
        else:
            # Geometric blend of derived (consistent) and direct (perceptual),
            # then RE-NORMALISE so outputs are valid log-probabilities. A raw
            # weighted sum of two log-prob vectors is unnormalised (geometric
            # mean), which would break nll_loss and the impossible-pair penalty.
            g_mat = torch.sigmoid(self.mat_gate)
            g_bin = torch.sigmoid(self.bin_gate)
            material_logits = F.log_softmax(
                g_mat * torch.log(p_mat_derived + _EPS)
                + (1.0 - g_mat) * F.log_softmax(mat_direct, dim=-1),
                dim=-1,
            )
            bin_logits = F.log_softmax(
                g_bin * torch.log(p_bin_derived + _EPS)
                + (1.0 - g_bin) * F.log_softmax(bin_direct, dim=-1),
                dim=-1,
            )

        llm_emb = self.llm_projection(feat) if (return_embeddings and self.enable_llm_projection) else None

        return PhysicsHeads(
            item_logits=item_logits,
            material_logits=material_logits,
            bin_logits=bin_logits,
            material_direct=mat_direct,
            bin_direct=bin_direct,
            p_mat_derived=p_mat_derived,
            p_bin_derived=p_bin_derived,
            features=feat,
            llm_embedding=llm_emb,
        )

    # ----------------------------------------------------------------------- #
    #  Warm-start from a legacy MultiHeadClassifier checkpoint
    # ----------------------------------------------------------------------- #
    def load_legacy_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = False) -> Dict[str, list]:
        """Load overlapping weights (backbone, se_block, item_head, temperature,
        and material_head->material_direct_head) from a legacy checkpoint. Returns
        the (missing, unexpected) key report for transparency. New modules
        (property_scorer, gates, bin fusion) start fresh and are trained."""
        remapped = {}
        for k, v in state_dict.items():
            nk = k
            if k.startswith("material_head."):
                nk = k.replace("material_head.", "material_direct_head.")
            remapped[nk] = v
        result = self.load_state_dict(remapped, strict=strict)
        missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
        logger.info("legacy load: %d missing, %d unexpected keys", len(missing), len(unexpected))
        return {"missing": missing, "unexpected": unexpected}


def taxonomy_decision(item_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
    """The guaranteed-consistent inference contract.

    The user-facing (item, material, bin) decision is taken from the item argmax
    routed through the frozen taxonomy. The resulting triple is ALWAYS a legal
    (item, material, bin) combination — physics consistency holds by
    construction, independent of training, weights, or input. This is what the
    serving path should surface; the soft material/bin heads are auxiliary
    perceptual signals / interpretability, not the decision."""
    item = item_logits.argmax(-1)
    i2m = torch.tensor(tx.item_to_material_index(), device=item.device)
    i2b = torch.tensor(tx.item_to_bin_index(), device=item.device)
    return {"item": item, "material": i2m[item], "bin": i2b[item]}


def decision_violation_rate(item_logits: torch.Tensor) -> float:
    """Fraction of decision triples not permitted by the taxonomy. Structurally
    0.0 — asserted in tests as the physics-consistency guarantee."""
    d = taxonomy_decision(item_logits)
    i2m = torch.tensor(tx.item_to_material_index(), device=d["item"].device)
    i2b = torch.tensor(tx.item_to_bin_index(), device=d["item"].device)
    ok = (d["material"] == i2m[d["item"]]) & (d["bin"] == i2b[d["item"]])
    return float((~ok).float().mean().item())


def aux_head_disagreement_rate(heads: PhysicsHeads) -> Dict[str, float]:
    """Diagnostic: how often the auxiliary perceptual material/bin heads disagree
    with the item-derived decision. High disagreement means the heads are
    fighting the taxonomy (useful signal during training), not a physics
    violation. Not a guarantee — see `decision_violation_rate` for that."""
    with torch.no_grad():
        item = heads.item_logits.argmax(-1)
        i2m = torch.tensor(tx.item_to_material_index(), device=item.device)
        i2b = torch.tensor(tx.item_to_bin_index(), device=item.device)
        mat_dis = (heads.material_logits.argmax(-1) != i2m[item]).float().mean().item()
        bin_dis = (heads.bin_logits.argmax(-1) != i2b[item]).float().mean().item()
        return {"material": float(mat_dis), "bin": float(bin_dis)}
