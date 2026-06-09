"""
Physics-informed multi-task loss for the waste classifier.

Components
----------
1. Logit-adjusted item loss  (Menon et al., ICLR 2021)
   Adds log(class_prior) to logits so the gradient stops being dominated by the
   mega-classes (clothing 11k, shoes 4.4k) at the expense of the 500-image
   tail. Directly targets the imbalance that inflates the legacy 90% headline.

2. Material / bin cross-entropy on the *fused* (already-consistent) log-prob
   outputs, with labels derived from the item ground truth via the taxonomy.

3. Consistency loss
   KL(direct-perceptual || item-derived.detach) pulls the direct material/bin
   heads toward the physically-consistent region without collapsing them.

4. Impossible-pair penalty
   Probability mass placed on (material, bin) combinations that never legally
   co-occur is penalised explicitly — a hard physics constraint.

5. Confusion-group supervised contrastive (optional)
   Pulls apart embeddings *within* the glass / film / metal groups so the item
   head can finally separate glass_beverage_bottles from glass_food_jars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vision import taxonomy as tx
from models.vision.physics_informed_classifier import PhysicsHeads, _EPS


def build_valid_material_bin_mask(device=None) -> torch.Tensor:
    """(NUM_MATERIALS, NUM_BINS) 1.0 where the pair legally co-occurs for some
    item, else 0.0. Derived purely from the taxonomy."""
    M = torch.zeros(tx.NUM_MATERIALS, tx.NUM_BINS, device=device)
    i2m = tx.item_to_material_index()
    i2b = tx.item_to_bin_index()
    for m, b in zip(i2m, i2b):
        M[m, b] = 1.0
    return M


@dataclass
class LossConfig:
    w_item: float = 1.0
    w_material: float = 1.0
    w_bin: float = 0.7
    w_consistency: float = 0.5
    w_impossible: float = 0.5
    w_contrastive: float = 0.2
    logit_adjust_tau: float = 1.0      # strength of logit adjustment (0 disables)
    label_smoothing: float = 0.05
    contrastive_temp: float = 0.1


class PhysicsInformedLoss(nn.Module):
    def __init__(self, class_log_prior: Optional[torch.Tensor] = None, cfg: Optional[LossConfig] = None):
        super().__init__()
        self.cfg = cfg or LossConfig()
        # class_log_prior: (NUM_ITEMS,) log frequency of each item class in train set
        if class_log_prior is None:
            class_log_prior = torch.zeros(tx.NUM_ITEMS)
        self.register_buffer("class_log_prior", class_log_prior)
        self.register_buffer("valid_mat_bin", build_valid_material_bin_mask())
        self.register_buffer("item2mat", torch.tensor(tx.item_to_material_index()))
        self.register_buffer("item2bin", torch.tensor(tx.item_to_bin_index()))
        # group membership vector: item idx -> group id (or -1)
        grp = torch.full((tx.NUM_ITEMS,), -1, dtype=torch.long)
        for gi, (_, members) in enumerate(tx.CONFUSION_GROUPS.items()):
            for m in members:
                grp[tx.ITEM_TO_IDX[m]] = gi
        self.register_buffer("item_group", grp)

    # ---- individual terms -------------------------------------------------
    def _item_loss(self, item_logits: torch.Tensor, item_labels: torch.Tensor) -> torch.Tensor:
        adjusted = item_logits + self.cfg.logit_adjust_tau * self.class_log_prior.unsqueeze(0)
        return F.cross_entropy(adjusted, item_labels, label_smoothing=self.cfg.label_smoothing)

    def _nll(self, log_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # material/bin heads already output log-probabilities (fused)
        return F.nll_loss(log_probs, labels)

    def _consistency_loss(self, heads: PhysicsHeads) -> torch.Tensor:
        p_mat_direct = F.softmax(heads.material_direct, dim=-1)
        p_bin_direct = F.softmax(heads.bin_direct, dim=-1)
        km = F.kl_div((p_mat_direct + _EPS).log(), heads.p_mat_derived.detach() + _EPS, reduction="batchmean")
        kb = F.kl_div((p_bin_direct + _EPS).log(), heads.p_bin_derived.detach() + _EPS, reduction="batchmean")
        return km + kb

    def _impossible_pair_penalty(self, heads: PhysicsHeads) -> torch.Tensor:
        p_mat = heads.material_logits.exp()      # (B, M) — log-probs -> probs
        p_bin = heads.bin_logits.exp()           # (B, B)
        joint = torch.einsum("bm,bn->bmn", p_mat, p_bin)   # (B, M, B)
        invalid = (1.0 - self.valid_mat_bin).unsqueeze(0)  # (1, M, B)
        return (joint * invalid).sum(dim=(1, 2)).mean()

    def _group_contrastive(self, feat: torch.Tensor, item_labels: torch.Tensor) -> torch.Tensor:
        """SupCon restricted to samples that belong to a confusion group, with
        positives = same item class. Separates within-group fine classes."""
        groups = self.item_group[item_labels]
        mask_in_group = groups >= 0
        if mask_in_group.sum() < 2:
            return feat.new_zeros(())
        f = F.normalize(feat[mask_in_group], dim=-1)
        lbl = item_labels[mask_in_group]
        sim = (f @ f.t()) / self.cfg.contrastive_temp
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp = torch.exp(sim)
        self_mask = torch.eye(f.size(0), device=f.device, dtype=torch.bool)
        exp = exp.masked_fill(self_mask, 0.0)
        pos = (lbl.unsqueeze(0) == lbl.unsqueeze(1)) & ~self_mask
        denom = exp.sum(1)
        pos_sum = (exp * pos).sum(1)
        valid = pos.sum(1) > 0
        if valid.sum() == 0:
            return feat.new_zeros(())
        log_prob = torch.log(pos_sum[valid] + _EPS) - torch.log(denom[valid] + _EPS)
        return -log_prob.mean()

    # ---- public -----------------------------------------------------------
    def forward(self, heads: PhysicsHeads, item_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        mat_labels = self.item2mat[item_labels]
        bin_labels = self.item2bin[item_labels]
        c = self.cfg

        l_item = self._item_loss(heads.item_logits, item_labels)
        l_mat = self._nll(heads.material_logits, mat_labels)
        l_bin = self._nll(heads.bin_logits, bin_labels)
        l_cons = self._consistency_loss(heads)
        l_imp = self._impossible_pair_penalty(heads)
        l_con = self._group_contrastive(heads.features, item_labels) if c.w_contrastive > 0 else heads.features.new_zeros(())

        total = (
            c.w_item * l_item + c.w_material * l_mat + c.w_bin * l_bin
            + c.w_consistency * l_cons + c.w_impossible * l_imp + c.w_contrastive * l_con
        )
        return {
            "total": total, "item": l_item.detach(), "material": l_mat.detach(),
            "bin": l_bin.detach(), "consistency": l_cons.detach(),
            "impossible": l_imp.detach(), "contrastive": l_con.detach(),
        }


def compute_class_log_prior(class_counts: torch.Tensor) -> torch.Tensor:
    """log(p_c) from per-item training counts, for logit adjustment."""
    counts = class_counts.float().clamp(min=1.0)
    return torch.log(counts / counts.sum())
