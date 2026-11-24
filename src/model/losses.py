#!/usr/bin/env python3
"""
Loss functions for recommendation

Implement InfoNCE, multi-view contrastive, Entity alignment, and other loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


def info_nce_loss(
    user_emb: torch.Tensor,
    pos_item_emb: torch.Tensor,
    neg_item_emb: torch.Tensor,
    temperature: float = 0.2
) -> torch.Tensor:
    """
    InfoNCE Contrastive Learning Loss (Recommendation Task)

    Args:
        user_emb: [batch_size, dim] - User embedding
        pos_item_emb: [batch_size, dim] - Positive item embedding
        neg_item_emb: [batch_size, num_neg, dim] - Negative item embeddings
        temperature: Temperature coefficient

    Returns:
        loss: scalar
    """
    # Positive pair score
    pos_score = (user_emb * pos_item_emb).sum(dim=-1) / temperature  # [batch_size]

    # Negative pairs scores
    neg_score = torch.bmm(
        neg_item_emb,
        user_emb.unsqueeze(-1)
    ).squeeze(-1) / temperature  # [batch_size, num_neg]

    # InfoNCE: -log( exp(pos) / (exp(pos) + sum(exp(neg))) )
    # Equivalent to CrossEntropy, where positive sample label=0
    logits = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)  # [batch_size, 1+num_neg]
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    loss = F.cross_entropy(logits, labels)

    return loss


def bpr_loss(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor
) -> torch.Tensor:
    """
    BPR (Bayesian Personalized Ranking) Loss

    Args:
        pos_score: [batch_size] or [batch_size, 1] - Positive sample scores
        neg_score: [batch_size, num_neg] - Negative sample scores

    Returns:
        loss: scalar
    """
    if pos_score.dim() == 1:
        pos_score = pos_score.unsqueeze(1)  # [batch_size, 1]

    # BPR loss: -log(sigmoid(pos - neg))
    loss = -F.logsigmoid(pos_score - neg_score).mean()

    return loss


def multiview_contrastive_loss(
    emb_view1: torch.Tensor,
    emb_view2: torch.Tensor,
    temperature: float = 0.1,
    batch_wise: bool = True
) -> torch.Tensor:
    """
    Multi-view contrastive learning loss

    Goal: Make User representations learned from two views similar but complementary

    Args:
        emb_view1: [batch_size or num_users, dim] - View 1 embedding (CF)
        emb_view2: [batch_size or num_users, dim] - View 2 embedding (KG)
        temperature: Temperature coefficient
        batch_wise: Whether to contrast only within the batch (recommended, faster)

    Returns:
        loss: scalar
    """
    # L2 normalization
    emb_view1 = F.normalize(emb_view1, dim=-1)
    emb_view2 = F.normalize(emb_view2, dim=-1)

    batch_size = emb_view1.size(0)

    # Positive samples: Two views of the same user
    pos_sim = (emb_view1 * emb_view2).sum(dim=-1) / temperature  # [batch_size]

    if batch_wise:
        # Negative samples: Cross-view similarity of other users in the batch
        neg_sim = emb_view1 @ emb_view2.T / temperature  # [batch_size, batch_size]

        # InfoNCE formulation
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch_size, 1+batch_size]
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)
    else:
        # Global contrast (more accurate but slower)
        # Positive sample scores
        pos_logits = pos_sim

        # Negative samples: All other users
        neg_sim_all = emb_view1 @ emb_view2.T / temperature  # [batch_size, batch_size]

        # Build logits matrix
        logits = neg_sim_all

        # Diagonal is positive samples
        labels = torch.arange(batch_size, dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)

    return loss


def entity_item_alignment_loss(
    entity_emb: torch.Tensor,
    item_emb: torch.Tensor,
    edge_index: torch.Tensor,
    num_neg: int = 5
) -> torch.Tensor:
    """
    Entity-Item Alignment Loss

    Make Entity and the Item it describes close in the embedding space

    Args:
        entity_emb: [num_entities, dim] - Entity embeddings
        item_emb: [num_items, dim] - Item embeddings
        edge_index: [2, num_edges] - (entity_describes_item edges)
            edge_index[0]: entity IDs
            edge_index[1]: item IDs
        num_neg: Number of negative samples

    Returns:
        loss: scalar
    """
    # Handle empty edge_index (e.g., when Item KG is empty in ablation)
    if edge_index.size(1) == 0:
        return torch.tensor(0.0, device=entity_emb.device, requires_grad=True)

    entity_ids = edge_index[0]  # [num_edges]
    item_ids = edge_index[1]    # [num_edges]

    num_edges = entity_ids.size(0)
    num_items = item_emb.size(0)

    # Positive pairs
    pos_entity = entity_emb[entity_ids]  # [num_edges, dim]
    pos_item = item_emb[item_ids]        # [num_edges, dim]
    pos_score = (pos_entity * pos_item).sum(dim=-1)  # [num_edges]

    # Negative sampling (random sampling)
    neg_items = torch.randint(
        0, num_items,
        (num_edges, num_neg),
        device=item_emb.device
    )
    neg_item_emb = item_emb[neg_items]  # [num_edges, num_neg, dim]

    neg_score = torch.bmm(
        neg_item_emb,
        pos_entity.unsqueeze(-1)
    ).squeeze(-1)  # [num_edges, num_neg]

    # BPR loss
    loss = bpr_loss(pos_score, neg_score)

    return loss


def mask_regularization(
    mask: torch.Tensor,
    lambda_sparse: float = 1.0,
    lambda_entropy: float = 0.1
) -> torch.Tensor:
    """
    Mask Regularization Loss

    Goal:
    1. Sparsity: Keep most Entities (mask≈1)
    2. Determinism: Avoid ambiguity (mask close to 0 or 1)

    Args:
        mask: [num_entities] - sigmoid output, range [0, 1]
        lambda_sparse: Sparsity regularization weight
        lambda_entropy: Entropy regularization weight

    Returns:
        loss: scalar
    """
    # L1 sparsity regularization: Encourage most to be 1 (not masked)
    L_sparse = (1 - mask).sum()

    # Entropy regularization: Encourage close to 0 or 1 (minimize entropy)
    eps = 1e-8
    entropy = -(
        mask * torch.log(mask + eps) +
        (1 - mask) * torch.log(1 - mask + eps)
    ).mean()

    # Combined loss
    loss = lambda_sparse * L_sparse - lambda_entropy * entropy

    return loss


class RecommendationLoss(nn.Module):
    """Complete recommendation loss (combine all losses)"""

    def __init__(
        self,
        alpha_contrast: float = 0.1,
        beta_align: float = 0.05,
        gamma_mask: float = 0.01,
        temperature_rec: float = 0.2,
        temperature_contrast: float = 0.1,
        lambda_sparse: float = 1.0,
        lambda_entropy: float = 0.1,
        num_neg_align: int = 5,
        use_contrast: bool = True,
        use_align: bool = True,
        use_mask: bool = True,
    ):
        """
        Args:
            alpha_contrast: Multi-view contrastive loss weight
            beta_align: Entity-Item alignment loss weight
            gamma_mask: Mask regularization weight
            temperature_rec: InfoNCE recommendation loss temperature
            temperature_contrast: Multi-view contrastive loss temperature
            lambda_sparse: Mask sparsity regularization weight
            lambda_entropy: Mask entropy regularization weight
            num_neg_align: Alignment loss negative sample count
            use_contrast: Whether to use multi-view contrast
            use_align: Whether to use Entity-Item alignment
            use_mask: Whether to use Mask regularization
        """
        super().__init__()

        self.alpha = alpha_contrast
        self.beta = beta_align
        self.gamma = gamma_mask

        self.temp_rec = temperature_rec
        self.temp_contrast = temperature_contrast

        self.lambda_sparse = lambda_sparse
        self.lambda_entropy = lambda_entropy

        self.num_neg_align = num_neg_align

        self.use_contrast = use_contrast
        self.use_align = use_align
        self.use_mask = use_mask

    def forward(
        self,
        user_emb_fused: torch.Tensor,
        pos_item_emb: torch.Tensor,
        neg_item_emb: torch.Tensor,
        user_emb_cf: Optional[torch.Tensor] = None,
        user_emb_kg: Optional[torch.Tensor] = None,
        entity_emb: Optional[torch.Tensor] = None,
        item_emb_kg: Optional[torch.Tensor] = None,
        entity_item_edges: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute complete loss

        Args:
            user_emb_fused: [batch_size, dim] - Fused user embedding
            pos_item_emb: [batch_size, dim] - Positive sample item
            neg_item_emb: [batch_size, num_neg, dim] - Negative sample items
            user_emb_cf: [num_users or batch_size, dim] - CF view user embedding
            user_emb_kg: [num_users or batch_size, dim] - KG view user embedding
            entity_emb: [num_entities, dim] - Entity embeddings
            item_emb_kg: [num_items, dim] - KG view Item embeddings
            entity_item_edges: [2, num_edges] - Entity-Item edges
            mask: [num_entities] - Learned mask

        Returns:
            loss_dict: {
                'loss': total_loss,
                'L_rec': ...,
                'L_contrast': ...,
                'L_align': ...,
                'L_mask': ...
            }
        """
        loss_dict = {}

        # === 1. Main Loss: InfoNCE Recommendation ===
        L_rec = info_nce_loss(
            user_emb_fused,
            pos_item_emb,
            neg_item_emb,
            temperature=self.temp_rec
        )
        loss_dict['L_rec'] = L_rec

        # === 2. Multi-view Contrastive Loss ===
        if self.use_contrast and user_emb_cf is not None and user_emb_kg is not None:
            L_contrast = multiview_contrastive_loss(
                user_emb_cf,
                user_emb_kg,
                temperature=self.temp_contrast,
                batch_wise=True  # Contrast within batch (faster)
            )
            loss_dict['L_contrast'] = L_contrast
        else:
            L_contrast = torch.tensor(0.0, device=L_rec.device)
            loss_dict['L_contrast'] = L_contrast

        # === 3. Entity-Item Alignment Loss ===
        if self.use_align and entity_emb is not None and item_emb_kg is not None and entity_item_edges is not None:
            L_align = entity_item_alignment_loss(
                entity_emb,
                item_emb_kg,
                entity_item_edges,
                num_neg=self.num_neg_align
            )
            loss_dict['L_align'] = L_align
        else:
            L_align = torch.tensor(0.0, device=L_rec.device)
            loss_dict['L_align'] = L_align

        # === 4. Mask Regularization ===
        if self.use_mask and mask is not None:
            L_mask = mask_regularization(
                mask,
                lambda_sparse=self.lambda_sparse,
                lambda_entropy=self.lambda_entropy
            )
            loss_dict['L_mask'] = L_mask
        else:
            L_mask = torch.tensor(0.0, device=L_rec.device)
            loss_dict['L_mask'] = L_mask

        # === Total Loss ===
        total_loss = (
            L_rec
            + self.alpha * L_contrast
            + self.beta * L_align
            + self.gamma * L_mask
        )
        loss_dict['loss'] = total_loss

        return loss_dict


if __name__ == '__main__':
    # Test loss functions
    torch.manual_seed(42)

    batch_size = 64
    dim = 128
    num_neg = 5

    # Test data
    user_emb = torch.randn(batch_size, dim)
    pos_item_emb = torch.randn(batch_size, dim)
    neg_item_emb = torch.randn(batch_size, num_neg, dim)

    # 1. InfoNCE
    loss_rec = info_nce_loss(user_emb, pos_item_emb, neg_item_emb)
    print(f"InfoNCE loss: {loss_rec.item():.4f}")

    # 2. Multi-view contrast
    user_cf = torch.randn(batch_size, dim)
    user_kg = torch.randn(batch_size, dim)
    loss_contrast = multiview_contrastive_loss(user_cf, user_kg)
    print(f"Contrastive loss: {loss_contrast.item():.4f}")

    # 3. Mask regularization
    mask = torch.sigmoid(torch.randn(400))
    loss_mask = mask_regularization(mask)
    print(f"Mask regularization: {loss_mask.item():.4f}")

    # 4. Complete loss
    criterion = RecommendationLoss()
    loss_dict = criterion(
        user_emb_fused=user_emb,
        pos_item_emb=pos_item_emb,
        neg_item_emb=neg_item_emb,
        user_emb_cf=user_cf,
        user_emb_kg=user_kg,
        mask=mask
    )

    print(f"\n✓ Total loss: {loss_dict['loss'].item():.4f}")
    for key, value in loss_dict.items():
        if key != 'loss':
            print(f"  {key}: {value.item():.4f}")
