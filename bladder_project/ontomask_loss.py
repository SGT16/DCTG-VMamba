from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F

from .ontomask_model import OntoMaskVariant


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def mask_scores(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    prediction = torch.sigmoid(logits) >= 0.5
    truth = target >= 0.5
    dims = tuple(range(1, truth.ndim))
    intersection = (prediction & truth).sum(dim=dims).float()
    pred_sum = prediction.sum(dim=dims).float()
    true_sum = truth.sum(dim=dims).float()
    union = (prediction | truth).sum(dim=dims).float()
    dice = ((2.0 * intersection + 1.0) / (pred_sum + true_sum + 1.0)).mean()
    iou = ((intersection + 1.0) / (union + 1.0)).mean()
    return {"mask_dice": dice, "mask_iou": iou}


def jensen_shannon(probability_a: torch.Tensor, probability_b: torch.Tensor) -> torch.Tensor:
    eps = 1e-7
    probability_a = probability_a.clamp_min(eps)
    probability_b = probability_b.clamp_min(eps)
    middle = 0.5 * (probability_a + probability_b)
    kl_a = (probability_a * (probability_a.log() - middle.log())).sum(dim=-1)
    kl_b = (probability_b * (probability_b.log() - middle.log())).sum(dim=-1)
    return (0.5 * (kl_a + kl_b)).mean()


def cross_dataset_supervised_contrastive(
    features: torch.Tensor,
    ontology_labels: torch.Tensor,
    dataset_ids: torch.Tensor,
    temperature: float = 0.12,
    positive_scope: str = "cross_dataset_only",
) -> torch.Tensor:
    if features.shape[0] < 2:
        return _zero(features)
    similarity = features @ features.transpose(0, 1) / float(temperature)
    eye = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    same_ontology = ontology_labels[:, None] == ontology_labels[None, :]
    if positive_scope == "all_same_ontology":
        
        
        
        
        
        positives = same_ontology & ~eye
    elif positive_scope == "cross_dataset_only":
        positives = same_ontology & (dataset_ids[:, None] != dataset_ids[None, :]) & ~eye
    else:
        raise ValueError(
            f"Unknown contrast_positive_scope={positive_scope!r}; expected "
            "'cross_dataset_only' or 'all_same_ontology'."
        )
    valid_anchors = positives.any(dim=1)
    if not bool(valid_anchors.any()):
        return _zero(features)
    similarity = similarity.masked_fill(eye, float("-inf"))
    log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    positive_count = positives.sum(dim=1).clamp_min(1)
    mean_positive_log_probability = (log_probability.masked_fill(~positives, 0.0).sum(dim=1) / positive_count)
    return -mean_positive_log_probability[valid_anchors].mean()


def within_dataset_subtype_contrastive(
    features: torch.Tensor,
    fine_labels: torch.Tensor,
    dataset_ids: torch.Tensor,
    temperature: float = 0.12,
) -> torch.Tensor:
    """Keep fine classes distinct inside each dataset in a separate embedding space.

    Ontology alignment is learned by ``cross_dataset_supervised_contrastive``.
    This complementary loss uses only within-dataset pairs, so HGC and LGC can
    remain distinct even though both map to the shared neoplastic ontology node.
    """

    if features.shape[0] < 2:
        return _zero(features)
    eye = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    same_dataset = dataset_ids[:, None] == dataset_ids[None, :]
    candidates = same_dataset & ~eye
    positives = candidates & (fine_labels[:, None] == fine_labels[None, :])
    valid_anchors = positives.any(dim=1)
    if not bool(valid_anchors.any()):
        return _zero(features)

    similarity = features @ features.transpose(0, 1) / float(temperature)
    similarity = similarity.masked_fill(~candidates, float("-inf"))
    log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    positive_count = positives.sum(dim=1).clamp_min(1)
    mean_positive_log_probability = (
        log_probability.masked_fill(~positives, 0.0).sum(dim=1) / positive_count
    )
    return -mean_positive_log_probability[valid_anchors].mean()


def compute_ontomask_loss(
    output: Mapping[str, Any],
    batch: Mapping[str, torch.Tensor],
    variant: OntoMaskVariant,
    ontology_matrices: Mapping[str, torch.Tensor],
    class_weights: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
) -> Dict[str, torch.Tensor]:
    reference = output["fused"]
    fine_losses = []
    consistency_losses = []
    for dataset_name, (indices, logits) in output["fine_logits"].items():
        targets = batch["fine_label"][indices]
        fine_losses.append(
            F.cross_entropy(logits, targets, weight=class_weights.get(dataset_name))
        )
        if variant.use_consistency:
            fine_probability = torch.softmax(logits, dim=-1)
            fine_to_ontology = ontology_matrices[dataset_name].to(logits.device, logits.dtype)
            aggregated = fine_probability @ fine_to_ontology
            shared = torch.softmax(output["ontology_logits"][indices], dim=-1)
            consistency_losses.append(jensen_shannon(aggregated, shared))
    fine_loss = torch.stack(fine_losses).mean() if fine_losses else _zero(reference)

    ontology_loss = _zero(reference)
    if variant.use_ontology:
        ontology_loss = F.cross_entropy(output["ontology_logits"], batch["ontology_label"])

    consistency_loss = (
        torch.stack(consistency_losses).mean() if consistency_losses else _zero(reference)
    )

    mask_loss = _zero(reference)
    separation_loss = _zero(reference)
    valid = batch["mask_valid"].bool()
    if variant.use_mask and bool(valid.any()):
        target = F.interpolate(
            batch["mask"][valid],
            size=output["mask_logits"].shape[-2:],
            mode="nearest",
        )
        predicted = output["mask_logits"][valid]
        mask_loss = F.binary_cross_entropy_with_logits(predicted, target) + soft_dice_loss(predicted, target)
        if variant.use_mask_fusion:
            similarity = F.cosine_similarity(output["roi_token"][valid], output["context_token"][valid], dim=-1)
            separation_loss = F.relu(
                similarity - float(weights.get("roi_context_margin", 0.2))
            ).mean()

    contrast_loss = _zero(reference)
    if variant.use_cross_dataset_contrast:
        contrast_loss = cross_dataset_supervised_contrastive(
            output["contrast_features"],
            batch["ontology_label"],
            batch["dataset_id"],
            temperature=float(weights.get("contrast_temperature", 0.12)),
            positive_scope=str(
                weights.get("contrast_positive_scope", "cross_dataset_only")
            ),
        )

    subtype_loss = _zero(reference)
    if variant.use_subtype_contrast:
        subtype_loss = within_dataset_subtype_contrastive(
            output["subtype_features"],
            batch["fine_label"],
            batch["dataset_id"],
            temperature=float(weights.get("subtype_temperature", 0.12)),
        )

    total = (
        float(weights.get("fine", 1.0)) * fine_loss
        + float(weights.get("ontology", 0.35)) * ontology_loss
        + float(weights.get("consistency", 0.20)) * consistency_loss
        + float(weights.get("mask", 0.60)) * mask_loss
        + float(weights.get("separation", 0.05)) * separation_loss
        + float(weights.get("contrast", 0.08)) * contrast_loss
        + float(weights.get("subtype", 0.05)) * subtype_loss
    )
    return {
        "total": total,
        "fine": fine_loss,
        "ontology": ontology_loss,
        "consistency": consistency_loss,
        "mask": mask_loss,
        "separation": separation_loss,
        "contrast": contrast_loss,
        "subtype": subtype_loss,
    }
