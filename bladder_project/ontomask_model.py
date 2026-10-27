from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import timm
import torch
import torch.nn.functional as F
from torch import nn

from .model_factory import ExternalFactoryWrapper
from .v3_data import ONTOLOGY_CLASSES


def _as_spatial(tensor: torch.Tensor, expected_channels: int) -> torch.Tensor:
    if tensor.ndim != 4:
        raise RuntimeError(f"Expected a 4-D feature map, got {tuple(tensor.shape)}")
    if tensor.shape[1] == expected_channels:
        return tensor.contiguous()
    if tensor.shape[-1] == expected_channels:
        return tensor.permute(0, 3, 1, 2).contiguous()
    raise RuntimeError(
        f"Could not identify channel axis for feature {tuple(tensor.shape)}; "
        f"expected {expected_channels} channels."
    )


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    if isinstance(value, Mapping):
        for item in value.values():
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    raise TypeError("No tensor found in module output")


class TimmFeatureBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool, out_indices: Sequence[int]):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=bool(pretrained),
            features_only=True,
            out_indices=tuple(int(i) for i in out_indices),
        )
        self.out_channels = list(self.model.feature_info.channels())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = self.model(x)
        return [_as_spatial(tensor, channels) for tensor, channels in zip(outputs, self.out_channels)]


class VMambaHookBackbone(nn.Module):
    """Expose VMamba stage maps while retaining the verified official checkpoint loader.

    The external VMamba revisions used by this project do not expose one stable
    ``features_only`` API. Hooks are attached to each stage's final block (before
    the stage downsampling operation), so the expected 96/192/384/768 channel
    pyramid is exposed without changing the checkpoint-backed forward path.
    """

    def __init__(self, model_cfg: Mapping[str, Any], out_channels: Sequence[int]):
        super().__init__()
        wrapper = ExternalFactoryWrapper(dict(model_cfg), num_classes=1000)
        self.model = wrapper.model
        self.all_channels = [int(v) for v in out_channels]
        stages = None
        for attr in ("layers", "stages"):
            candidate = getattr(self.model, attr, None)
            if candidate is not None and len(candidate) >= len(self.all_channels):
                stages = list(candidate)
                break
        if stages is None:
            raise RuntimeError(
                "VMamba stage modules were not found. Expected model.layers or model.stages. "
                "Inspect the checked-out VMamba revision before changing this adapter."
            )
        selected_stages = stages[-len(self.all_channels):]
        hook_targets = [self._pre_downsample_target(stage) for stage in selected_stages]
        self._captured: List[torch.Tensor] = []
        self._shape_reported = False
        self._hooks = [target.register_forward_hook(self._capture) for target in hook_targets]

    @staticmethod
    def _pre_downsample_target(stage: nn.Module) -> nn.Module:
        """Return the last stage block, before VSSLayer applies downsampling.

        Official VMamba revisions have used both ``ModuleList`` and
        ``Sequential`` containers for ``stage.blocks``. Hooking the final block
        works for both. Falling back to the whole stage preserves compatibility
        with revisions that do not expose ``blocks``.
        """

        blocks = getattr(stage, "blocks", None)
        if blocks is not None:
            try:
                if len(blocks) > 0:
                    return blocks[-1]
            except (TypeError, AttributeError):
                pass
        return stage

    def _capture(self, _module, _inputs, output) -> None:
        self._captured.append(_first_tensor(output))

    @property
    def out_channels(self) -> List[int]:
        return self.all_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        self._captured = []
        if hasattr(self.model, "forward_features"):
            _ = self.model.forward_features(x)
        else:
            _ = self.model(x)
        if len(self._captured) != len(self.all_channels):
            raise RuntimeError(
                f"VMamba captured {len(self._captured)} stage maps; expected {len(self.all_channels)}."
            )
        outputs = [
            _as_spatial(tensor, channels)
            for tensor, channels in zip(self._captured, self.all_channels)
        ]
        if not self._shape_reported:
            shapes = ", ".join(str(tuple(tensor.shape)) for tensor in outputs)
            print(f"[OntoMask-VMamba] pre-downsample feature maps: {shapes}")
            self._shape_reported = True
        return outputs


class TopDownMaskPyramid(nn.Module):
    def __init__(self, in_channels: Sequence[int], embed_dim: int):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(int(c), embed_dim, kernel_size=1) for c in in_channels])
        self.smooth = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(16, embed_dim),
                    nn.GELU(),
                )
                for _ in in_channels
            ]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        projected = [layer(feature) for layer, feature in zip(self.lateral, features)]
        current = projected[-1]
        outputs = [None] * len(projected)
        outputs[-1] = self.smooth[-1](current)
        for index in range(len(projected) - 2, -1, -1):
            current = F.interpolate(current, size=projected[index].shape[-2:], mode="bilinear", align_corners=False)
            current = current + projected[index]
            outputs[index] = self.smooth[index](current)
        return outputs[0]


@dataclass(frozen=True)
class OntoMaskVariant:
    name: str
    use_mask: bool
    use_ontology: bool
    use_consistency: bool
    use_cross_dataset_contrast: bool
    use_subtype_contrast: bool = False
    use_mask_confidence: bool = False
    use_mask_fusion: bool = True
    use_dataset_conditioned_gate: bool = False
    global_only_classification: bool = False
    use_gated_ontology_residual: bool = False


VARIANTS: Dict[str, OntoMaskVariant] = {
    "baseline": OntoMaskVariant("baseline", False, False, False, False),
    "mask_only": OntoMaskVariant("mask_only", True, False, False, False),
    "ontology_only": OntoMaskVariant("ontology_only", False, True, True, True),
    "full": OntoMaskVariant("full", True, True, True, True),
    
    "full_subtype": OntoMaskVariant(
        "full_subtype", True, True, True, True, use_subtype_contrast=True
    ),
    "full_mask_confidence": OntoMaskVariant(
        "full_mask_confidence", True, True, True, True, use_mask_confidence=True
    ),
    
    
    
    "ontology_auxmask": OntoMaskVariant(
        "ontology_auxmask",
        True,
        True,
        True,
        True,
        use_mask_fusion=False,
    ),
    
    
    
    "full_dataset_gate": OntoMaskVariant(
        "full_dataset_gate",
        True,
        True,
        True,
        True,
        use_dataset_conditioned_gate=True,
    ),
    
    
    
    "dctg_global_only": OntoMaskVariant(
        "dctg_global_only",
        True,
        True,
        True,
        True,
        use_mask_fusion=True,
        use_dataset_conditioned_gate=True,
        global_only_classification=True,
    ),
    
    
    
    
    
    "full_dataset_gate_rebalanced": OntoMaskVariant(
        "full_dataset_gate_rebalanced",
        True,
        True,
        True,
        True,
        use_dataset_conditioned_gate=True,
        use_gated_ontology_residual=True,
    ),
    
    
    
    
    
    
    "full_dataset_gate_secondary_coarse": OntoMaskVariant(
        "full_dataset_gate_secondary_coarse",
        True,
        True,
        False,
        True,
        use_dataset_conditioned_gate=True,
        use_gated_ontology_residual=True,
    ),
    "dctg_revised_global_only": OntoMaskVariant(
        "dctg_revised_global_only",
        True,
        True,
        True,
        True,
        use_dataset_conditioned_gate=True,
        global_only_classification=True,
        use_gated_ontology_residual=True,
    ),
    
    
    
    
    "full_rebalanced_no_dataset_gate": OntoMaskVariant(
        "full_rebalanced_no_dataset_gate",
        True,
        True,
        True,
        True,
        use_dataset_conditioned_gate=False,
        use_gated_ontology_residual=True,
    ),
    
    
    
    "full_dataset_gate_no_taxonomy_aux": OntoMaskVariant(
        "full_dataset_gate_no_taxonomy_aux",
        True,
        False,
        False,
        True,
        use_dataset_conditioned_gate=True,
    ),
    
    
    
    
    "full_dataset_gate_no_taxonomy": OntoMaskVariant(
        "full_dataset_gate_no_taxonomy",
        True,
        False,
        False,
        False,
        use_dataset_conditioned_gate=True,
    ),
    
    
    "full_dataset_gate_subtype": OntoMaskVariant(
        "full_dataset_gate_subtype",
        True,
        True,
        True,
        True,
        use_subtype_contrast=True,
        use_dataset_conditioned_gate=True,
    ),
    
    "full_combined": OntoMaskVariant(
        "full_combined",
        True,
        True,
        True,
        True,
        use_subtype_contrast=True,
        use_mask_confidence=True,
    ),
}


class OntoMaskNet(nn.Module):
    """Mask-supervised, ontology-consistent multi-dataset classifier.

    Ground-truth masks are never passed to ``forward``. The network predicts a
    soft ROI map from shared features. Fusion variants pool ROI/context/global
    tokens; auxiliary-mask variants use the mask loss only to regularize the
    shared representation before dataset-specific classification.
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_channels: Sequence[int],
        dataset_classes: Mapping[str, Sequence[str]],
        variant: OntoMaskVariant,
        embed_dim: int = 192,
        dropout: float = 0.15,
    ):
        super().__init__()
        if len(backbone_channels) < 3:
            raise ValueError("OntoMaskNet requires at least three spatial backbone stages")
        self.backbone = backbone
        self.variant = variant
        self.dataset_names = list(dataset_classes)
        self.dataset_name_by_id = {i: name for i, name in enumerate(self.dataset_names)}
        selected_channels = list(backbone_channels)[-3:]
        self.pyramid = TopDownMaskPyramid(selected_channels, embed_dim=embed_dim)
        self.mask_decoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=1),
        )
        self.ontology_head = (
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, len(ONTOLOGY_CLASSES)),
            )
            if variant.use_ontology
            else None
        )
        self.ontology_embeddings = (
            nn.Parameter(torch.empty(len(ONTOLOGY_CLASSES), embed_dim))
            if variant.use_ontology
            else None
        )
        if self.ontology_embeddings is not None:
            nn.init.trunc_normal_(self.ontology_embeddings, std=0.02)
        self.ontology_residual_gate = (
            nn.Sequential(
                nn.LayerNorm(embed_dim * 2),
                nn.Linear(embed_dim * 2, embed_dim),
            )
            if variant.use_gated_ontology_residual
            else None
        )
        if self.ontology_residual_gate is not None:
            
            
            
            nn.init.zeros_(self.ontology_residual_gate[-1].weight)
            nn.init.zeros_(self.ontology_residual_gate[-1].bias)
        self.dataset_embeddings = (
            nn.Embedding(len(self.dataset_names), embed_dim)
            if variant.use_dataset_conditioned_gate
            else None
        )
        if self.dataset_embeddings is not None:
            nn.init.trunc_normal_(self.dataset_embeddings.weight, std=0.02)
        token_gate_inputs = 5 if variant.use_dataset_conditioned_gate else 4
        self.token_gate = nn.Sequential(
            nn.Linear(embed_dim * token_gate_inputs, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 3),
        )
        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.fine_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim, len(classes)),
                )
                for name, classes in dataset_classes.items()
            }
        )
        self.contrast_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 96),
        )
        self.subtype_projection = (
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 96),
            )
            if variant.use_subtype_contrast
            else None
        )
        self.mask_confidence_head = (
            nn.Sequential(
                nn.LayerNorm(embed_dim * 4),
                nn.Linear(embed_dim * 4, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1),
            )
            if variant.use_mask_confidence
            else None
        )

    @staticmethod
    def _masked_pool(feature: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        numerator = (feature * weights).sum(dim=(-2, -1))
        denominator = weights.sum(dim=(-2, -1)).clamp_min(1e-5)
        return numerator / denominator

    @staticmethod
    def _mask_certainty(roi_weights: torch.Tensor) -> torch.Tensor:
        """Return finite entropy-based mask certainty, including under FP16 AMP.

        ``1 - 1e-6`` rounds back to ``1`` in float16.  Computing binary entropy
        directly from such a tensor can therefore evaluate ``0 * log(0)`` and
        inject NaNs into the confidence gate.  Entropy is deliberately computed
        in float32 with representable bounds; a non-finite value falls back to
        zero certainty, which makes the residual gate use the global token.
        """
        probability = roi_weights.float().clamp(1e-4, 1.0 - 1e-4)
        complement = 1.0 - probability
        binary_entropy = -(
            probability * probability.log()
            + complement * complement.log()
        )
        certainty = 1.0 - binary_entropy.mean(dim=(-3, -2, -1)) / 0.6931471805599453
        return torch.nan_to_num(
            certainty,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

    def forward(self, image: torch.Tensor, dataset_ids: torch.Tensor) -> Dict[str, Any]:
        features = self.backbone(image)
        feature_map = self.pyramid(features[-3:])
        global_token = feature_map.mean(dim=(-2, -1))
        mask_logits = self.mask_decoder(feature_map)

        if self.variant.use_mask:
            roi_weights = torch.sigmoid(mask_logits)
            if self.variant.use_mask_fusion:
                roi_token = self._masked_pool(feature_map, roi_weights)
                context_token = self._masked_pool(feature_map, 1.0 - roi_weights)
            else:
                
                
                
                roi_token = global_token
                context_token = global_token
        else:
            roi_weights = torch.full_like(mask_logits, 0.5)
            roi_token = global_token
            context_token = global_token

        if self.ontology_head is not None:
            ontology_logits = self.ontology_head(global_token)
            ontology_probs = torch.softmax(ontology_logits, dim=-1)
            ontology_context = ontology_probs @ self.ontology_embeddings
        else:
            ontology_logits = torch.zeros(
                image.shape[0],
                len(ONTOLOGY_CLASSES),
                device=image.device,
                dtype=global_token.dtype,
            )
            ontology_context = torch.zeros_like(global_token)

        if self.ontology_residual_gate is not None:
            ontology_gate = torch.tanh(
                self.ontology_residual_gate(
                    torch.cat([global_token, ontology_context], dim=-1)
                )
            )
            gated_ontology_context = ontology_gate * ontology_context
        else:
            ontology_gate = torch.ones_like(global_token)
            gated_ontology_context = ontology_context

        if self.variant.global_only_classification:
            gate_weights = torch.zeros(
                image.shape[0], 3, device=image.device, dtype=image.dtype
            )
            gate_weights[:, 2] = 1.0
            mask_confidence = torch.zeros(
                image.shape[0], 1, device=image.device, dtype=global_token.dtype
            )
            
            
            
            
            fused = global_token
        elif self.variant.use_mask and self.variant.use_mask_fusion:
            fusion_context = torch.cat(
                [roi_token, context_token, global_token, ontology_context], dim=-1
            )
            token_gate_input = fusion_context
            if self.variant.use_dataset_conditioned_gate:
                dataset_context = self.dataset_embeddings(dataset_ids).to(
                    fusion_context.dtype
                )
                token_gate_input = torch.cat([fusion_context, dataset_context], dim=-1)
            gate_weights = torch.softmax(self.token_gate(token_gate_input), dim=-1)
            masked_fused = (
                gate_weights[:, 0:1] * roi_token
                + gate_weights[:, 1:2] * context_token
                + gate_weights[:, 2:3] * global_token
            )
            if self.variant.use_mask_confidence:
                mask_certainty = self._mask_certainty(roi_weights)
                learned_confidence = torch.sigmoid(
                    self.mask_confidence_head(fusion_context)
                ).squeeze(-1)
                
                
                mask_confidence = torch.nan_to_num(
                    mask_certainty * learned_confidence.float(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0).to(global_token.dtype).unsqueeze(-1)
                
                fused = global_token + mask_confidence * (masked_fused - global_token)
                fused = fused + gated_ontology_context
            else:
                mask_confidence = torch.ones(
                    image.shape[0], 1, device=image.device, dtype=global_token.dtype
                )
                fused = masked_fused + gated_ontology_context
        else:
            gate_weights = torch.zeros(image.shape[0], 3, device=image.device, dtype=image.dtype)
            gate_weights[:, 2] = 1.0
            mask_confidence = torch.zeros(
                image.shape[0], 1, device=image.device, dtype=global_token.dtype
            )
            fused = global_token + gated_ontology_context
        fused = self.fusion_norm(fused)

        contrast_features = F.normalize(self.contrast_projection(fused), dim=-1)
        subtype_features = (
            F.normalize(self.subtype_projection(fused), dim=-1)
            if self.subtype_projection is not None
            else contrast_features
        )

        fine_logits: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for dataset_id, dataset_name in self.dataset_name_by_id.items():
            indices = torch.nonzero(dataset_ids == dataset_id, as_tuple=False).flatten()
            if indices.numel() > 0:
                fine_logits[dataset_name] = (indices, self.fine_heads[dataset_name](fused[indices]))

        return {
            "fine_logits": fine_logits,
            "ontology_logits": ontology_logits,
            "mask_logits": mask_logits,
            "mask_probs": roi_weights,
            "roi_token": roi_token,
            "context_token": context_token,
            "fused": fused,
            "contrast_features": contrast_features,
            "subtype_features": subtype_features,
            "gate_weights": gate_weights,
            "mask_confidence": mask_confidence,
            "ontology_residual_gate": ontology_gate,
        }


def create_ontomask_model(
    model_cfg: Mapping[str, Any],
    dataset_classes: Mapping[str, Sequence[str]],
    variant_name: str,
    method_cfg: Mapping[str, Any],
) -> OntoMaskNet:
    if variant_name not in VARIANTS:
        raise ValueError(f"Unknown OntoMask variant {variant_name!r}; choose from {sorted(VARIANTS)}")
    backbone_cfg = dict(model_cfg["backbone"])
    backbone_type = str(backbone_cfg.get("type", "timm")).lower()
    if backbone_type == "timm":
        backbone = TimmFeatureBackbone(
            model_name=str(backbone_cfg["timm_name"]),
            pretrained=bool(backbone_cfg.get("pretrained", True)),
            out_indices=backbone_cfg.get("out_indices", [1, 2, 3]),
        )
        channels = backbone.out_channels
    elif backbone_type == "vmamba_hook":
        channels = [int(v) for v in backbone_cfg.pop("feature_channels", [96, 192, 384, 768])]
        backbone_cfg.pop("type", None)
        backbone = VMambaHookBackbone(backbone_cfg, out_channels=channels)
    else:
        raise ValueError(f"Unsupported OntoMask backbone type: {backbone_type}")
    return OntoMaskNet(
        backbone=backbone,
        backbone_channels=channels,
        dataset_classes=dataset_classes,
        variant=VARIANTS[variant_name],
        embed_dim=int(method_cfg.get("embed_dim", 192)),
        dropout=float(method_cfg.get("dropout", 0.15)),
    )


def count_trainable_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
