from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
from torch import nn

from .ontomask_model import (
    TimmFeatureBackbone,
    VMambaHookBackbone,
    count_trainable_params,
    create_ontomask_model,
)
from .model_factory import create_model


class ClassifierEmbeddingBackbone(nn.Module):
    """Use a standard pretrained classifier as a shared embedding encoder.

    The original classification head is replaced with an `embedding_dim`
    projection.  This gives CNN, Transformer, MetaFormer and HF Mamba models
    the same dataset-specific fine-head interface without requiring every
    architecture to expose spatial `features_only` maps.
    """

    def __init__(self, source_cfg: Mapping[str, Any], embedding_dim: int):
        super().__init__()
        cfg = dict(source_cfg)
        self.input_adapter_size = int(cfg.pop("input_adapter_size", 0) or 0)
        source_type = cfg.pop("source_type", cfg.pop("type_original", "timm"))
        cfg["type"] = source_type
        cfg.setdefault("key", str(cfg.get("timm_name", cfg.get("hf_model_id", "encoder"))))
        self.encoder = create_model(cfg, num_classes=int(embedding_dim))
        self.out_channels = [int(embedding_dim)]

    def forward(self, image: torch.Tensor) -> Sequence[torch.Tensor]:
        if self.input_adapter_size and image.shape[-2:] != (self.input_adapter_size, self.input_adapter_size):
            image = torch.nn.functional.interpolate(
                image,
                size=(self.input_adapter_size, self.input_adapter_size),
                mode="bilinear",
                align_corners=False,
            )
        embedding = self.encoder(image)
        if isinstance(embedding, dict):
            embedding = embedding.get("logits", next(iter(embedding.values())))
        if hasattr(embedding, "logits"):
            embedding = embedding.logits
        if isinstance(embedding, (tuple, list)):
            embedding = embedding[0]
        if embedding.ndim > 2:
            embedding = embedding.mean(dim=tuple(range(2, embedding.ndim)))
        if embedding.ndim != 2:
            raise RuntimeError(f"Expected a 2-D embedding, received {tuple(embedding.shape)}")
        return [embedding.unsqueeze(-1).unsqueeze(-1)]


class JointBaselineNet(nn.Module):
    """A true pooled-backbone control with one fine head per dataset."""

    is_dctg = False

    def __init__(
        self,
        backbone: nn.Module,
        feature_channels: Sequence[int],
        dataset_classes: Mapping[str, Sequence[str]],
        dataset_id_by_name: Mapping[str, int] | None = None,
        single_dataset_name: str | None = None,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.backbone = backbone
        self.dataset_names = list(dataset_classes)
        self.single_dataset_name = single_dataset_name
        if dataset_id_by_name:
            self.dataset_name_by_id = {
                int(dataset_id_by_name[name]): name for name in self.dataset_names
            }
        else:
            self.dataset_name_by_id = {
                index: name for index, name in enumerate(self.dataset_names)
            }
        channels = int(feature_channels[-1])
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(channels),
                    nn.Dropout(dropout),
                    nn.Linear(channels, len(classes)),
                )
                for name, classes in dataset_classes.items()
            }
        )

    def forward(self, image: torch.Tensor, dataset_ids: torch.Tensor) -> Dict[str, Any]:
        features = self.backbone(image)
        pooled = features[-1].mean(dim=(-2, -1))
        fine_logits: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        if self.single_dataset_name is not None:
            indices = torch.arange(image.shape[0], device=image.device)
            fine_logits[self.single_dataset_name] = (
                indices,
                self.heads[self.single_dataset_name](pooled),
            )
            return {"fine_logits": fine_logits, "fused": pooled}
        for dataset_id, dataset_name in self.dataset_name_by_id.items():
            indices = torch.nonzero(dataset_ids == dataset_id, as_tuple=False).flatten()
            if indices.numel():
                fine_logits[dataset_name] = (indices, self.heads[dataset_name](pooled[indices]))
        return {"fine_logits": fine_logits, "fused": pooled}


def _create_backbone(backbone_cfg: Mapping[str, Any]) -> Tuple[nn.Module, Sequence[int]]:
    cfg = dict(backbone_cfg)
    kind = str(cfg.get("type", "timm")).lower()
    if kind == "timm":
        backbone = TimmFeatureBackbone(
            model_name=str(cfg["timm_name"]),
            pretrained=bool(cfg.get("pretrained", True)),
            out_indices=cfg.get("out_indices", [1, 2, 3]),
        )
        return backbone, backbone.out_channels
    if kind == "classifier_embedding":
        embedding_dim = int(cfg.pop("embedding_dim", 192))
        cfg.pop("type", None)
        backbone = ClassifierEmbeddingBackbone(cfg, embedding_dim=embedding_dim)
        return backbone, backbone.out_channels
    if kind == "vmamba_hook":
        channels = [int(value) for value in cfg.pop("feature_channels", [96, 192, 384, 768])]
        cfg.pop("type", None)
        checkpoint = str(cfg.get("checkpoint_path", ""))
        if checkpoint and not __import__("pathlib").Path(checkpoint).exists():
            raise FileNotFoundError(f"Required pretrained VMamba checkpoint is missing: {checkpoint}")
        return VMambaHookBackbone(cfg, out_channels=channels), channels
    raise ValueError(f"Unsupported backbone type: {kind}")


def create_study_model(
    model_cfg: Mapping[str, Any],
    dataset_classes: Mapping[str, Sequence[str]],
    method_cfg: Mapping[str, Any],
) -> nn.Module:
    architecture = str(model_cfg.get("architecture", "joint_baseline"))
    if architecture in {"dctg", "ontomask_ablation"}:
        variant_name = str(
            model_cfg.get(
                "variant",
                "full_dataset_gate" if architecture == "dctg" else "full",
            )
        )
        model = create_ontomask_model(
            model_cfg,
            dataset_classes=dataset_classes,
            variant_name=variant_name,
            method_cfg=method_cfg,
        )
        
        model.is_dctg = True
        return model
    if architecture in {"joint_baseline", "single_baseline"}:
        backbone, channels = _create_backbone(model_cfg["backbone"])
        return JointBaselineNet(
            backbone,
            channels,
            dataset_classes,
            dataset_id_by_name=model_cfg.get("dataset_id_by_name"),
            single_dataset_name=(
                str(model_cfg["single_dataset"])
                if architecture == "single_baseline"
                else None
            ),
            dropout=float(method_cfg.get("dropout", 0.15)),
        )
    raise ValueError(f"Unknown architecture {architecture!r}")


__all__ = [
    "ClassifierEmbeddingBackbone",
    "JointBaselineNet",
    "create_study_model",
    "count_trainable_params",
]
