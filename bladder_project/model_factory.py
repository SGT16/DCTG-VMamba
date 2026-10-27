from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import timm
import torch
from torch import nn


class HFImageClassificationWrapper(nn.Module):
    """Wrap Hugging Face image-classification models so training.py receives logits.

    Used for MambaVision because the official HF usage relies on
    AutoModelForImageClassification(..., trust_remote_code=True). The classifier
    head is resized with ignore_mismatched_sizes=True when num_classes differs
    from ImageNet-1K.
    """

    def __init__(self, model_id: str, num_classes: int, trust_remote_code: bool = True, pretrained: bool = True):
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModelForImageClassification
        except Exception as exc:
            raise RuntimeError("transformers is required for HF image models. Install requirements_mamba_wsl.txt") from exc

        if pretrained:
            self.model = AutoModelForImageClassification.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            cfg.num_labels = num_classes
            self.model = AutoModelForImageClassification.from_config(cfg, trust_remote_code=trust_remote_code)

        
        
        
        
        cur = _classifier_out_features(self.model)
        if cur is None or cur != num_classes:
            self.model = _replace_common_classifier(self.model, num_classes)
            new = _classifier_out_features(self.model)
            print(f"[hf] classifier head resized: {cur} -> {new} (target {num_classes})")
            if new is not None and new != num_classes:
                raise RuntimeError(
                    f"Could not resize HF classifier head to {num_classes} classes for {model_id} "
                    f"(still {new}). Inspect the model's head attribute name in model_factory.py."
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):
            return out["logits"]
        return out.logits


def _last_linear_name(model: nn.Module) -> Optional[str]:
    """Return the fully-qualified name of the last nn.Linear in the module tree."""
    last = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last = name
    return last


def _classifier_out_features(model: nn.Module) -> Optional[int]:
    """Return the out_features of the model's classification head, if detectable."""
    for attr in ["classifier", "head", "fc", "score"]:
        if hasattr(model, attr):
            layer = getattr(model, attr)
            if isinstance(layer, nn.Linear):
                return layer.out_features
            if isinstance(layer, nn.Sequential):
                for m in reversed(list(layer.children())):
                    if isinstance(m, nn.Linear):
                        return m.out_features
    
    last = _last_linear_name(model)
    if last is not None:
        return dict(model.named_modules())[last].out_features
    return None


def _replace_common_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    """Best-effort classifier replacement for external repos."""
    
    for attr in ["head", "classifier", "fc"]:
        if hasattr(model, attr):
            layer = getattr(model, attr)
            if isinstance(layer, nn.Linear):
                setattr(model, attr, nn.Linear(layer.in_features, num_classes))
                return model
            
            if isinstance(layer, nn.Sequential):
                modules = list(layer.children())
                for i in range(len(modules) - 1, -1, -1):
                    if isinstance(modules[i], nn.Linear):
                        modules[i] = nn.Linear(modules[i].in_features, num_classes)
                        setattr(model, attr, nn.Sequential(*modules))
                        return model
    
    if hasattr(model, "reset_classifier"):
        try:
            model.reset_classifier(num_classes=num_classes)
            return model
        except TypeError:
            try:
                model.reset_classifier(num_classes)
                return model
            except Exception:
                pass
    
    last = _last_linear_name(model)
    if last is not None:
        parent = model
        parts = last.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
        leaf = parts[-1]
        old = getattr(parent, leaf) if not leaf.isdigit() else parent[int(leaf)]
        if isinstance(old, nn.Linear) and old.out_features != num_classes:
            new_layer = nn.Linear(old.in_features, num_classes)
            if leaf.isdigit():
                parent[int(leaf)] = new_layer
            else:
                setattr(parent, leaf, new_layer)
    return model


class ExternalFactoryWrapper(nn.Module):
    """Create a model from a local cloned official repository.

    This is intentionally generic because VMamba, Vim/Vision-Mamba2 and MedMamba
    repositories expose different factory names across revisions. The config can
    supply either a direct factory name or a list of candidate names.
    """

    def __init__(self, model_cfg: Dict[str, Any], num_classes: int, project_root: Optional[Path] = None):
        super().__init__()
        project_root = project_root or Path.cwd()
        repo_path = Path(model_cfg["repo_path"])
        if not repo_path.is_absolute():
            repo_path = (project_root / repo_path).resolve()
        if not repo_path.exists():
            raise RuntimeError(
                f"External repo path not found: {repo_path}\n"
                "Run scripts/install_wsl_mamba_native.sh or clone the official repository under external_models/."
            )
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        for extra in model_cfg.get("extra_sys_paths", []):
            ep = Path(extra)
            if not ep.is_absolute():
                ep = (project_root / ep).resolve()
            if ep.exists() and str(ep) not in sys.path:
                sys.path.insert(0, str(ep))

        module_names = model_cfg.get("module_candidates") or [model_cfg.get("module")]
        factories = model_cfg.get("factory_candidates") or [model_cfg.get("factory")]
        init_kwargs = dict(model_cfg.get("init_kwargs", {}))
        
        init_kwargs.setdefault("num_classes", num_classes)

        errors = []
        model = None
        for module_name in [m for m in module_names if m]:
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:
                errors.append(f"import {module_name}: {exc}")
                continue
            for factory_name in [f for f in factories if f]:
                if not hasattr(module, factory_name):
                    errors.append(f"{module_name}.{factory_name}: not found")
                    continue
                factory = getattr(module, factory_name)
                try:
                    if isinstance(factory, type):
                        model = factory(**init_kwargs)
                    else:
                        model = factory(**init_kwargs)
                    break
                except TypeError as exc:
                    
                    try:
                        init2 = {k: v for k, v in init_kwargs.items() if k != "num_classes"}
                        model = factory(**init2)
                        model = _replace_common_classifier(model, num_classes)
                        break
                    except Exception as exc2:
                        errors.append(f"{module_name}.{factory_name}: {exc}; retry: {exc2}")
                except Exception as exc:
                    errors.append(f"{module_name}.{factory_name}: {exc}")
            if model is not None:
                break

        if model is None:
            raise RuntimeError(
                f"Could not instantiate external model {model_cfg.get('key')}.\nTried: {errors}\n"
                "Open the official repo and adjust module/factory/init_kwargs in configs/bladder_config.yaml."
            )

        ckpt = model_cfg.get("checkpoint_path")
        if ckpt:
            ckpt_path = Path(ckpt)
            if not ckpt_path.is_absolute():
                ckpt_path = (project_root / ckpt_path).resolve()
            if ckpt_path.exists():
                state = torch.load(
                    ckpt_path,
                    map_location="cpu",
                    weights_only=True,
                )
                if isinstance(state, dict):
                    for k in ["model", "state_dict", "model_state_dict", "net"]:
                        if k in state and isinstance(state[k], dict):
                            state = state[k]
                            break
                missing, unexpected = model.load_state_dict(state, strict=False)
                print(f"[external] loaded checkpoint {ckpt_path} | missing={len(missing)} unexpected={len(unexpected)}")
            else:
                print(f"[external] checkpoint not found, using repo initialization: {ckpt_path}")

        self.model = _replace_common_classifier(model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict) and "logits" in out:
            return out["logits"]
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def _candidate_names(name: str) -> list[str]:
    try:
        base = name.split('.')[0].replace('hf_hub:timm/', '')
        return timm.list_models(f"*{base}*")[:50]
    except Exception:
        return []


def create_model(model_cfg: Dict[str, Any], num_classes: int) -> torch.nn.Module:
    model_type = str(model_cfg.get("type", "timm")).lower()
    pretrained = bool(model_cfg.get("pretrained", True))
    init_kwargs = dict(model_cfg.get("init_kwargs", {}))

    if model_type in {"hf", "hf_image", "hf_image_classification"}:
        return HFImageClassificationWrapper(
            model_id=str(model_cfg["hf_model_id"]),
            num_classes=num_classes,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
            pretrained=pretrained,
        )

    if model_type in {"external", "external_factory", "repo"}:
        return ExternalFactoryWrapper(model_cfg, num_classes=num_classes)

    name = str(model_cfg["timm_name"])
    errors: list[str] = []
    try:
        return timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=num_classes,
            **init_kwargs,
        )
    except Exception as exc:
        errors.append(f"direct '{name}': {exc}")

    hf_hub_id = model_cfg.get("hf_hub_id")
    hf_name = None
    if hf_hub_id:
        hf_name = f"hf_hub:{hf_hub_id}"
    elif name.startswith("mambaout_"):
        hf_name = f"hf_hub:timm/{name}"

    if hf_name:
        try:
            return timm.create_model(
                hf_name,
                pretrained=pretrained,
                num_classes=num_classes,
                **init_kwargs,
            )
        except Exception as exc:
            errors.append(f"hf '{hf_name}': {exc}")

    candidates = _candidate_names(name)
    raise RuntimeError(
        f"Could not create model for key='{model_cfg.get('key', name)}'.\n"
        f"Tried: {errors}\n"
        f"Closest candidates from timm: {candidates}\n"
        "Tip: upgrade timm/huggingface_hub, or set hf_hub_id/type in configs/bladder_config.yaml."
    )


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
