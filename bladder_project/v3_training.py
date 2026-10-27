from __future__ import annotations

import gc
import copy
import hashlib
import json
import os
import random
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from .ontomask_loss import compute_ontomask_loss, mask_scores
from .ontomask_model import VARIANTS
from .seeds import set_seed
from .v3_data import (
    build_fold_datasets,
    dataset_class_weights,
    equal_dataset_sampler,
    global_tensor_cache,
    ontology_matrix,
    set_dataset_epoch_seed,
)
from .v3_metrics import compute_all_metrics
from .v3_models import count_trainable_params, create_study_model


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _effective_method_cfg(
    base: Mapping[str, Any], model_cfg: Mapping[str, Any]
) -> Dict[str, Any]:
    """Apply model-local loss overrides without mutating the study config."""

    result = copy.deepcopy(dict(base))
    overrides = dict(model_cfg.get("method_overrides", {}))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            merged = dict(result[key])
            merged.update(value)
            result[key] = merged
        else:
            result[key] = value
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    from bladder_project.atomic_files import replace_with_retry

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True, default=str), encoding="utf-8")
    replace_with_retry(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    from bladder_project.atomic_files import replace_with_retry

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    replace_with_retry(temporary, path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    from bladder_project.atomic_files import replace_with_retry

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    replace_with_retry(temporary, path)


def _seed_worker(base_seed: int, worker_id: int) -> None:
    seed = int(base_seed + worker_id)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def _to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    output = dict(batch)
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device, non_blocking=True)
    return output


def _class_weights(
    cfg: Mapping[str, Any], frames: Mapping[str, pd.DataFrame], device: torch.device
) -> Dict[str, torch.Tensor]:
    result = {}
    for dataset_name, frame in frames.items():
        dataset_cfg = cfg["datasets"][dataset_name]
        result[dataset_name] = dataset_class_weights(
            frame,
            dataset_name,
            str(dataset_cfg.get("class_column", "class_name_v3")),
            dataset_cfg["classes"],
        ).to(device)
    return result


def _standard_loss(
    output: Mapping[str, Any], batch: Mapping[str, Any], class_weights: Mapping[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    losses = []
    for dataset_name, (indices, logits) in output["fine_logits"].items():
        losses.append(F.cross_entropy(logits, batch["fine_label"][indices], weight=class_weights[dataset_name]))
    if not losses:
        raise RuntimeError("No dataset-specific logits were produced")
    fine = torch.stack(losses).mean()
    zero = fine.detach() * 0.0
    return {"total": fine, "fine": fine, "ontology": zero, "consistency": zero, "mask": zero, "separation": zero, "contrast": zero, "subtype": zero}


def _loss_for(
    model: nn.Module,
    output: Mapping[str, Any],
    batch: Mapping[str, Any],
    ontology_matrices: Mapping[str, torch.Tensor],
    class_weights: Mapping[str, torch.Tensor],
    method_cfg: Mapping[str, Any],
) -> Dict[str, torch.Tensor]:
    
    
    if bool(method_cfg.get("fine_only_supervision", False)):
        return _standard_loss(output, batch, class_weights)
    if bool(getattr(model, "is_dctg", False)):
        return compute_ontomask_loss(
            output,
            batch,
            getattr(model, "variant", VARIANTS["full_dataset_gate"]),
            ontology_matrices,
            class_weights,
            method_cfg.get("loss_weights", {}),
        )
    return _standard_loss(output, batch, class_weights)


def _make_loaders(
    cfg: Mapping[str, Any], fold: int, seed: int, only_dataset: str | None = None
) -> Tuple[DataLoader, Dict[str, DataLoader], Dict[str, DataLoader], Dict[str, pd.DataFrame]]:
    hardware = cfg.get("hardware", {})
    workers = int(hardware.get("num_workers", 4))
    cache_gb = float(cfg.get("cache", {}).get("max_gb", 16))
    if workers > 0:
        process_limit = max(0.25, cache_gb / workers)
        os.environ["ONTOMASK_CACHE_MAX_GB"] = str(process_limit)
        cache = None
    else:
        os.environ["ONTOMASK_CACHE_MAX_GB"] = str(cache_gb)
        cache = global_tensor_cache(cache_gb)
    train_datasets, train_frames = build_fold_datasets(cfg, fold=fold, split="train", cache=cache)
    val_datasets, _ = build_fold_datasets(cfg, fold=fold, split="val", cache=cache)
    revision = cfg.get("revision", {})
    test_datasets = []
    if not bool(revision.get("skip_test", False)):
        test_datasets, _ = build_fold_datasets(cfg, fold=fold, split="test", cache=cache)
    if bool(revision.get("fine_only_supervision", False)):
        
        
        
        for dataset in train_datasets:
            dataset.mask_locator = None
            dataset.df = dataset.df[
                [name for name in ("path", "group_id", dataset.class_column)
                 if name in dataset.df.columns]
            ].copy()
            dataset.ontology_class_column = dataset.class_column
            dataset.ontology_map = {
                name: "normal_or_anatomy" for name in dataset.class_names
            }
    if only_dataset:
        train_datasets = [d for d in train_datasets if d.dataset_name == only_dataset]
        val_datasets = [d for d in val_datasets if d.dataset_name == only_dataset]
        test_datasets = [d for d in test_datasets if d.dataset_name == only_dataset]
        train_frames = {
            name: frame for name, frame in train_frames.items() if name == only_dataset
        }
        if not train_datasets or not val_datasets or (
            not test_datasets and not revision.get("skip_test", False)
        ):
            raise ValueError(f"Unknown or disabled single_dataset={only_dataset!r}")

    if workers == 0 and bool(cfg.get("cache", {}).get("warm_before_training", True)):
        for split_name, datasets in (("train", train_datasets), ("val", val_datasets), ("test", test_datasets)):
            for dataset in datasets:
                report = dataset.warm_cache()
                print(
                    f"[cache-warm] fold={fold} | {split_name}:{dataset.dataset_name} | "
                    f"images={report['images']} | valid_masks={report['valid_masks']} | "
                    f"expected_missing={report['expected_masks_missing']}"
                )
        print(f"[cache-warm] complete | {cache.summary()}")
    elif workers > 0:
        print(
            f"[cache-warm] lazy per-worker | workers={workers} | "
            f"prefetch={int(hardware.get('prefetch_factor', 4))} | total_limit={cache_gb:g} GiB"
        )

    generator = torch.Generator().manual_seed(seed)
    common: Dict[str, Any] = {
        "batch_size": int(cfg["training"].get("batch_size", 32)),
        "num_workers": workers,
        "pin_memory": bool(hardware.get("pin_memory", True)),
        "worker_init_fn": partial(_seed_worker, seed),
        "generator": generator,
    }
    if workers > 0:
        common.update(
            {
                "prefetch_factor": int(hardware.get("prefetch_factor", 4)),
                "persistent_workers": (
                    False if revision.get("resume_training") and not revision.get("persistent_epoch_workers")
                    else bool(hardware.get("persistent_workers", True))
                ),
                "multiprocessing_context": str(hardware.get("multiprocessing_context", "spawn")),
            }
        )
    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        sampler=equal_dataset_sampler(train_datasets, seed=seed),
        drop_last=False,
        **common,
    )
    if revision.get("persistent_epoch_workers") or revision.get("checkpoint_policy") == "fold_end":
        
        
        
        train_loader._revision_epoch_seed_setter = partial(
            set_dataset_epoch_seed, train_loader.dataset
        )
    
    
    
    evaluation_common = {
        "batch_size": int(cfg["training"].get("evaluation_batch_size", common["batch_size"])),
        "num_workers": 0,
        "pin_memory": bool(hardware.get("pin_memory", True)),
    }
    val_loaders = {dataset.dataset_name: DataLoader(dataset, shuffle=False, drop_last=False, **evaluation_common) for dataset in val_datasets}
    test_loaders = {dataset.dataset_name: DataLoader(dataset, shuffle=False, drop_last=False, **evaluation_common) for dataset in test_datasets}
    return train_loader, val_loaders, test_loaders, train_frames


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    ontology_matrices: Mapping[str, torch.Tensor],
    class_weights: Mapping[str, torch.Tensor],
    method_cfg: Mapping[str, Any],
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, torch.Tensor] = {}
    count = 0
    for batch in loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler is not None):
            output = model(batch["image"], batch["dataset_id"])
            losses = _loss_for(model, output, batch, ontology_matrices, class_weights, method_cfg)
        if not torch.isfinite(losses["total"]):
            diagnostics = {name: float(value.detach().float().cpu()) for name, value in losses.items()}
            raise FloatingPointError(f"Non-finite training loss: {diagnostics}")
        if scaler is not None:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        for name, value in losses.items():
            totals[name] = totals.get(name, torch.zeros((), device=device)) + value.detach().float()
        count += 1
    return {name: float((value / max(1, count)).item()) for name, value in totals.items()}


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    dataset_name: str,
    class_names: Sequence[str],
    device: torch.device,
    *,
    export_details: bool = False,
    overlay_dir: Path | None = None,
    overlay_limit: int = 0,
) -> Dict[str, Any]:
    model.eval()
    true_values, pred_values, probabilities = [], [], []
    paths, groups, logit_values, mask_rows = [], [], [], []
    overlays_written = 0
    losses, ontology_true, ontology_prob = [], [], []
    gate_values, ontology_gate_values, mask_dice_values, mask_iou_values = [], [], [], []
    for batch in loader:
        batch_device = _to_device(batch, device)
        output = model(batch_device["image"], batch_device["dataset_id"])
        indices, logits = output["fine_logits"][dataset_name]
        targets = batch_device["fine_label"][indices]
        probability = torch.softmax(logits.float(), dim=-1)
        losses.append(F.cross_entropy(logits.float(), targets).detach())
        true_values.append(targets.cpu())
        pred_values.append(probability.argmax(dim=-1).cpu())
        probabilities.append(probability.cpu())
        if export_details:
            logit_values.append(logits.float().cpu())
        selected = indices.cpu().tolist()
        paths.extend([str(batch["path"][index]) for index in selected])
        groups.extend([str(batch["group_id"][index]) for index in selected])
        if "ontology_logits" in output:
            ontology_true.append(batch_device["ontology_label"].cpu())
            ontology_prob.append(torch.softmax(output["ontology_logits"].float(), dim=-1).cpu())
        if "gate_weights" in output:
            gate_values.append(output["gate_weights"].float().cpu())
        if "ontology_residual_gate" in output:
            ontology_gate_values.append(output["ontology_residual_gate"].float().cpu())
        valid = batch_device["mask_valid"].bool()
        if bool(valid.any()) and "mask_logits" in output:
            target_mask = F.interpolate(batch_device["mask"][valid], size=output["mask_logits"].shape[-2:], mode="nearest")
            values = mask_scores(output["mask_logits"][valid], target_mask)
            mask_dice_values.append(values["mask_dice"].cpu())
            mask_iou_values.append(values["mask_iou"].cpu())
            if export_details:
                valid_indices = torch.nonzero(valid, as_tuple=False).flatten().tolist()
                prediction = (torch.sigmoid(output["mask_logits"][valid].float()) >= 0.5).float()
                truth = (target_mask >= 0.5).float()
                for local_index, batch_index in enumerate(valid_indices):
                    predicted_mask = prediction[local_index]
                    true_mask = truth[local_index]
                    intersection = float((predicted_mask * true_mask).sum().item())
                    predicted_area = float(predicted_mask.sum().item())
                    true_area = float(true_mask.sum().item())
                    mask_rows.append({
                        "image_path": str(batch["path"][batch_index]),
                        "group_id": str(batch["group_id"][batch_index]),
                        "mask_dice": (2 * intersection + 1) / (predicted_area + true_area + 1),
                        "mask_iou": (intersection + 1) / (predicted_area + true_area - intersection + 1),
                        "true_mask_pixels": true_area,
                        "predicted_mask_pixels": predicted_area,
                        "mask_threshold": 0.5,
                        "mask_height": int(predicted_mask.shape[-2]),
                        "mask_width": int(predicted_mask.shape[-1]),
                    })
                    if overlay_dir is not None and overlays_written < overlay_limit:
                        from PIL import Image, ImageDraw
                        image_tensor = batch["image"][batch_index].float().cpu()
                        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
                        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
                        pixels = ((image_tensor * std + mean).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
                        original = Image.fromarray(pixels)
                        width, height = original.size
                        panel = Image.new("RGB", (width * 3, height + 24), "white")
                        for panel_index, (label, selected_mask) in enumerate(
                            (("Image", None), ("Annotated mask", true_mask), ("Predicted mask", predicted_mask))
                        ):
                            display = original.copy()
                            if selected_mask is not None:
                                mask_image = Image.fromarray((selected_mask.squeeze(0).cpu().numpy() * 120).astype("uint8"))
                                mask_image = mask_image.resize(original.size, Image.Resampling.NEAREST)
                                display.paste(Image.new("RGB", original.size, (255, 0, 0)), mask=mask_image)
                            panel.paste(display, (panel_index * width, 24))
                            ImageDraw.Draw(panel).text((panel_index * width + 4, 5), label, fill="black")
                        overlay_dir.mkdir(parents=True, exist_ok=True)
                        filename = hashlib.sha256(str(batch["path"][batch_index]).encode()).hexdigest()[:16]
                        panel.save(overlay_dir / f"{filename}.png")
                        overlays_written += 1

    y_true = torch.cat(true_values).numpy()
    y_pred = torch.cat(pred_values).numpy()
    y_prob = torch.cat(probabilities).numpy()
    metrics, per_class, _ = compute_all_metrics(y_true, y_pred, y_prob, class_names)
    
    
    
    
    metrics["fine_ce_loss"] = float(torch.stack(losses).mean().item())
    metrics["mask_dice"] = float(torch.stack(mask_dice_values).mean().item()) if mask_dice_values else float("nan")
    metrics["mask_iou"] = float(torch.stack(mask_iou_values).mean().item()) if mask_iou_values else float("nan")
    if gate_values:
        gates = torch.cat(gate_values).numpy()
        metrics.update({"gate_roi_mean": float(gates[:, 0].mean()), "gate_context_mean": float(gates[:, 1].mean()), "gate_global_mean": float(gates[:, 2].mean())})
    else:
        metrics.update({"gate_roi_mean": float("nan"), "gate_context_mean": float("nan"), "gate_global_mean": float("nan")})
    metrics["ontology_residual_gate_abs_mean"] = (
        float(torch.cat(ontology_gate_values).abs().mean().item())
        if ontology_gate_values
        else float("nan")
    )
    frame = pd.DataFrame(
        {
            "image_path": paths,
            "group_id": groups,
            "true_index": y_true,
            "true_class": [class_names[index] for index in y_true],
            "predicted_index": y_pred,
            "predicted_class": [class_names[index] for index in y_pred],
            "correct": y_true == y_pred,
        }
    )
    for index, class_name in enumerate(class_names):
        frame[f"prob_{class_name}"] = y_prob[:, index]
        if export_details:
            frame[f"logit_{class_name}"] = torch.cat(logit_values).numpy()[:, index]
    if export_details and mask_rows:
        
        
        metrics["mask_dice"] = float(np.mean([row["mask_dice"] for row in mask_rows]))
        metrics["mask_iou"] = float(np.mean([row["mask_iou"] for row in mask_rows]))
        metrics["mask_annotated_n"] = len(mask_rows)
    return {
        "metrics": metrics, "per_class": per_class, "predictions": frame,
        "class_names": list(class_names),
        "mask_predictions": pd.DataFrame(mask_rows, columns=[
            "image_path", "group_id", "mask_dice", "mask_iou", "true_mask_pixels",
            "predicted_mask_pixels", "mask_threshold", "mask_height", "mask_width",
        ]),
    }


def _estimate_complexity(model: nn.Module, image_size: int, device: torch.device) -> Tuple[float, float]:
    params = count_trainable_params(model) / 1e6
    flops = float("nan")
    try:
        from thop import profile

        model.eval()
        image = torch.randn(1, 3, image_size, image_size, device=device)
        dataset_ids = torch.zeros(1, dtype=torch.long, device=device)
        operations, _ = profile(model, inputs=(image, dataset_ids), verbose=False)
        flops = float(operations) / 1e9
    except Exception as exc:
        print(f"[flops][WARN] {exc}")
    finally:
        for module in model.modules():
            module._buffers.pop("total_ops", None)
            module._buffers.pop("total_params", None)
    return float(params), float(flops)


def _clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _epoch_seed(loader: DataLoader, fold_seed: int, epoch: int) -> int:
    """Reset model, sampler and worker RNG independently of earlier iterators."""
    epoch_seed = int(fold_seed + epoch * 1000003)
    set_seed(epoch_seed)
    loader.worker_init_fn = partial(_seed_worker, epoch_seed)
    for generator in (getattr(loader, "generator", None), getattr(loader.sampler, "generator", None)):
        if generator is not None:
            generator.manual_seed(epoch_seed)
    worker_epoch_setter = getattr(loader, "_revision_epoch_seed_setter", None)
    if worker_epoch_setter is not None:
        worker_epoch_setter(epoch_seed)
    return epoch_seed


def _rng_state(loader: DataLoader) -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": loader.generator.get_state() if loader.generator is not None else None,
        "sampler": (
            loader.sampler.generator.get_state()
            if getattr(loader.sampler, "generator", None) is not None else None
        ),
    }


def _restore_rng(state: Mapping[str, Any], loader: DataLoader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    for name, generator in (
        ("loader", getattr(loader, "generator", None)),
        ("sampler", getattr(loader.sampler, "generator", None)),
    ):
        if generator is not None and state.get(name) is not None:
            generator.set_state(state[name])


def _validate_resume_state(state: Mapping[str, Any], config_hash: str, split_hash: str) -> None:
    if state.get("config_sha256") != config_hash or state.get("split_manifest_sha256") != split_hash:
        raise ValueError("Resume checkpoint config/split hashes differ; use a new run_id, never overwrite this run")
    required = {"model", "optimizer", "scheduler", "rng", "epoch_history", "best_state", "next_epoch"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"Incomplete epoch resume checkpoint: {sorted(missing)}")


def _completion_valid(
    fold_dir: Path,
    config_hash: str,
    split_hash: str,
    expected_datasets: Sequence[str],
    revision: Mapping[str, Any] | None = None,
) -> bool:
    path = fold_dir / "completion.json"
    if not path.exists() or not (fold_dir / "metrics.json").exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        requirements = revision or {}
        additional_files = []
        if requirements.get("require_checkpoint") or requirements.get("checkpoint_policy") == "fold_end":
            additional_files.append(fold_dir / "best_model.pt")
        if requirements.get("export_validation"):
            additional_files.extend(fold_dir / f"predictions_val_{name}.csv" for name in expected_datasets)
        if requirements.get("export_details"):
            additional_files.extend(fold_dir / f"masks_val_{name}.csv" for name in expected_datasets)
            if not requirements.get("skip_test"):
                additional_files.extend(fold_dir / f"masks_test_{name}.csv" for name in expected_datasets)
        return bool(
            payload.get("status") == "complete"
            and payload.get("config_sha256") == config_hash
            and payload.get("split_manifest_sha256") == split_hash
            and all(
                (fold_dir / f"predictions_{dataset}.csv").exists()
                for dataset in ([] if requirements.get("skip_test") else expected_datasets)
            )
            and all(file.is_file() and file.stat().st_size > 0 for file in additional_files)
        )
    except Exception:
        return False


def run_study(cfg: MutableMapping[str, Any]) -> None:
    project_root = Path(cfg["_project_root"])
    revision = cfg.get("revision", {})
    checkpoint_policy = str(revision.get("checkpoint_policy", "epoch"))
    if checkpoint_policy not in {"epoch", "fold_end"}:
        raise ValueError(f"Unknown revision.checkpoint_policy={checkpoint_policy!r}")
    epoch_checkpointing = bool(revision.get("resume_training")) and checkpoint_policy != "fold_end"
    reseed_each_epoch = bool(revision.get("resume_training")) or checkpoint_policy == "fold_end"
    if (revision.get("require_checkpoint") or checkpoint_policy == "fold_end") and not cfg["training"].get("save_best_checkpoint"):
        raise ValueError("revision.require_checkpoint requires training.save_best_checkpoint")
    device = torch.device(str(cfg.get("hardware", {}).get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    use_amp = bool(cfg.get("hardware", {}).get("mixed_precision", True)) and device.type == "cuda"
    selected_keys = list(
        dict.fromkeys(cfg.get("_only_models") or cfg.get("run", {}).get("models", []))
    )
    catalog = {model["key"]: model for model in cfg["models"]}
    missing_keys = [key for key in selected_keys if key not in catalog]
    if missing_keys:
        raise KeyError(f"Selected model keys are not defined: {missing_keys}")
    models = [catalog[key] for key in selected_keys] if selected_keys else list(cfg["models"])
    folds = list(cfg.get("_only_folds") or range(1, 6))
    dataset_classes = {name: value["classes"] for name, value in cfg["datasets"].items()}
    config_for_hash = copy.deepcopy(
        {key: value for key, value in cfg.items() if not str(key).startswith("_")}
    )
    
    
    
    config_for_hash.pop("cache", None)
    if not reseed_each_epoch:
        for key in ("num_workers", "prefetch_factor", "persistent_workers", "multiprocessing_context", "pin_memory"):
            config_for_hash.get("hardware", {}).pop(key, None)
    config_for_hash.pop("reporting", None)
    config_hash = _json_hash(config_for_hash)
    manifest_files = []
    for name, dataset_cfg in cfg["datasets"].items():
        configured_root = Path(str(dataset_cfg.get("locked_fold_root", "")))
        locked_root = (
            configured_root
            if str(configured_root) not in {"", "."}
            else project_root / "data" / "locked_folds" / name
        )
        manifest_files.append(locked_root / "split_manifest.json")
    if not all(path.exists() for path in manifest_files):
        raise FileNotFoundError("Locked fold manifests are missing. Run 01_prepare_data.cmd first.")
    split_hash = _json_hash({path.name + str(index): hashlib.sha256(path.read_bytes()).hexdigest() for index, path in enumerate(manifest_files)})
    protocol_id = str(cfg.get("protocol_id", "locked5_seed42"))
    run_id = str(cfg.get("run_id", "confirmatory"))
    results_root = project_root / str(cfg.get("results", {}).get("root", "results"))
    output_root = results_root / "runs" / protocol_id / run_id

    requested_fold_total = len(models) * len(folds)
    completed_requested_folds = 0
    if not cfg.get("_force", False):
        for progress_model_cfg in models:
            progress_scope = str(
                progress_model_cfg.get("training_scope", "joint")
            )
            progress_only_dataset = (
                str(progress_model_cfg["single_dataset"])
                if progress_scope == "single_dataset"
                else None
            )
            progress_expected_datasets = (
                [progress_only_dataset]
                if progress_only_dataset
                else list(dataset_classes)
            )
            for progress_fold in folds:
                progress_fold_dir = (
                    output_root
                    / str(progress_model_cfg["key"])
                    / f"fold_{progress_fold}"
                )
                completed_requested_folds += int(
                    _completion_valid(
                        progress_fold_dir,
                        config_hash,
                        split_hash,
                        progress_expected_datasets,
                        revision,
                    )
                )

    global_fold_total = int(
        cfg.get("_progress_total_units", requested_fold_total)
    )
    global_completed_offset = int(cfg.get("_progress_completed_offset", 0))

    def progress_text() -> str:
        requested_remaining = max(
            0, requested_fold_total - completed_requested_folds
        )
        global_completed = global_completed_offset + completed_requested_folds
        global_remaining = max(0, global_fold_total - global_completed)
        if global_fold_total == requested_fold_total and global_completed_offset == 0:
            return f"fold_kalan/toplam={requested_remaining}/{requested_fold_total}"
        return (
            f"fold_kalan/toplam={global_remaining}/{global_fold_total} | "
            f"calisma_kalan/toplam={requested_remaining}/{requested_fold_total}"
        )

    print(f"[progress] {progress_text()}", flush=True)

    for model_cfg in models:
        training_scope = str(model_cfg.get("training_scope", "joint"))
        only_dataset = (
            str(model_cfg["single_dataset"])
            if training_scope == "single_dataset"
            else None
        )
        expected_datasets = [only_dataset] if only_dataset else list(dataset_classes)
        scoped_classes = (
            {only_dataset: dataset_classes[only_dataset]}
            if only_dataset
            else dataset_classes
        )
        effective_model_cfg = copy.deepcopy(model_cfg)
        if only_dataset:
            effective_model_cfg["dataset_id_by_name"] = {
                only_dataset: list(cfg["datasets"]).index(only_dataset)
            }
        effective_method_cfg = _effective_method_cfg(cfg["method"], model_cfg)
        for fold in folds:
            fold_seed = int(cfg.get("seed", 42)) + int(fold)
            fold_dir = output_root / str(model_cfg["key"]) / f"fold_{fold}"
            if not cfg.get("_force", False) and _completion_valid(
                fold_dir, config_hash, split_hash, expected_datasets, revision
            ):
                print(
                    f"[resume] skip completed | {model_cfg['key']} | "
                    f"fold={fold} | {progress_text()}"
                )
                continue
            if checkpoint_policy == "fold_end":
                if (fold_dir / "last_training_state.pt").exists():
                    raise ValueError(
                        "fold_end policy cannot reuse epoch checkpoints; archive the incomplete fold "
                        "and start a new run_id"
                    )
                
                
                _atomic_json(fold_dir / "progress.json", {
                    "status": "initializing", "updated_at": datetime_now(),
                    "epoch": 0, "max_epochs": int(cfg["training"].get("max_epochs", 100)),
                    "best_epoch": 0, "best_val_f1": None,
                    "model_key": model_cfg["key"], "fold": fold, "seed": fold_seed,
                    "epoch_history": [], "last_durable_epoch": None,
                    "checkpoint_policy": checkpoint_policy,
                    "resume_policy": "restart_incomplete_fold_from_seed",
                })
            scope_label = only_dataset if only_dataset else "CystoDS+EBTC+KaggleWLI"
            print(
                f"\n--- {scope_label} | {model_cfg['display_name']} | "
                f"Fold {fold}/5 | {progress_text()} ---"
            )
            set_seed(fold_seed)
            train_loader, val_loaders, test_loaders, train_frames = _make_loaders(
                cfg, fold, fold_seed, only_dataset=only_dataset
            )
            model = create_study_model(
                effective_model_cfg, scoped_classes, effective_method_cfg
            ).to(device)
            params_m, flops_g = _estimate_complexity(model, int(cfg["training"]["image_size"]), device)
            class_weights = _class_weights(cfg, train_frames, device)
            ontology_matrices = {
                name: ontology_matrix(cfg["datasets"][name], name).to(device)
                for name in scoped_classes
            }
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(cfg["training"].get("learning_rate", 1e-4)),
                weight_decay=float(cfg["training"].get("weight_decay", 1e-2)),
            )
            max_epochs = int(cfg["training"].get("max_epochs", 100))
            early_stopping_enabled = bool(
                cfg["training"].get("early_stopping_enabled", True)
            )
            patience = int(cfg["training"].get("early_stopping_patience", 15))
            if early_stopping_enabled and patience <= 0:
                raise ValueError(
                    "early_stopping_patience must be positive when early stopping is enabled"
                )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
            scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
            best_f1, best_epoch, best_state, bad_epochs = -float("inf"), 0, None, 0
            epoch_history, epoch_times = [], []
            training_start = time.perf_counter()
            checkpoint_path = fold_dir / "best_model.pt"
            last_state_path = fold_dir / "last_training_state.pt"
            first_epoch = 1
            if epoch_checkpointing and last_state_path.exists():
                
                
                resume = torch.load(last_state_path, map_location="cpu", weights_only=False)
                _validate_resume_state(resume, config_hash, split_hash)
                model.load_state_dict(resume["model"], strict=True)
                optimizer.load_state_dict(resume["optimizer"])
                scheduler.load_state_dict(resume["scheduler"])
                if scaler is not None and resume.get("scaler") is not None:
                    scaler.load_state_dict(resume["scaler"])
                best_f1, best_epoch = float(resume["best_f1"]), int(resume["best_epoch"])
                best_state, bad_epochs = resume["best_state"], int(resume["bad_epochs"])
                epoch_history, epoch_times = resume["epoch_history"], resume["epoch_times"]
                first_epoch = int(resume["next_epoch"])
                _restore_rng(resume["rng"], train_loader)
                del resume
                print(f"[resume-epoch] model={model_cfg['key']} fold={fold} next_epoch={first_epoch}", flush=True)

            def checkpoint_payload() -> dict:
                return {
                    "model_state_dict": best_state,
                    "best_epoch": best_epoch, "best_val_f1": best_f1,
                    "selection_metric": "validation_macro_f1",
                    "model_key": model_cfg["key"], "fold": fold, "seed": fold_seed,
                    "run_id": run_id, "protocol_id": protocol_id,
                    "config_sha256": config_hash,
                    "split_manifest_sha256": split_hash,
                    "revision_provenance": revision.get("provenance", {}),
                    "checkpoint_write_policy": checkpoint_policy,
                }

            if checkpoint_policy == "fold_end":
                _atomic_json(fold_dir / "progress.json", {
                    "status": "training", "updated_at": datetime_now(),
                    "epoch": 0, "max_epochs": max_epochs,
                    "best_epoch": 0, "best_val_f1": None,
                    "model_key": model_cfg["key"], "fold": fold, "seed": fold_seed,
                    "epoch_history": [], "last_durable_epoch": None,
                    "checkpoint_policy": checkpoint_policy,
                    "resume_policy": "restart_incomplete_fold_from_seed",
                })
            for epoch in range(first_epoch, max_epochs + 1):
                if reseed_each_epoch:
                    _epoch_seed(train_loader, fold_seed, epoch)
                epoch_start = time.perf_counter()
                train_losses = _train_epoch(
                    model, train_loader, optimizer, scaler, device, ontology_matrices,
                    class_weights, effective_method_cfg,
                    float(cfg["training"].get("gradient_clip_norm", 1.0)),
                )
                validation = {
                    name: _evaluate(model, loader, name, dataset_classes[name], device)
                    for name, loader in val_loaders.items()
                }
                f1 = float(np.mean([result["metrics"]["f1_macro"] for result in validation.values()]))
                auroc_values = [result["metrics"]["roc_auc_ovr_macro"] for result in validation.values()]
                auroc = float(np.nanmean(auroc_values))
                accuracy = float(np.mean([result["metrics"]["accuracy"] for result in validation.values()]))
                val_loss = float(np.mean([result["metrics"]["fine_ce_loss"] for result in validation.values()]))
                epoch_sec = time.perf_counter() - epoch_start
                epoch_times.append(epoch_sec)
                if f1 > best_f1 + float(cfg["training"].get("early_stopping_min_delta", 1e-6)):
                    best_f1, best_epoch, best_state, bad_epochs = f1, epoch, _clone_state(model), 0
                    if revision.get("require_checkpoint") and checkpoint_policy != "fold_end":
                        _atomic_torch_save(checkpoint_path, checkpoint_payload())
                else:
                    bad_epochs += 1
                row = {
                    "model": model_cfg["display_name"], "model_key": model_cfg["key"], "family": model_cfg["family"],
                    "fold": fold, "seed": fold_seed, "epoch": epoch, "val_f1_macro": f1,
                    "val_auroc_macro": auroc, "val_accuracy": accuracy, "train_loss": train_losses["total"],
                    "val_fine_ce_loss": val_loss, "epoch_sec": epoch_sec, "esb": bad_epochs, "best_val_f1": best_f1,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                loss_weights = effective_method_cfg.get("loss_weights", {})
                row["train_total_objective"] = train_losses["total"]
                for name, value in train_losses.items():
                    if name == "total":
                        continue
                    row[f"train_{name}_loss_raw"] = value
                    row[f"train_{name}_weighted_contribution"] = (
                        value * float(loss_weights.get(name, 0.0))
                    )
                epoch_history.append(row)
                print(
                    f"{epoch:03d}/{max_epochs:03d} | {scope_label} | Fold {fold} | {model_cfg['display_name']} | "
                    f"f1={f1:.4f} | auroc={auroc:.4f} | acc={accuracy:.4f} | loss={val_loss:.4f} | "
                    f"time={epoch_sec:.1f}s | esb={bad_epochs} | best={best_f1:.4f} | "
                    f"{progress_text()}",
                    flush=True,
                )
                scheduler.step()
                if epoch_checkpointing:
                    _atomic_torch_save(last_state_path, {
                        "config_sha256": config_hash, "split_manifest_sha256": split_hash,
                        "next_epoch": epoch + 1, "model": _clone_state(model),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "best_state": best_state, "best_f1": best_f1, "best_epoch": best_epoch,
                        "bad_epochs": bad_epochs, "epoch_history": epoch_history,
                        "epoch_times": epoch_times, "rng": _rng_state(train_loader),
                    })
                if revision:
                    _atomic_json(fold_dir / "progress.json", {
                        "status": "training", "updated_at": datetime_now(),
                        "epoch": epoch, "max_epochs": max_epochs, "best_epoch": best_epoch,
                        "best_val_f1": best_f1, "model_key": model_cfg["key"], "fold": fold,
                        "epoch_sec": epoch_sec, "epoch_history": epoch_history,
                        "last_durable_epoch": epoch if epoch_checkpointing else None,
                        "checkpoint_policy": checkpoint_policy,
                        "resume_policy": (
                            "resume_after_completed_epoch_repeat_interrupted_epoch"
                            if epoch_checkpointing else "restart_incomplete_fold_from_seed"
                        ),
                    })
                if early_stopping_enabled and bad_epochs >= patience:
                    print(f"[early-stop] patience={patience} | best_epoch={best_epoch} | best_f1={best_f1:.4f}")
                    break
            if best_state is None:
                raise RuntimeError(f"No finite best state for {model_cfg['key']} fold {fold}")
            checkpoint_saved = bool(
                cfg["training"].get("save_best_checkpoint", False)
            )
            if checkpoint_saved:
                _atomic_torch_save(
                    checkpoint_path,
                    checkpoint_payload(),
                )
            model.load_state_dict(best_state, strict=True)
            export_details = bool(revision.get("export_details", False))

            def evaluate_export(name: str, loader: DataLoader, split: str) -> dict:
                result = _evaluate(
                    model, loader, name, dataset_classes[name], device,
                    export_details=export_details,
                    overlay_dir=(fold_dir / "mask_overlays" / split / name if export_details else None),
                    overlay_limit=int(revision.get("mask_overlay_limit", 8)),
                )
                if export_details:
                    _atomic_csv(fold_dir / f"masks_{split}_{name}.csv", result["mask_predictions"])
                return result

            validation_results = {}
            if revision.get("export_validation"):
                validation_results = {
                    name: evaluate_export(name, loader, "val")
                    for name, loader in val_loaders.items()
                }
                for name, result in validation_results.items():
                    _atomic_csv(fold_dir / f"predictions_val_{name}.csv", result["predictions"])
            test_results = {
                name: evaluate_export(name, loader, "test") for name, loader in test_loaders.items()
            }
            fold_dir.mkdir(parents=True, exist_ok=True)
            for dataset, result in test_results.items():
                _atomic_csv(fold_dir / f"predictions_{dataset}.csv", result["predictions"])
                print(
                    f"[test] {dataset} | {model_cfg['display_name']} | fold={fold} | "
                    f"f1={result['metrics']['f1_macro']:.4f} | auroc={result['metrics']['roc_auc_ovr_macro']:.4f} | "
                    f"acc={result['metrics']['accuracy']:.4f}"
                )
            total_training_sec = time.perf_counter() - training_start
            metrics_payload = {
                "run_id": run_id, "protocol_id": protocol_id, "config_sha256": config_hash,
                "split_manifest_sha256": split_hash, "model_key": model_cfg["key"],
                "model_display": model_cfg["display_name"], "model_family": model_cfg["family"],
                "architecture": model_cfg["architecture"], "fold": fold, "seed": fold_seed,
                "training_scope": training_scope,
                "single_dataset": only_dataset or "",
                "variant": model_cfg.get("variant", "standard"),
                "pretraining": model_cfg.get("pretraining", "unspecified"),
                "selection_role": model_cfg.get("selection_role", ""),
                "epochs_run": len(epoch_history), "best_epoch": best_epoch, "best_val_f1": best_f1,
                "avg_epoch_sec": float(np.mean(epoch_times)), "total_training_sec": total_training_sec,
                "params_m": params_m, "flops_g": flops_g,
                "early_stopping_enabled": early_stopping_enabled,
                "checkpoint_policy": "best_validation_macro_f1",
                "checkpoint_write_policy": checkpoint_policy,
                "best_checkpoint_path": (
                    (
                        str(checkpoint_path.relative_to(project_root))
                        if checkpoint_path.is_relative_to(project_root)
                        else str(checkpoint_path)
                    )
                    if checkpoint_saved
                    else ""
                ),
                "augmentation_enabled": bool(cfg.get("augmentation", {}).get("enabled", False)),
                "epoch_history": epoch_history,
                "test": {name: {"metrics": result["metrics"], "per_class": result["per_class"], "class_names": result["class_names"]} for name, result in test_results.items()},
            }
            if revision:
                metrics_payload["validation"] = {
                    name: {"metrics": result["metrics"], "class_names": result["class_names"]}
                    for name, result in validation_results.items()
                }
                metrics_payload["revision_provenance"] = revision.get("provenance", {})
                metrics_payload["test_evaluated"] = not bool(revision.get("skip_test"))
                metrics_payload["total_epoch_sec"] = float(sum(epoch_times))
                metrics_payload["resume_policy"] = {
                    "enabled": epoch_checkpointing,
                    "reseed_each_epoch": reseed_each_epoch,
                    "worker_persistence": bool(train_loader.persistent_workers),
                    "incomplete_fold": (
                        "resume_after_completed_epoch_repeat_interrupted_epoch"
                        if epoch_checkpointing else "restart_incomplete_fold_from_seed"
                    ),
                    "limitation": (
                        "Same device/software expected; CUDA kernels may still be nondeterministic. "
                        + ("Interrupted epoch repeats." if epoch_checkpointing else "Interrupted fold restarts from its seed.")
                    ),
                }
            _atomic_json(fold_dir / "metrics.json", metrics_payload)
            _atomic_json(
                fold_dir / "completion.json",
                {
                    "status": "complete", "completed_at": datetime_now(), "config_sha256": config_hash,
                    "split_manifest_sha256": split_hash, "model_key": model_cfg["key"], "fold": fold,
                },
            )
            if revision:
                _atomic_json(fold_dir / "progress.json", {
                    "status": "complete", "updated_at": datetime_now(),
                    "epoch": len(epoch_history), "max_epochs": max_epochs,
                    "best_epoch": best_epoch, "best_val_f1": best_f1,
                    "model_key": model_cfg["key"], "fold": fold,
                    "checkpoint_policy": checkpoint_policy,
                    "last_durable_epoch": len(epoch_history) if epoch_checkpointing else None,
                    "checkpoint_written": checkpoint_saved,
                    "resume_policy": (
                        "resume_after_completed_epoch_repeat_interrupted_epoch"
                        if epoch_checkpointing else "restart_incomplete_fold_from_seed"
                    ),
                })
            completed_requested_folds += 1
            print(
                f"[progress] completed | model={model_cfg['key']} | "
                f"fold={fold} | {progress_text()}",
                flush=True,
            )
            del model, optimizer, scheduler, scaler, best_state
            del train_loader, val_loaders, test_loaders
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(
        f"[study] all requested folds completed | {progress_text()}",
        flush=True,
    )


def datetime_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
