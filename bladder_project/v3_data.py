from __future__ import annotations

import json
import hashlib
import math
import os
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageOps
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler, get_worker_info
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ONTOLOGY_CLASSES = (
    "neoplastic",
    "non_neoplastic_lesion",
    "normal_or_anatomy",
    "instrument_or_procedure",
)


def _norm_label(value: Any) -> str:
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


CLASS_ALIASES: Dict[str, Dict[str, str]] = {
    "cystods": {
        "malignant": "malignant",
        "non malignant": "non-malignant",
        "anatomical": "anatomical",
        "anatomical landmarks": "anatomical",
        "foreign": "foreign bodies",
        "foreign body": "foreign bodies",
        "foreign bodies": "foreign bodies",
        "normal": "normal",
        "normal mucosa": "normal",
    },
    "ebtc": {
        "hgc": "HGC",
        "lgc": "LGC",
        "nst": "NST",
        "ntl": "NTL",
    },
    "kaggle_wli": {
        "above t1": "Above T1",
        "hose": "Hose",
        "loop": "Loop",
        "normal": "Normal",
        "ta": "TA",
        "t a": "TA",
        "turbt": "TURBT",
    },
}


DEFAULT_ONTOLOGY_MAP: Dict[str, Dict[str, str]] = {
    "cystods": {
        "malignant": "neoplastic",
        "non-malignant": "non_neoplastic_lesion",
        "anatomical": "normal_or_anatomy",
        "foreign bodies": "instrument_or_procedure",
        "normal": "normal_or_anatomy",
    },
    "ebtc": {
        "HGC": "neoplastic",
        "LGC": "neoplastic",
        "NST": "normal_or_anatomy",
        "NTL": "non_neoplastic_lesion",
    },
    "kaggle_wli": {
        "Above T1": "neoplastic",
        "Hose": "instrument_or_procedure",
        "Loop": "instrument_or_procedure",
        "Normal": "normal_or_anatomy",
        "TA": "neoplastic",
        "TURBT": "instrument_or_procedure",
    },
}


def canonical_class(dataset_name: str, value: Any) -> str:
    dataset_name = str(dataset_name)
    normalized = _norm_label(value)
    aliases = CLASS_ALIASES.get(dataset_name, {})
    if normalized not in aliases:
        raise ValueError(
            f"Unknown class {value!r} for {dataset_name}. "
            f"Known normalized labels: {sorted(aliases)}"
        )
    return aliases[normalized]


def windows_path_to_wsl(raw: str) -> str:
    value = str(raw).strip().replace("\\", "/")
    match = re.match(r"^([A-Za-z]):/(.*)$", value)
    if match:
        return f"/mnt/{match.group(1).lower()}/{match.group(2)}"
    return value


def resolve_existing_path(
    raw: str | Path,
    project_root: str | Path,
    baseline_root: str | Path,
) -> Path:
    """Resolve CSV paths copied between Windows projects without rewriting CSV files."""
    raw_text = str(raw).strip()
    direct_candidates = [Path(raw_text), Path(windows_path_to_wsl(raw_text))]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate.resolve()

    normalized = raw_text.replace("\\", "/")
    lower = normalized.lower()
    roots = [Path(project_root), Path(baseline_root)]
    for marker in ("data/raw/", "data/processed/", "data/augmented/"):
        pos = lower.find(marker)
        if pos < 0:
            continue
        suffix = normalized[pos:]
        for root in roots:
            candidate = root / Path(suffix)
            if candidate.exists():
                return candidate.resolve()

    raise FileNotFoundError(
        f"Image/path not found: {raw}\n"
        f"Tried as-is, WSL drive conversion, project_root={project_root}, "
        f"baseline_root={baseline_root}."
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NativeDataIndex:
    """Verified copies used for IO; canonical source paths remain public IDs."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema") != 1 or payload.get("status") != "verified":
            raise ValueError(f"Unverified native data index: {self.path}")
        self.root = Path(payload["cache_root"]).resolve()
        self.entries = payload["entries"]
        self.aliases = payload["aliases"]
        self.annotations = payload["annotations"]
        self._verified: Dict[str, Tuple[int, int]] = {}
        
        
        for source in self.entries:
            self.read_path(Path(source))

    def source_path(self, raw: str | Path, project_root: Path, baseline_root: Path) -> Path:
        alias = str(raw).strip().replace("\\", "/")
        source = self.aliases.get(alias)
        if source is None:
            source = str(resolve_existing_path(raw, project_root, baseline_root))
        if source not in self.entries:
            raise ValueError(f"Source is absent from verified native data index: {source}")
        return Path(source)

    def read_path(self, source: Path) -> Path:
        entry = self.entries.get(str(source))
        if entry is None:
            raise ValueError(f"Source is absent from verified native data index: {source}")
        cached = Path(entry["cached_path"])
        if not cached.is_absolute() or self.root not in cached.resolve().parents:
            raise ValueError(f"Cached object escapes native cache root: {cached}")
        stat = cached.stat()
        stamp = (int(stat.st_size), int(stat.st_mtime_ns))
        if stat.st_size != int(entry["size"]):
            raise ValueError(f"Native cache size mismatch: {cached}")
        if self._verified.get(str(cached)) != stamp:
            if _file_sha256(cached) != entry["sha256"]:
                raise ValueError(f"Native cache SHA256 mismatch: {cached}")
            self._verified[str(cached)] = stamp
        return cached


_NATIVE_INDICES: Dict[Tuple[str, int, int], NativeDataIndex] = {}


def native_data_index(path: str | Path | None) -> Optional[NativeDataIndex]:
    if not path:
        return None
    index_path = Path(path)
    stat = index_path.stat()
    key = (str(index_path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    if key not in _NATIVE_INDICES:
        _NATIVE_INDICES[key] = NativeDataIndex(index_path)
    return _NATIVE_INDICES[key]


_WORKER_EPOCH_TOKEN: Optional[Tuple[int, int]] = None


def set_dataset_epoch_seed(dataset: Dataset, seed: int) -> None:
    """Update epoch seed before iterating a loader, including concat children."""
    setter = getattr(dataset, "set_epoch_seed", None)
    if setter is not None:
        setter(int(seed))
    elif isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            set_dataset_epoch_seed(child, seed)


class TensorLRUCache:
    """Process-local cache of resized, unnormalized tensors.

    Random augmentation is applied after this cache. Spawned workers receive an
    empty cache and use a per-process share of the configured 16 GiB budget.
    """

    def __init__(self, max_gb: float = 16.0):
        self.max_bytes = max(0, int(float(max_gb) * 1024**3))
        self.used_bytes = 0
        self.items: OrderedDict[str, Tuple[torch.Tensor, torch.Tensor, bool, int]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _nbytes(image: torch.Tensor, mask: torch.Tensor) -> int:
        return image.numel() * image.element_size() + mask.numel() * mask.element_size()

    def get(self, key: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, bool]]:
        item = self.items.get(key)
        if item is None:
            self.misses += 1
            return None
        self.hits += 1
        self.items.move_to_end(key)
        image, mask, valid, _ = item
        return image, mask, valid

    def put(self, key: str, image: torch.Tensor, mask: torch.Tensor, valid: bool) -> None:
        if key in self.items:
            return
        size = self._nbytes(image, mask)
        if self.max_bytes and size > self.max_bytes:
            return
        while self.max_bytes and self.items and self.used_bytes + size > self.max_bytes:
            _, (_, _, _, old_size) = self.items.popitem(last=False)
            self.used_bytes -= old_size
        self.items[key] = (image, mask, bool(valid), size)
        self.used_bytes += size

    def summary(self) -> str:
        return (
            f"items={len(self.items)} | used={self.used_bytes / 1024**3:.2f} GiB | "
            f"hits={self.hits} | misses={self.misses}"
        )


_CACHE: Optional[TensorLRUCache] = None


def global_tensor_cache(max_gb: Optional[float] = None) -> TensorLRUCache:
    global _CACHE
    if _CACHE is None:
        if max_gb is None:
            max_gb = float(os.environ.get("ONTOMASK_CACHE_MAX_GB", "16"))
        _CACHE = TensorLRUCache(max_gb=max_gb)
        print(f"[cache] enabled | process_limit={float(max_gb):g} GiB")
    return _CACHE


def _draw_region(draw: ImageDraw.ImageDraw, region: Mapping[str, Any]) -> bool:
    shape = region.get("shape_attributes", region)
    name = str(shape.get("name", shape.get("shape_type", "polygon"))).lower()
    if "points" in region and isinstance(region["points"], list):
        points = region["points"]
        if points and isinstance(points[0], Mapping):
            points = [(float(p["x"]), float(p["y"])) for p in points]
        if name in {"rectangle", "rect"} and len(points) >= 2:
            draw.rectangle([tuple(points[0]), tuple(points[1])], fill=255)
            return True
        if name == "circle" and len(points) >= 2:
            cx, cy = float(points[0][0]), float(points[0][1])
            radius = math.dist(points[0], points[1])
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=255)
            return True
        if len(points) >= 3:
            draw.polygon([tuple(point) for point in points], fill=255)
            return True
    if "all_points_x" in shape and "all_points_y" in shape:
        points = list(zip(shape["all_points_x"], shape["all_points_y"]))
        if len(points) >= 3:
            draw.polygon(points, fill=255)
            return True
    if "points" in shape and isinstance(shape["points"], list):
        points = shape["points"]
        if len(points) >= 2:
            if name in {"rectangle", "rect"} and len(points) >= 2:
                draw.rectangle([tuple(points[0]), tuple(points[1])], fill=255)
            else:
                draw.polygon([tuple(p) for p in points], fill=255)
            return True
    if name in {"rect", "rectangle"} and all(k in shape for k in ("x", "y", "width", "height")):
        x, y = float(shape["x"]), float(shape["y"])
        draw.rectangle([x, y, x + float(shape["width"]), y + float(shape["height"])], fill=255)
        return True
    if name == "circle" and all(k in shape for k in ("cx", "cy", "r")):
        cx, cy, radius = float(shape["cx"]), float(shape["cy"]), float(shape["r"])
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=255)
        return True
    return False


def _json_regions(payload: Any, image_name: str) -> List[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    if "shapes" in payload and isinstance(payload["shapes"], list):
        return list(payload["shapes"])
    if "regions" in payload:
        regions = payload["regions"]
        return list(regions.values()) if isinstance(regions, Mapping) else list(regions)
    if "objects" in payload and isinstance(payload["objects"], list):
        regions: List[Mapping[str, Any]] = []
        for obj in payload["objects"]:
            if isinstance(obj, Mapping) and "points" in obj:
                exterior = obj.get("points", {}).get("exterior") if isinstance(obj.get("points"), Mapping) else obj.get("points")
                if exterior:
                    regions.append({"points": exterior})
        return regions
    via = payload.get("_via_img_metadata", payload)
    if isinstance(via, Mapping):
        for entry in via.values():
            if not isinstance(entry, Mapping):
                continue
            filename = str(entry.get("filename", ""))
            if filename and Path(filename).name != Path(image_name).name:
                continue
            if "regions" in entry:
                regions = entry["regions"]
                return list(regions.values()) if isinstance(regions, Mapping) else list(regions)
    if "annotations" in payload and isinstance(payload["annotations"], list):
        regions = []
        for annotation in payload["annotations"]:
            segmentation = annotation.get("segmentation") if isinstance(annotation, Mapping) else None
            if isinstance(segmentation, list):
                for polygon in segmentation:
                    if isinstance(polygon, list) and len(polygon) >= 6:
                        regions.append({"points": list(zip(polygon[0::2], polygon[1::2]))})
        return regions
    return []


def mask_from_annotation(path: Path, image_name: str, width: int, height: int) -> Tuple[Image.Image, bool]:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        with Image.open(path) as raw:
            mask = ImageOps.exif_transpose(raw).convert("L")
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
        mask = mask.point(lambda value: 255 if value > 0 else 0, mode="1").convert("L")
        return mask, bool(mask.getbbox())

    if suffix != ".json":
        return Image.new("L", (width, height), 0), False
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        print(f"[mask][WARN] unreadable JSON {path}: {exc}")
        return Image.new("L", (width, height), 0), False
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    drawn = False
    for region in _json_regions(payload, image_name):
        try:
            drawn = _draw_region(draw, region) or drawn
        except Exception as exc:
            print(f"[mask][WARN] skipped region in {path.name}: {exc}")
    return mask, bool(drawn and mask.getbbox())


@dataclass(frozen=True)
class MaskMatch:
    path: Optional[Path]
    expected: bool


class CystoDSMaskLocator:
    def __init__(
        self,
        raw_root: str | Path,
        configured_roots: Sequence[str | Path] = (),
        metadata_csv: str | Path | None = None,
    ):
        self.raw_root = Path(raw_root)
        roots = [Path(p) for p in configured_roots]
        roots.extend(
            [
                self.raw_root / "masks",
                self.raw_root / "mask",
                self.raw_root / "json",
                self.raw_root / "JSON",
                self.raw_root / "jsons",
                self.raw_root / "annotations",
                self.raw_root / "annotations" / "json",
                self.raw_root / "annotations" / "masks",
                self.raw_root / "labels",
                self.raw_root / "segmentations",
                self.raw_root / "roi_masks",
            ]
        )
        self.roots = list(dict.fromkeys(roots))
        self.expected_by_name: Dict[str, bool] = {}
        metadata_path = Path(metadata_csv) if metadata_csv else self.raw_root / "cystods.csv"
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path, usecols=lambda c: c in {"filename", "json"})
            if {"filename", "json"}.issubset(metadata.columns):
                self.expected_by_name = {
                    Path(str(row.filename)).name: bool(int(row.json))
                    for row in metadata.itertuples(index=False)
                }

    def find(self, image_path: Path, explicit_mask: Any = None) -> MaskMatch:
        image_name = image_path.name
        expected = self.expected_by_name.get(image_name, False)
        if explicit_mask is not None and str(explicit_mask).strip() and str(explicit_mask).lower() != "nan":
            explicit = Path(windows_path_to_wsl(str(explicit_mask)))
            if explicit.exists():
                return MaskMatch(explicit, expected=True)
        stem = image_path.stem
        candidates: List[Path] = []
        for suffix in (".json", ".png", ".jpg", ".tif"):
            if suffix == ".json":
                candidates.append(image_path.with_suffix(suffix))
            for root in self.roots:
                candidates.append(root / f"{stem}{suffix}")
        for candidate in candidates:
            if candidate.exists() and candidate.resolve() != image_path.resolve():
                return MaskMatch(candidate.resolve(), expected=expected or candidate.suffix.lower() == ".json")
        return MaskMatch(None, expected=expected)


class OntoMaskCsvDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str,
        dataset_index: int,
        class_names: Sequence[str],
        class_column: str,
        ontology_class_column: str | None,
        ontology_map: Mapping[str, str],
        image_size: int,
        project_root: str | Path,
        baseline_root: str | Path,
        mask_locator: Optional[CystoDSMaskLocator] = None,
        cache: Optional[TensorLRUCache] = None,
        augment: bool = False,
        augmentation_cfg: Optional[Mapping[str, Any]] = None,
        native_index: Optional[NativeDataIndex] = None,
    ):
        if "path" not in dataframe.columns:
            raise ValueError(f"CSV for {dataset_name} has no 'path' column")
        if class_column not in dataframe.columns:
            raise ValueError(f"CSV for {dataset_name} has no class column {class_column!r}")
        if ontology_class_column and ontology_class_column not in dataframe.columns:
            raise ValueError(
                f"CSV for {dataset_name} has no ontology class column "
                f"{ontology_class_column!r}"
            )
        self.df = dataframe.reset_index(drop=True).copy()
        self.dataset_name = dataset_name
        self.dataset_index = int(dataset_index)
        self.class_names = list(class_names)
        self.class_to_index = {name: i for i, name in enumerate(self.class_names)}
        self.class_column = class_column
        self.ontology_class_column = ontology_class_column
        self.ontology_map = dict(ontology_map)
        self.ontology_to_index = {name: i for i, name in enumerate(ONTOLOGY_CLASSES)}
        self.image_size = int(image_size)
        self.project_root = Path(project_root)
        self.baseline_root = Path(baseline_root)
        self.mask_locator = mask_locator
        self.cache = cache
        self.augment = bool(augment)
        self.augmentation_cfg = dict(augmentation_cfg or {})
        self.native_index = native_index
        self._resolved_paths: Dict[int, Path] = {}
        self._resolved_masks: Dict[Tuple[int, int], Optional[Path]] = {}
        self._shared_epoch_seed = torch.tensor([-1], dtype=torch.int64).share_memory_()

    def set_epoch_seed(self, seed: int) -> None:
        self._shared_epoch_seed[0] = int(seed)

    def _maybe_seed_worker_epoch(self) -> None:
        global _WORKER_EPOCH_TOKEN
        worker = get_worker_info()
        epoch_seed = int(self._shared_epoch_seed[0])
        if worker is None or epoch_seed < 0:
            return
        token = (int(worker.id), epoch_seed)
        
        
        if _WORKER_EPOCH_TOKEN != token:
            seed = epoch_seed + int(worker.id)
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            torch.manual_seed(seed)
            _WORKER_EPOCH_TOKEN = token

    def __getstate__(self):
        state = dict(self.__dict__)
        state["cache"] = None
        return state

    def __len__(self) -> int:
        return len(self.df)

    def _labels(self, row: pd.Series) -> Tuple[str, int, int]:
        raw_fine_name = str(row[self.class_column]).strip()
        fine_name = (
            raw_fine_name
            if raw_fine_name in self.class_to_index
            else canonical_class(self.dataset_name, raw_fine_name)
        )
        if fine_name not in self.class_to_index:
            raise ValueError(f"{fine_name!r} is not in configured classes for {self.dataset_name}")
        raw_ontology_name = str(
            row[
                self.ontology_class_column
                if self.ontology_class_column
                else self.class_column
            ]
        ).strip()
        ontology_source_name = (
            raw_ontology_name
            if raw_ontology_name in self.ontology_map
            else canonical_class(self.dataset_name, raw_ontology_name)
        )
        ontology_name = self.ontology_map[ontology_source_name]
        return fine_name, self.class_to_index[fine_name], self.ontology_to_index[ontology_name]

    def _paths(self, index: int, row: pd.Series) -> Tuple[Path, Optional[Path]]:
        if index not in self._resolved_paths:
            self._resolved_paths[index] = (
                self.native_index.source_path(row["path"], self.project_root, self.baseline_root)
                if self.native_index is not None
                else resolve_existing_path(row["path"], self.project_root, self.baseline_root)
            )
        path = self._resolved_paths[index]
        if self.mask_locator is None:
            return path, None
        mask_key = (index, id(self.mask_locator))
        if mask_key not in self._resolved_masks:
            if self.native_index is not None:
                if str(path) not in self.native_index.annotations:
                    raise ValueError(f"Missing annotation lookup in native cache: {path}")
                raw_mask = self.native_index.annotations[str(path)]
                annotation = Path(raw_mask) if raw_mask is not None else None
            else:
                annotation = self.mask_locator.find(path, explicit_mask=row.get("mask_path", None)).path
            self._resolved_masks[mask_key] = annotation
        return path, self._resolved_masks[mask_key]

    def _load(self, path: Path, annotation_path: Optional[Path]) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        read_path = self.native_index.read_path(path) if self.native_index is not None else path
        with Image.open(read_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        width, height = image.size
        mask = Image.new("L", image.size, 0)
        mask_valid = False
        if annotation_path is not None:
            annotation_read = (
                self.native_index.read_path(annotation_path)
                if self.native_index is not None else annotation_path
            )
            mask, mask_valid = mask_from_annotation(annotation_read, path.name, width, height)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        image_tensor = TF.to_tensor(image)
        mask_tensor = (TF.to_tensor(mask) > 0.5).float()
        if not bool(mask_tensor.any()):
            mask_valid = False
        return image_tensor, mask_tensor, bool(mask_valid)

    def _augment_pair(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.augmentation_cfg
        if torch.rand(()) < float(cfg.get("horizontal_flip_p", 0.5)):
            image, mask = TF.hflip(image), TF.hflip(mask)
        if torch.rand(()) < float(cfg.get("vertical_flip_p", 0.2)):
            image, mask = TF.vflip(image), TF.vflip(mask)
        degrees = float(cfg.get("rotation_degrees", 12.0))
        if degrees > 0:
            angle = float(torch.empty(()).uniform_(-degrees, degrees).item())
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
        for key, adjust in (
            ("brightness", TF.adjust_brightness),
            ("contrast", TF.adjust_contrast),
            ("saturation", TF.adjust_saturation),
        ):
            amount = float(cfg.get(key, 0.0))
            if amount > 0:
                factor = float(torch.empty(()).uniform_(1.0 - amount, 1.0 + amount).item())
                image = adjust(image, factor)
        return image.clamp_(0.0, 1.0), (mask > 0.5).float()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        self._maybe_seed_worker_epoch()
        row = self.df.iloc[int(index)]
        fine_name, fine_label, ontology_label = self._labels(row)
        path, annotation_path = self._paths(int(index), row)
        key = f"{path}|{self.image_size}|{self.dataset_name}|mask={annotation_path}"
        cache = self.cache or global_tensor_cache()
        cached = cache.get(key)
        if cached is None:
            image, mask, mask_valid = self._load(path, annotation_path)
            cache.put(key, image, mask, mask_valid)
        else:
            image, mask, mask_valid = cached
        image, mask = image.clone(), mask.clone()
        if self.augment:
            image, mask = self._augment_pair(image, mask)
        image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)
        return {
            "image": image,
            "fine_label": torch.tensor(fine_label, dtype=torch.long),
            "ontology_label": torch.tensor(ontology_label, dtype=torch.long),
            "dataset_id": torch.tensor(self.dataset_index, dtype=torch.long),
            "mask": mask,
            "mask_valid": torch.tensor(mask_valid, dtype=torch.bool),
            "path": str(path),
            "fine_class": fine_name,
            "dataset_name": self.dataset_name,
            "group_id": str(row.get("group_id", row.get("patient_id", ""))),
        }

    def warm_cache(self) -> Dict[str, int]:
        valid_masks = 0
        expected_missing = 0
        for index in range(len(self)):
            item = self[index]
            valid_masks += int(item["mask_valid"].item())
            if self.mask_locator is not None and not item["mask_valid"].item():
                row = self.df.iloc[index]
                filename = Path(str(row["path"]).replace("\\", "/")).name
                expected_missing += int(self.mask_locator.expected_by_name.get(filename, False))
        return {"images": len(self), "valid_masks": valid_masks, "expected_masks_missing": expected_missing}


def load_fold_frame(
    baseline_root: str | Path,
    dataset_cfg: Mapping[str, Any],
    fold: int,
    split: str,
) -> pd.DataFrame:
    locked_root = Path(str(dataset_cfg.get("locked_fold_root", "")))
    if str(locked_root) not in {"", "."}:
        path = locked_root / f"fold_{int(fold)}" / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Locked split not found: {path}. Run scripts/prepare_v3_data.py first."
            )
        return pd.read_csv(path).reset_index(drop=True)
    root = Path(baseline_root) / str(dataset_cfg["processed_subdir"])
    if str(dataset_cfg.get("split_strategy", "existing_csv")).lower() == "stratified_group_from_index":
        index_path = root / str(dataset_cfg.get("index_file", "image_index_clean.csv"))
        if not index_path.exists():
            raise FileNotFoundError(f"Grouped-split source index not found: {index_path}")
        frame = pd.read_csv(index_path)
        class_column = str(dataset_cfg.get("class_column", "class_name"))
        pattern = re.compile(str(dataset_cfg.get("group_regex", r"case_(\d+)")), re.IGNORECASE)
        groups = []
        for raw_path in frame["path"].astype(str):
            match = pattern.search(raw_path.replace("\\", "/"))
            if not match:
                raise ValueError(f"Could not extract patient group from EBTC path: {raw_path}")
            token = match.group(1)
            groups.append(str(int(token)) if str(token).isdigit() else str(token))
        frame = frame.copy()
        frame["patient_group"] = groups
        labels = frame[class_column].astype(str).to_numpy()
        group_values = frame["patient_group"].astype(str).to_numpy()
        dummy = np.zeros(len(frame), dtype=np.int8)
        seed = int(dataset_cfg.get("split_seed", 42))
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        outer_splits = list(outer.split(dummy, labels, group_values))
        if fold < 1 or fold > len(outer_splits):
            raise ValueError(f"Fold must be 1..{len(outer_splits)}, got {fold}")
        development_indices, test_indices = outer_splits[int(fold) - 1]
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + int(fold))
        inner_train_relative, val_relative = next(
            inner.split(
                dummy[development_indices],
                labels[development_indices],
                group_values[development_indices],
            )
        )
        train_indices = development_indices[inner_train_relative]
        val_indices = development_indices[val_relative]
        split_indices = {"train": train_indices, "val": val_indices, "test": test_indices}
        selected = frame.iloc[split_indices[split]].reset_index(drop=True)
        present = set(selected[class_column].astype(str))
        expected = set(str(value) for value in dataset_cfg["classes"])
        if present != expected:
            raise ValueError(
                f"Grouped {split} fold {fold} is missing classes: {sorted(expected - present)}"
            )
        return selected

    configured = dataset_cfg.get("split_files", {})
    filename = configured.get(split)
    if not filename:
        filename = "train_original.csv" if split == "train" else f"{split}.csv"
    path = root / "folds" / f"fold_{int(fold)}" / str(filename)
    if not path.exists() and split == "train" and str(filename) == "train_original.csv":
        path = path.with_name("train.csv")
    if not path.exists():
        raise FileNotFoundError(f"Fold CSV not found: {path}")
    frame = pd.read_csv(path)
    if "is_augmented" in frame.columns:
        frame = frame[pd.to_numeric(frame["is_augmented"], errors="coerce").fillna(0).astype(int) == 0]
    return frame.reset_index(drop=True)


def build_fold_datasets(
    cfg: Mapping[str, Any],
    fold: int,
    split: str,
    cache: Optional[TensorLRUCache] = None,
) -> Tuple[List[OntoMaskCsvDataset], Dict[str, pd.DataFrame]]:
    project_root = Path(str(cfg["_project_root"]))
    baseline_root = Path(str(cfg["baseline_project_root"]))
    datasets: List[OntoMaskCsvDataset] = []
    frames: Dict[str, pd.DataFrame] = {}
    native_index = native_data_index(cfg.get("cache", {}).get("native_index"))
    for dataset_index, (dataset_name, dataset_cfg) in enumerate(cfg["datasets"].items()):
        if not bool(dataset_cfg.get("enabled", True)):
            continue
        frame = load_fold_frame(baseline_root, dataset_cfg, fold=fold, split=split)
        frames[dataset_name] = frame
        ontology_map = dict(DEFAULT_ONTOLOGY_MAP[dataset_name])
        ontology_map.update(dataset_cfg.get("ontology_map", {}))
        mask_locator = None
        if dataset_name == "cystods":
            raw_root = baseline_root / str(dataset_cfg.get("raw_subdir", "data/raw/CystoDS"))
            configured_roots = [baseline_root / str(p) for p in dataset_cfg.get("mask_subdirs", [])]
            metadata_csv = baseline_root / str(dataset_cfg.get("metadata_csv", "data/raw/CystoDS/cystods.csv"))
            mask_locator = CystoDSMaskLocator(raw_root, configured_roots, metadata_csv)
        datasets.append(
            OntoMaskCsvDataset(
                frame,
                dataset_name=dataset_name,
                dataset_index=dataset_index,
                class_names=dataset_cfg["classes"],
                class_column=str(dataset_cfg.get("class_column", "class_name")),
                ontology_class_column=(
                    str(dataset_cfg["ontology_class_column"])
                    if dataset_cfg.get("ontology_class_column")
                    else None
                ),
                ontology_map=ontology_map,
                image_size=int(cfg["training"].get("image_size", 224)),
                project_root=project_root,
                baseline_root=baseline_root,
                mask_locator=mask_locator,
                cache=cache,
                augment=bool(
                    split == "train"
                    and cfg.get("augmentation", {}).get("enabled", False)
                ),
                augmentation_cfg=cfg.get("augmentation", {}),
                native_index=native_index,
            )
        )
    return datasets, frames


def equal_dataset_sampler(datasets: Sequence[Dataset], seed: int) -> WeightedRandomSampler:
    weights: List[float] = []
    for dataset in datasets:
        if len(dataset) <= 0:
            raise ValueError("Cannot balance an empty dataset")
        weights.extend([1.0 / len(dataset)] * len(dataset))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def ontology_matrix(dataset_cfg: Mapping[str, Any], dataset_name: str) -> torch.Tensor:
    classes = list(dataset_cfg["classes"])
    mapping = dict(DEFAULT_ONTOLOGY_MAP[dataset_name])
    mapping.update(dataset_cfg.get("ontology_map", {}))
    matrix = torch.zeros(len(classes), len(ONTOLOGY_CLASSES), dtype=torch.float32)
    for fine_index, fine_name in enumerate(classes):
        ontology_name = mapping[fine_name]
        matrix[fine_index, ONTOLOGY_CLASSES.index(ontology_name)] = 1.0
    return matrix


def dataset_class_weights(frame: pd.DataFrame, dataset_name: str, class_column: str, classes: Sequence[str]) -> torch.Tensor:
    configured = set(str(value) for value in classes)
    canonical = []
    for value in frame[class_column].tolist():
        raw_value = str(value).strip()
        canonical.append(
            raw_value
            if raw_value in configured
            else canonical_class(dataset_name, raw_value)
        )
    counts = pd.Series(canonical).value_counts()
    values = torch.tensor([float(counts.get(name, 0)) for name in classes], dtype=torch.float32)
    if bool((values <= 0).any()):
        raise ValueError(f"Empty fine class in {dataset_name}: {dict(zip(classes, values.tolist()))}")
    weights = torch.rsqrt(values)
    return weights / weights.mean()


def compact_data_audit(cfg: Mapping[str, Any], fold: int = 1) -> Dict[str, Any]:
    cache = global_tensor_cache(float(cfg.get("cache", {}).get("max_gb", 16)))
    report: Dict[str, Any] = {
        "fold": int(fold),
        "ontology_classes": list(ONTOLOGY_CLASSES),
        "augmentation_enabled": bool(cfg.get("augmentation", {}).get("enabled", False)),
        "datasets": {},
    }
    for split in ("train", "val", "test"):
        datasets, _ = build_fold_datasets(cfg, fold=fold, split=split, cache=cache)
        for dataset in datasets:
            row = report["datasets"].setdefault(dataset.dataset_name, {"splits": {}, "classes": dataset.class_names})
            audit = dataset.warm_cache()
            row["splits"][split] = audit
    report["cache"] = cache.summary()
    return report
