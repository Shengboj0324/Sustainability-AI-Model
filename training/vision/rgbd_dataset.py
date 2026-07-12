"""Manifest-based RGB-D/3D dataset loader for calibrated mobile training data.

The loader is intentionally strict. It validates that every sample has RGB,
depth, camera intrinsics, pose, and 3D labels when requested. It supports local
paths directly and can cache `s3://` R2 objects when boto3 credentials are
available.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from models.vision.depth_geometry import (
    CameraIntrinsics,
    DepthGeometryAnalyzer,
    validate_camera_to_world_transform,
)


@dataclass(frozen=True)
class Box3D:
    label: str
    center_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    yaw_rad: float


@dataclass(frozen=True)
class RGBDManifestSample:
    dataset: str
    split: str
    rgb: str
    depth: str
    intrinsics: CameraIntrinsics
    pose: np.ndarray
    boxes_3d: tuple[Box3D, ...]


class R2ObjectCache:
    """Small explicit cache for S3/R2 objects referenced by manifests."""

    def __init__(self, cache_dir: str | Path, endpoint_url: str | None = None, bucket: str | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.endpoint_url = endpoint_url or os.getenv("R2_ENDPOINT_URL")
        self.default_bucket = bucket or os.getenv("R2_BUCKET")

    def resolve(self, uri: str) -> Path:
        if not uri.startswith("s3://"):
            return Path(uri)
        if not self.endpoint_url:
            raise RuntimeError("R2_ENDPOINT_URL is required to resolve s3:// RGB-D manifest paths")
        if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
            raise RuntimeError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required to cache R2 objects")

        bucket, key = self._parse_s3_uri(uri)
        digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()
        suffix = Path(key).suffix
        target = self.cache_dir / bucket / f"{digest}{suffix}"
        if target.exists():
            return target

        try:
            import boto3
            from botocore.config import Config
        except Exception as exc:  # pragma: no cover - dependency gate
            raise RuntimeError("boto3 is required to cache R2 objects") from exc

        target.parent.mkdir(parents=True, exist_ok=True)
        client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            region_name=os.getenv("AWS_DEFAULT_REGION", "auto"),
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            config=Config(signature_version="s3v4"),
        )
        client.download_file(bucket, key, str(target))
        return target

    def _parse_s3_uri(self, uri: str) -> tuple[str, str]:
        without_scheme = uri[len("s3://") :]
        bucket, _, key = without_scheme.partition("/")
        if not bucket or not key:
            raise ValueError(f"Invalid s3 URI: {uri}")
        if self.default_bucket and bucket != self.default_bucket:
            raise ValueError(f"Unexpected S3 bucket {bucket!r}; expected {self.default_bucket!r}")
        return bucket, key


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    if not rows:
        raise ValueError(f"Manifest has no samples: {path}")
    return rows


def parse_intrinsics(payload: dict[str, Any]) -> CameraIntrinsics:
    required = ("fx", "fy", "cx", "cy", "width", "height")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing intrinsics fields: {missing}")
    return CameraIntrinsics(
        fx=float(payload["fx"]),
        fy=float(payload["fy"]),
        cx=float(payload["cx"]),
        cy=float(payload["cy"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
    )


def parse_box3d(payload: dict[str, Any]) -> Box3D:
    for key in ("label", "center_m", "size_m", "yaw_rad"):
        if key not in payload:
            raise ValueError(f"Missing 3D box field: {key}")
    center = tuple(float(v) for v in payload["center_m"])
    size = tuple(float(v) for v in payload["size_m"])
    if len(center) != 3 or len(size) != 3:
        raise ValueError("3D box center_m and size_m must have length 3")
    if any(v <= 0 or not np.isfinite(v) for v in size):
        raise ValueError("3D box size_m values must be positive and finite")
    yaw = float(payload["yaw_rad"])
    if not np.isfinite(yaw):
        raise ValueError("3D box yaw_rad must be finite")
    return Box3D(label=str(payload["label"]), center_m=center, size_m=size, yaw_rad=yaw)


def parse_manifest_sample(payload: dict[str, Any], require_boxes: bool = True) -> RGBDManifestSample:
    for key in ("dataset", "split", "rgb", "depth", "intrinsics", "pose"):
        if key not in payload:
            raise ValueError(f"Missing RGB-D manifest field: {key}")
    boxes_payload = payload.get("boxes_3d", [])
    if require_boxes and not boxes_payload:
        raise ValueError("RGB-D sample is missing boxes_3d")
    pose = validate_camera_to_world_transform(payload["pose"], "pose")
    return RGBDManifestSample(
        dataset=str(payload["dataset"]),
        split=str(payload["split"]),
        rgb=str(payload["rgb"]),
        depth=str(payload["depth"]),
        intrinsics=parse_intrinsics(payload["intrinsics"]),
        pose=pose,
        boxes_3d=tuple(parse_box3d(box) for box in boxes_payload),
    )


class RGBD3DManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str | None = None,
        cache: R2ObjectCache | None = None,
        transform: Callable[[Image.Image], Any] | None = None,
        require_boxes: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.cache = cache
        self.transform = transform
        samples = [parse_manifest_sample(row, require_boxes=require_boxes) for row in _read_jsonl(self.manifest_path)]
        if split is not None:
            samples = [sample for sample in samples if sample.split == split]
        if not samples:
            raise ValueError(f"No RGB-D samples available for split={split!r}")
        self.samples = samples
        self.depth_analyzer = DepthGeometryAnalyzer()

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve(self, uri: str) -> Path:
        if uri.startswith("s3://"):
            if self.cache is None:
                raise RuntimeError("R2ObjectCache is required for s3:// RGB-D manifest paths")
            return self.cache.resolve(uri)
        return Path(uri)

    def _load_rgb(self, uri: str) -> Image.Image:
        path = self._resolve(uri)
        if not path.exists():
            raise FileNotFoundError(f"RGB file not found: {path}")
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")

    def _load_depth(self, uri: str) -> np.ndarray:
        path = self._resolve(uri)
        if not path.exists():
            raise FileNotFoundError(f"Depth file not found: {path}")
        if path.suffix.lower() == ".npy":
            depth = np.load(path, allow_pickle=False).astype("float32")
        else:
            with Image.open(path) as image:
                depth = np.asarray(image).astype("float32") / 1000.0
        if depth.ndim != 2:
            raise ValueError(f"Depth must be a 2D array, got {depth.shape} from {path}")
        if not np.isfinite(depth).any() or not (depth > 0).any():
            raise ValueError(f"Depth has no positive finite pixels: {path}")
        return depth

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        rgb = self._load_rgb(sample.rgb)
        depth = self._load_depth(sample.depth)
        self.depth_analyzer.validate_intrinsics(sample.intrinsics, depth)
        image_tensor_or_pil = self.transform(rgb) if self.transform else rgb
        boxes = [
            {
                "label": box.label,
                "center_m": torch.tensor(box.center_m, dtype=torch.float32),
                "size_m": torch.tensor(box.size_m, dtype=torch.float32),
                "yaw_rad": torch.tensor(box.yaw_rad, dtype=torch.float32),
            }
            for box in sample.boxes_3d
        ]
        return {
            "dataset": sample.dataset,
            "split": sample.split,
            "rgb": image_tensor_or_pil,
            "depth_m": torch.from_numpy(depth.astype("float32")),
            "intrinsics": sample.intrinsics,
            "pose": torch.from_numpy(sample.pose.astype("float32")),
            "boxes_3d": boxes,
        }


def rgbd_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset": [item["dataset"] for item in batch],
        "split": [item["split"] for item in batch],
        "rgb": [item["rgb"] for item in batch],
        "depth_m": [item["depth_m"] for item in batch],
        "intrinsics": [item["intrinsics"] for item in batch],
        "pose": torch.stack([item["pose"] for item in batch]),
        "boxes_3d": [item["boxes_3d"] for item in batch],
    }
