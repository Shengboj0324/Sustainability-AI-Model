"""Strict RGB-D/depth geometry analysis for mobile/web 3D capture.

This module does not pretend to classify waste from 3D data. It implements the
real, testable substrate needed before a learned 3D model can be trained:
validated depth-map ingestion, camera intrinsics checks, point-cloud projection,
and geometric quality metrics.
"""

from __future__ import annotations

import base64
import io
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class DepthGeometryResult:
    capability: str
    model_available: bool
    depth_format: str
    width: int
    height: int
    valid_pixel_ratio: float
    depth_min_m: float
    depth_max_m: float
    depth_mean_m: float
    point_count: int
    centroid_m: list[float]
    extent_m: list[float]
    surface_roughness_m: float
    confidence: float
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DepthGeometryAnalyzer:
    """Validated depth-map geometry analyzer.

    Supported input formats:
    - base64 encoded `.npy` float/uint depth array;
    - base64 encoded 16-bit PNG/TIFF depth map.

    Units are explicit. PNG/TIFF depth values default to millimeters.
    """

    def decode_depth_b64(
        self,
        depth_b64: str,
        depth_format: str,
        depth_unit_scale: float,
    ) -> np.ndarray:
        if not depth_b64:
            raise ValueError("depth_b64 is required")
        if depth_unit_scale <= 0:
            raise ValueError("depth_unit_scale must be positive")

        raw = base64.b64decode(depth_b64, validate=True)
        normalized_format = depth_format.lower().strip()

        if normalized_format == "npy":
            arr = np.load(io.BytesIO(raw), allow_pickle=False)
        elif normalized_format in {"png16", "png", "tiff", "tif"}:
            with Image.open(io.BytesIO(raw)) as image:
                arr = np.asarray(image)
        else:
            raise ValueError(f"Unsupported depth_format {depth_format!r}; expected npy, png16, png, tiff, or tif")

        if arr.ndim == 3:
            if arr.shape[2] != 1:
                raise ValueError(f"Depth map must be single-channel, got shape {arr.shape}")
            arr = arr[:, :, 0]
        if arr.ndim != 2:
            raise ValueError(f"Depth map must be 2D, got shape {arr.shape}")
        if min(arr.shape) < 2:
            raise ValueError(f"Depth map is too small for geometry analysis: {arr.shape}")

        arr = arr.astype("float32") * float(depth_unit_scale)
        if not np.isfinite(arr).any():
            raise ValueError("Depth map contains no finite values")
        return arr

    def validate_intrinsics(self, intrinsics: CameraIntrinsics, depth: np.ndarray) -> None:
        height, width = depth.shape
        if intrinsics.width != width or intrinsics.height != height:
            raise ValueError(
                f"Intrinsics dimensions {intrinsics.width}x{intrinsics.height} do not match depth map {width}x{height}"
            )
        for name in ("fx", "fy"):
            value = getattr(intrinsics, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Camera intrinsic {name} must be positive")
        if not (0 <= intrinsics.cx < width):
            raise ValueError("Camera intrinsic cx must lie within image width")
        if not (0 <= intrinsics.cy < height):
            raise ValueError("Camera intrinsic cy must lie within image height")

    def depth_to_points(
        self,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
        max_points: int = 20000,
    ) -> np.ndarray:
        if max_points <= 0:
            raise ValueError("max_points must be positive")
        self.validate_intrinsics(intrinsics, depth)
        valid = np.isfinite(depth) & (depth > 0)
        ys, xs = np.nonzero(valid)
        if len(xs) == 0:
            raise ValueError("Depth map has no positive finite depth pixels")

        if len(xs) > max_points:
            step = int(np.ceil(len(xs) / max_points))
            xs = xs[::step]
            ys = ys[::step]

        z = depth[ys, xs]
        x = (xs.astype("float32") - intrinsics.cx) * z / intrinsics.fx
        y = (ys.astype("float32") - intrinsics.cy) * z / intrinsics.fy
        return np.stack([x, y, z], axis=1)

    def analyze(
        self,
        depth_b64: str,
        depth_format: str,
        depth_unit_scale: float,
        intrinsics: CameraIntrinsics,
    ) -> DepthGeometryResult:
        depth = self.decode_depth_b64(depth_b64, depth_format, depth_unit_scale)
        points = self.depth_to_points(depth, intrinsics)

        valid = np.isfinite(depth) & (depth > 0)
        valid_depth = depth[valid]
        centroid = points.mean(axis=0)
        extent = points.max(axis=0) - points.min(axis=0)
        roughness = float(np.std(valid_depth - np.median(valid_depth)))
        valid_ratio = float(valid.sum() / depth.size)

        warnings = []
        if valid_ratio < 0.25:
            warnings.append("LOW_DEPTH_COVERAGE")
        if roughness > 0.25:
            warnings.append("HIGH_SURFACE_ROUGHNESS_OR_MULTIPLE_DEPTH_LAYERS")
        if valid_depth.max() > 8.0:
            warnings.append("DEPTH_RANGE_EXCEEDS_TYPICAL_MOBILE_LIDAR")

        confidence = min(1.0, max(0.0, valid_ratio * (1.0 if not warnings else 0.75)))
        return DepthGeometryResult(
            capability="depth_geometry_analysis",
            model_available=False,
            depth_format=depth_format,
            width=int(depth.shape[1]),
            height=int(depth.shape[0]),
            valid_pixel_ratio=round(valid_ratio, 6),
            depth_min_m=round(float(valid_depth.min()), 6),
            depth_max_m=round(float(valid_depth.max()), 6),
            depth_mean_m=round(float(valid_depth.mean()), 6),
            point_count=int(points.shape[0]),
            centroid_m=[round(float(v), 6) for v in centroid],
            extent_m=[round(float(v), 6) for v in extent],
            surface_roughness_m=round(roughness, 6),
            confidence=round(confidence, 6),
            warnings=warnings,
            metadata={
                "intrinsics": asdict(intrinsics),
                "classification_model": "not_configured",
                "classification_status": "not_available_without_trained_3d_model",
            },
        )
