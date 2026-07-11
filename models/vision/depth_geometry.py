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


@dataclass
class CameraStabilizedFlowResult:
    capability: str
    model_available: bool
    point_count: int
    mean_flow_m: float
    median_flow_m: float
    max_flow_m: float
    moving_point_ratio: float
    centroid_flow_m: list[float]
    confidence: float
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_float_array(name: str, value: Any, expected_last_dim: int) -> np.ndarray:
    arr = np.asarray(value, dtype="float64")
    if arr.ndim < 1 or arr.shape[-1] != expected_last_dim:
        raise ValueError(f"{name} must have last dimension {expected_last_dim}, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr


def validate_camera_to_world_transform(transform: Any, name: str = "camera_to_world") -> np.ndarray:
    """Validate a 4x4 rigid camera-to-world transform.

    The transform is treated as an SE(3) pose: the top-left 3x3 block must be a
    proper rotation matrix and the bottom row must be homogeneous coordinates.
    """
    matrix = np.asarray(transform, dtype="float64")
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), rtol=1e-6, atol=1e-6):
        raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 0, 1]")

    rotation = matrix[:3, :3]
    should_be_identity = rotation.T @ rotation
    determinant = float(np.linalg.det(rotation))
    if not np.allclose(should_be_identity, np.eye(3), rtol=1e-5, atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(determinant, 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")
    return matrix


def transform_points(transform: Any, points: Any) -> np.ndarray:
    """Apply a rigid 4x4 transform to an array of 3D points."""
    matrix = validate_camera_to_world_transform(transform, "transform")
    pts = _as_float_array("points", points, 3)
    return pts @ matrix[:3, :3].T + matrix[:3, 3]


def inverse_rigid_transform(transform: Any) -> np.ndarray:
    """Invert a validated SE(3) transform without a general matrix inverse."""
    matrix = validate_camera_to_world_transform(transform, "transform")
    inverse = np.eye(4, dtype="float64")
    rotation_t = matrix[:3, :3].T
    inverse[:3, :3] = rotation_t
    inverse[:3, 3] = -rotation_t @ matrix[:3, 3]
    return inverse


def camera_stabilized_flow(
    points_t: Any,
    points_future: Any,
    camera_to_world_t: Any,
    camera_to_world_future: Any,
) -> np.ndarray:
    """Compute ego-motion-compensated 3D point displacement.

    `points_t` and `points_future` are corresponding 3D points in their own
    camera frames. The future points are transformed into the current camera
    frame before subtraction, so static world points have near-zero flow even
    when the camera moves.
    """
    current_points = _as_float_array("points_t", points_t, 3)
    future_points = _as_float_array("points_future", points_future, 3)
    if current_points.shape != future_points.shape:
        raise ValueError(f"points_t and points_future must have matching shapes, got {current_points.shape} and {future_points.shape}")
    if current_points.ndim != 2:
        raise ValueError(f"points_t and points_future must have shape (N, 3), got {current_points.shape}")
    if current_points.shape[0] == 0:
        raise ValueError("points_t and points_future must contain at least one point")

    pose_t = validate_camera_to_world_transform(camera_to_world_t, "camera_to_world_t")
    pose_future = validate_camera_to_world_transform(camera_to_world_future, "camera_to_world_future")
    future_world = transform_points(pose_future, future_points)
    future_in_current_camera = transform_points(inverse_rigid_transform(pose_t), future_world)
    return future_in_current_camera - current_points


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

    def analyze_camera_stabilized_flow(
        self,
        points_t: Any,
        points_future: Any,
        camera_to_world_t: Any,
        camera_to_world_future: Any,
        movement_threshold_m: float = 0.03,
    ) -> CameraStabilizedFlowResult:
        if movement_threshold_m <= 0 or not np.isfinite(movement_threshold_m):
            raise ValueError("movement_threshold_m must be positive and finite")

        flow = camera_stabilized_flow(
            points_t=points_t,
            points_future=points_future,
            camera_to_world_t=camera_to_world_t,
            camera_to_world_future=camera_to_world_future,
        )
        magnitudes = np.linalg.norm(flow, axis=1)
        centroid_flow = flow.mean(axis=0)
        moving_ratio = float(np.mean(magnitudes > movement_threshold_m))

        warnings = []
        if flow.shape[0] < 8:
            warnings.append("LOW_TRACKED_POINT_COUNT")
        if moving_ratio > 0.75:
            warnings.append("HIGH_SCENE_OR_OBJECT_MOTION_AFTER_CAMERA_STABILIZATION")
        if float(magnitudes.max()) > 2.0:
            warnings.append("LARGE_STABILIZED_FLOW_CHECK_POINT_CORRESPONDENCE_OR_POSES")

        confidence = min(1.0, max(0.0, (flow.shape[0] / 32.0) * (1.0 if not warnings else 0.75)))
        return CameraStabilizedFlowResult(
            capability="camera_stabilized_3d_flow",
            model_available=False,
            point_count=int(flow.shape[0]),
            mean_flow_m=round(float(magnitudes.mean()), 6),
            median_flow_m=round(float(np.median(magnitudes)), 6),
            max_flow_m=round(float(magnitudes.max()), 6),
            moving_point_ratio=round(moving_ratio, 6),
            centroid_flow_m=[round(float(v), 6) for v in centroid_flow],
            confidence=round(confidence, 6),
            warnings=warnings,
            metadata={
                "movement_threshold_m": float(movement_threshold_m),
                "source": "egowam_sample_camera_stabilized_flow_port",
                "classification_model": "not_configured",
                "classification_status": "not_available_without_trained_3d_model",
                "contract": "corresponding_points_and_calibrated_camera_to_world_poses_required",
            },
        )
