import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.vision.depth_geometry import (
    CameraIntrinsics,
    DepthGeometryAnalyzer,
    camera_stabilized_flow,
    inverse_rigid_transform,
    validate_camera_to_world_transform,
)


def _depth_npy_b64(depth: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, depth)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_depth_geometry_analyzer_projects_valid_depth_map():
    depth = np.full((4, 5), 1.25, dtype="float32")
    analyzer = DepthGeometryAnalyzer()

    result = analyzer.analyze(
        depth_b64=_depth_npy_b64(depth),
        depth_format="npy",
        depth_unit_scale=1.0,
        intrinsics=CameraIntrinsics(fx=200.0, fy=200.0, cx=2.0, cy=1.5, width=5, height=4),
    )

    assert result.capability == "depth_geometry_analysis"
    assert result.model_available is False
    assert result.width == 5
    assert result.height == 4
    assert result.valid_pixel_ratio == 1.0
    assert result.point_count == 20
    assert result.depth_mean_m == 1.25
    assert result.metadata["classification_status"] == "not_available_without_trained_3d_model"


def test_depth_geometry_rejects_intrinsic_dimension_mismatch():
    depth = np.ones((4, 5), dtype="float32")
    analyzer = DepthGeometryAnalyzer()

    with pytest.raises(ValueError, match="do not match"):
        analyzer.analyze(
            depth_b64=_depth_npy_b64(depth),
            depth_format="npy",
            depth_unit_scale=1.0,
            intrinsics=CameraIntrinsics(fx=200.0, fy=200.0, cx=2.0, cy=1.5, width=6, height=4),
        )


def test_depth_geometry_rejects_multi_channel_depth():
    depth = np.ones((4, 5, 3), dtype="float32")
    analyzer = DepthGeometryAnalyzer()

    with pytest.raises(ValueError, match="single-channel"):
        analyzer.decode_depth_b64(_depth_npy_b64(depth), "npy", 1.0)


def test_depth_to_points_matches_pinhole_camera_equations():
    depth = np.array([[2.0, 2.0], [2.0, 2.0]], dtype="float32")
    analyzer = DepthGeometryAnalyzer()

    points = analyzer.depth_to_points(
        depth,
        CameraIntrinsics(fx=2.0, fy=2.0, cx=0.0, cy=0.0, width=2, height=2),
    )

    expected = np.array(
        [
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 2.0],
        ],
        dtype="float32",
    )
    np.testing.assert_allclose(points, expected, rtol=1e-6, atol=1e-6)


def test_depth_geometry_rejects_intrinsics_on_exclusive_upper_boundary():
    depth = np.ones((4, 5), dtype="float32")
    analyzer = DepthGeometryAnalyzer()

    with pytest.raises(ValueError, match="cx"):
        analyzer.depth_to_points(
            depth,
            CameraIntrinsics(fx=200.0, fy=200.0, cx=5.0, cy=1.5, width=5, height=4),
        )

    with pytest.raises(ValueError, match="cy"):
        analyzer.depth_to_points(
            depth,
            CameraIntrinsics(fx=200.0, fy=200.0, cx=2.0, cy=4.0, width=5, height=4),
        )


def test_depth_geometry_rejects_non_positive_max_points():
    analyzer = DepthGeometryAnalyzer()

    with pytest.raises(ValueError, match="max_points"):
        analyzer.depth_to_points(
            np.ones((4, 5), dtype="float32"),
            CameraIntrinsics(fx=200.0, fy=200.0, cx=2.0, cy=1.5, width=5, height=4),
            max_points=0,
        )


def test_camera_stabilized_flow_zero_for_static_world_under_camera_translation():
    pose_t = np.eye(4, dtype="float64")
    pose_future = np.eye(4, dtype="float64")
    pose_future[0, 3] = 3.0
    points_t = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]], dtype="float64")
    points_future = np.array([[-2.0, 0.0, 2.0], [-3.0, 1.0, 3.0]], dtype="float64")

    flow = camera_stabilized_flow(points_t, points_future, pose_t, pose_future)

    np.testing.assert_allclose(flow, np.zeros_like(points_t), rtol=1e-9, atol=1e-9)


def test_camera_stabilized_flow_reports_true_object_motion():
    pose = np.eye(4, dtype="float64")
    points_t = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]], dtype="float64")
    points_future = points_t + np.array([[0.1, 0.0, 0.0]], dtype="float64")

    flow = camera_stabilized_flow(points_t, points_future, pose, pose)

    np.testing.assert_allclose(flow, np.array([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]), rtol=1e-9, atol=1e-9)


def test_camera_stabilized_flow_rejects_non_rigid_pose():
    invalid_pose = np.eye(4, dtype="float64")
    invalid_pose[0, 0] = 2.0

    with pytest.raises(ValueError, match="orthonormal"):
        validate_camera_to_world_transform(invalid_pose, "bad_pose")


def test_inverse_rigid_transform_round_trip_is_identity():
    pose = np.eye(4, dtype="float64")
    pose[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype="float64",
    )
    pose[:3, 3] = np.array([1.5, -0.25, 2.0], dtype="float64")

    inverse = inverse_rigid_transform(pose)

    np.testing.assert_allclose(inverse @ pose, np.eye(4), rtol=1e-9, atol=1e-9)


def test_analyze_camera_stabilized_flow_returns_structured_metrics():
    analyzer = DepthGeometryAnalyzer()
    pose = np.eye(4, dtype="float64")
    points_t = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [1.0, 1.0, 2.0]], dtype="float64")
    points_future = points_t + np.array([[0.04, 0.0, 0.0]], dtype="float64")

    result = analyzer.analyze_camera_stabilized_flow(points_t, points_future, pose, pose, movement_threshold_m=0.03)

    assert result.capability == "camera_stabilized_3d_flow"
    assert result.model_available is False
    assert result.point_count == 4
    assert result.mean_flow_m == 0.04
    assert result.moving_point_ratio == 1.0
    assert result.metadata["classification_status"] == "not_available_without_trained_3d_model"


def test_camera_stabilized_flow_randomized_static_world_invariant():
    rng = np.random.default_rng(42)

    for _ in range(100):
        angle = float(rng.uniform(-np.pi, np.pi))
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_future = np.array(
            [
                [cos_a, -sin_a, 0.0],
                [sin_a, cos_a, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype="float64",
        )
        pose_t = np.eye(4, dtype="float64")
        pose_future = np.eye(4, dtype="float64")
        pose_future[:3, :3] = rotation_future
        pose_future[:3, 3] = rng.uniform(-2.0, 2.0, size=3)

        world_points = rng.uniform(-1.0, 1.0, size=(16, 3))
        world_points[:, 2] += 3.0
        points_t = world_points.copy()
        points_future = (world_points - pose_future[:3, 3]) @ rotation_future

        flow = camera_stabilized_flow(points_t, points_future, pose_t, pose_future)

        np.testing.assert_allclose(flow, np.zeros_like(flow), rtol=1e-8, atol=1e-8)
