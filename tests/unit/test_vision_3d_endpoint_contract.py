import base64
import io
from pathlib import Path
import sys

import numpy as np
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.vision_service.server_v2 import app


def _depth_payload(width: int = 5, height: int = 4):
    depth = np.full((height, width), 1.0, dtype="float32")
    buffer = io.BytesIO()
    np.save(buffer, depth)
    return {
        "depth_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "depth_format": "npy",
        "depth_unit_scale": 1.0,
        "intrinsics": {
            "fx": 200.0,
            "fy": 200.0,
            "cx": 2.0,
            "cy": 1.5,
            "width": width,
            "height": height,
        },
    }


def _flow_payload():
    return {
        "points_t": [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]],
        "points_future": [[-2.0, 0.0, 2.0], [-3.0, 1.0, 3.0]],
        "camera_to_world_t": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "camera_to_world_future": [
            [1.0, 0.0, 0.0, 3.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "movement_threshold_m": 0.03,
    }


def test_analyze_3d_endpoint_returns_geometry_not_fake_classifier():
    client = TestClient(app)

    response = client.post("/analyze-3d", json=_depth_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["capability"] == "depth_geometry_analysis"
    assert payload["model_available"] is False
    assert payload["point_count"] == 20
    assert payload["metadata"]["classification_status"] == "not_available_without_trained_3d_model"


def test_analyze_3d_endpoint_rejects_bad_intrinsics():
    client = TestClient(app)
    payload = _depth_payload()
    payload["intrinsics"]["width"] = 6

    response = client.post("/analyze-3d", json=payload)

    assert response.status_code == 400
    assert "do not match" in response.json()["detail"]


def test_analyze_3d_flow_endpoint_compensates_camera_motion():
    client = TestClient(app)

    response = client.post("/analyze-3d-flow", json=_flow_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["capability"] == "camera_stabilized_3d_flow"
    assert payload["model_available"] is False
    assert payload["point_count"] == 2
    assert payload["mean_flow_m"] == 0.0
    assert payload["moving_point_ratio"] == 0.0
    assert payload["metadata"]["classification_status"] == "not_available_without_trained_3d_model"


def test_analyze_3d_flow_endpoint_rejects_non_rigid_pose():
    client = TestClient(app)
    payload = _flow_payload()
    payload["camera_to_world_future"][0][0] = 2.0

    response = client.post("/analyze-3d-flow", json=payload)

    assert response.status_code == 400
    assert "orthonormal" in response.json()["detail"]


def test_analyze_3d_flow_endpoint_rejects_mismatched_correspondences():
    client = TestClient(app)
    payload = _flow_payload()
    payload["points_future"] = payload["points_future"][:1]

    response = client.post("/analyze-3d-flow", json=payload)

    assert response.status_code == 400
    assert "matching shapes" in response.json()["detail"]
