# 3D Dataset Research And R2 Plan

Date: 2026-07-12

## Decision

The best immediate primary dataset for ReleAF's mobile 3D vision path is ARKitScenes 3DOD. It is the closest match to the target website/iOS deployment because it is captured with Apple LiDAR-style mobile RGB-D data and includes camera poses, intrinsics, depth, and 3D oriented object boxes.

Objectron is the best complementary object-centric mobile AR dataset for everyday object 3D boxes and camera-pose robustness. It overlaps with ReleAF classes such as bottle, cup, book, cereal box, shoe, laptop, and chair, but it does not provide disposal labels.

SUN RGB-D is a compact benchmark baseline. ScanNet and ScanNet++ are strong reference datasets, but they are gated by terms/application/token access and should not be mirrored without explicit approval.

## Source Review

### ARKitScenes

Primary source: https://github.com/apple/ARKitScenes

Relevant properties:

- Mobile RGB-D / Apple LiDAR alignment.
- 5,047 captures of 1,661 unique scenes.
- Raw data includes camera pose and surface reconstruction.
- 3DOD provides low-resolution RGB, low-resolution depth, and labels.
- Documented 3DOD size: 623.4 GB for 5,047 scans.
- Data formats include PNG RGB/depth/confidence, `.pincam` intrinsics, JSON annotations, `.traj` camera pose files, `.ply` mesh/point cloud files, and `.mov` raw video.

Risk:

- Apple license terms must be honored. Do not mirror to R2 for redistribution unless the intended use fits the license and required notices/terms are preserved.

### Objectron

Primary source: https://github.com/google-research-datasets/Objectron

Relevant properties:

- 15K annotated object-centric videos and 4M annotated images.
- Includes high-resolution images, object poses, camera poses, sparse point clouds, and surface planes.
- Manual 3D bounding boxes.
- Raw size is reported as 1.9 TB; total size with records/sequences is 4.4 TB.
- Public HTTP access is available through `https://storage.googleapis.com/objectron`.
- Relevant classes for ReleAF: `bottle`, `cup`, `book`, `cereal_box`, `shoe`, `laptop`, `chair`, `camera`.

Risk:

- Released under C-UDA 1.0. Redistribution requires attribution and binding downstream recipients to the same terms.
- Not disposal-labelled. It should support 3D/mobile robustness, not final sustainability classification by itself.

### SUN RGB-D

Primary source: https://rgbd.cs.princeton.edu/

Relevant properties:

- 10,335 RGB-D images.
- Dense annotations, 2D polygons, 3D bounding boxes, object orientations, room layouts.
- Good compact benchmark for RGB-D 3D object detection validation.

Risk:

- Older indoor sensor mix; less aligned with iOS LiDAR deployment than ARKitScenes.
- Some annotations require MATLAB-oriented tooling in the original release.

### ScanNet / ScanNet++

Primary sources:

- https://www.scan-net.org/
- https://scannetpp.mlsg.cit.tum.de/scannetpp/

Relevant properties:

- ScanNet: RGB-D video with camera poses, reconstructions, semantic annotations.
- ScanNet++: high-fidelity laser scans, registered DSLR images, iPhone RGB-D streams, long-tail semantic labels.

Risk:

- Access is gated by terms/application/token approval.
- Do not mirror these datasets until access and redistribution/storage terms are explicitly approved.

## R2 Mirroring Position

The bucket information alone is not enough to upload data. The mirror script requires:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- optional `AWS_SESSION_TOKEN`

The script intentionally fails closed if credentials are missing. This prevents false claims that data was uploaded.

## Recommended Initial R2 Layout

```text
s3://sustainabilityaibucket/external/3d_vision/
  arkitscenes/
    3dod/
    raw/
  objectron/
    v1/index/
    annotations/
    videos/
    v1/records_shuffled/
    v1/sequences/
  sunrgbd/
  manifests/
```

## Training Integration Contract

The local code now supports a strict manifest-based RGB-D sample contract:

```json
{
  "dataset": "arkitscenes_3dod",
  "split": "train",
  "rgb": "s3://sustainabilityaibucket/external/3d_vision/arkitscenes/3dod/.../rgb.png",
  "depth": "s3://sustainabilityaibucket/external/3d_vision/arkitscenes/3dod/.../depth.png",
  "intrinsics": {"fx": 450.0, "fy": 450.0, "cx": 256.0, "cy": 192.0, "width": 512, "height": 384},
  "pose": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
  "boxes_3d": [{"label": "bottle", "center_m": [0, 0, 1], "size_m": [0.1, 0.1, 0.3], "yaw_rad": 0.0}]
}
```

Training code must not silently convert this to 2D-only training. If depth, intrinsics, pose, or 3D boxes are missing for a sample, the validator reports it.
