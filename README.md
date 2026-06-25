# RGB-D Object Pose Estimation with FoundationPose 6D

Intel RealSense D455와 YOLOv8 인스턴스 세그멘테이션으로 산업용 부품 3종(`cross`, `cylinder`, `hole`)을 검출하고, RGB-D + CAD mesh + mask를 FoundationPose에 입력해 카메라 좌표계 기준 **6D pose(position + orientation)**를 추정하는 프로젝트입니다.

기존 `detect_3d_pose.py`는 빠른 2.5D baseline입니다. 새로 추가된 `foundation_pose_node.py`는 FoundationPose를 사용해 CAD 정합 기반 6D pose를 ROS2 토픽으로 발행합니다.

---

## Highlights

- **YOLOv8n-seg instance segmentation**: 평균 Mask mAP50 **0.9290**
- **RealSense D455 RGB-D alignment**: `rs.align`으로 color/depth 픽셀 정합
- **2.5D baseline**: mask 중심점 + depth median + `minAreaRect` 방향각
- **FoundationPose 6D extension**: CAD mesh + RGB-D + instance mask 기반 4x4 pose matrix 추정
- **ROS2 output**: JSON `/object_poses`, `geometry_msgs/PoseStamped` `/object_pose_stamped`
- **Fail-soft design**: FoundationPose 실패 시 depth PCA fallback으로 디버깅용 pose 유지

---

## System Overview

### Hardware

| Item | Spec |
|------|------|
| RGB-D Camera | Intel RealSense D455 |
| GPU | NVIDIA GeForce RTX 4060 Laptop |
| Target Objects | cylinder, hole, cross |

### Software

| Layer | Stack |
|------|-------|
| OS | Ubuntu 22.04 |
| Language | Python 3.10 |
| Detection | YOLOv8n-seg, Ultralytics |
| 6D Pose | FoundationPose, CAD mesh, RGB-D |
| Camera | Intel RealSense SDK / `pyrealsense2` |
| Middleware | ROS2 Humble |
| Visualization | OpenCV |

---

## Pipeline

```text
RealSense D455
  -> aligned RGB-D frame
  -> YOLOv8 instance segmentation
  -> binary object mask
  -> nearest valid target selection
  -> FoundationPose register/track with CAD mesh
  -> 4x4 pose matrix in camera_color_optical_frame
  -> JSON + PoseStamped publish
```

### 2.5D Baseline (`detect_3d_pose.py`)

```text
YOLO mask -> mask centroid -> median depth -> camera back-projection
          -> minAreaRect orientation angle
```

```text
X = (cx - ppx) * depth_m / fx
Y = (cy - ppy) * depth_m / fy
Z = depth_m
```

This baseline is lightweight and useful for object picking experiments, but orientation is estimated from a 2D mask angle. It does not perform CAD-to-image 6D alignment.

### FoundationPose 6D (`foundation_pose_node.py`)

```text
YOLO mask + RGB + depth + camera K + CAD mesh
  -> FoundationPose.register() on first observation
  -> FoundationPose.track_one() on subsequent frames
  -> translation [m] + quaternion [x,y,z,w] + 4x4 pose matrix
```

The node publishes the nearest valid target only. This matches a practical bin-picking flow where the robot first handles the closest reachable part.

---

## When Does Real 6D Pose Come Out?

`foundation_pose_node.py` publishes a message every frame when a valid target pose exists. The JSON field `is_cad_aligned_6d` tells whether the output is true FoundationPose 6D or a fallback estimate.

| Condition | Output | `pose_source` | `is_cad_aligned_6d` |
|-----------|--------|---------------|---------------------|
| YOLO mask detected, valid depth exists, CAD mesh exists, FoundationPose imports, CUDA/nvdiffrast works, register/track succeeds | CAD-aligned 6D pose | `foundationpose` | `true` |
| FoundationPose is missing or fails, but mask depth has enough points | approximate centroid + PCA orientation | `depth_pca_fallback` | `false` |
| No mask, no valid depth, or too few depth points | no target pose | `none` or `target: null` | `false` |

The fallback is intentionally kept for debugging and system continuity. For robot grasping, use outputs where:

```json
"is_cad_aligned_6d": true
```

---

## ROS2 Output

### Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/object_poses` | `std_msgs/String` | JSON payload with class, confidence, pose source, position, quaternion, pose matrix |
| `/object_pose_stamped` | `geometry_msgs/PoseStamped` | PoseStamped for the selected target |

### JSON Example

```json
{
  "target": {
    "class": "cylinder",
    "confidence": 0.921,
    "pose_source": "foundationpose",
    "is_cad_aligned_6d": true,
    "frame_id": "camera_color_optical_frame",
    "position": {
      "x": 0.0312,
      "y": -0.0124,
      "z": 0.4231
    },
    "orientation": {
      "x": 0.012341,
      "y": -0.004221,
      "z": 0.701884,
      "w": 0.712171
    },
    "pose_matrix": [[...], [...], [...], [...]],
    "priority": {
      "selected": true,
      "reason": "nearest_valid_mask_depth",
      "depth_median_m": 0.423,
      "detected_count": 2
    }
  },
  "detected_count": 2
}
```

---

## Performance

The repo contains measured YOLO segmentation performance and training curves. 6D pose accuracy is not reported as a numeric ADD/ADD-S score here because the dataset does not include motion-capture or robot-calibrated 6D ground truth. The FoundationPose path is implemented and observable through `pose_source=foundationpose`; quantitative 6D pose benchmarking should be added with calibrated GT poses.

### Segmentation Metrics

| Class | mAP50 Box | mAP50 Mask | Precision | Recall |
|-------|:---------:|:----------:|:---------:|:------:|
| cylinder | 0.9950 | 0.9950 | 0.9839 | 1.0000 |
| hole | 0.8473 | 0.8394 | 0.9458 | 0.8571 |
| cross | 0.9527 | 0.9527 | 0.9062 | 0.9666 |
| **mean** | **0.9317** | **0.9290** | **0.9453** | **0.9412** |

### Training Summary

![Training summary table](training_summary_table.png)

### Best Metrics

![Best training metrics](training_best_metrics.png)

### Training Curves

![Training curves](training_curves.png)

### Validation Samples

| Labels | Predictions |
|--------|-------------|
| ![Validation labels](runs/segment/train/val_batch0_labels.jpg) | ![Validation predictions](runs/segment/train/val_batch0_pred.jpg) |

### Notes on Runtime

- `detect_3d_pose.py` is the fastest path and is suitable for quick RGB-D validation.
- `foundation_pose_node.py` is heavier because FoundationPose performs neural scoring/refinement and CUDA rasterization.
- First observation of a class uses `register()` and is slower; subsequent frames use `track_one()`.
- Runtime depends on GPU, CUDA/PyTorch/nvdiffrast versions, mesh size, `FP_REGISTER_ITER`, and `FP_TRACK_ITER`.

---

## Dataset

RGB images were collected with RealSense D455 and labeled as polygon segmentation masks in Roboflow. The three class-specific datasets were merged and class IDs were remapped for YOLOv8-seg training.

| Class | Train | Valid |
|-------|------:|------:|
| cylinder | 112 | 21 |
| hole | 137 | 25 |
| cross | 141 | 26 |

---

## Installation

### 1. Base Python Dependencies

```bash
python3 -m pip install -r requirements.txt
```

`pyrealsense2` requires Intel RealSense SDK 2.0 / librealsense on the host.

### 2. ROS2 Humble

`foundation_pose_node.py` publishes ROS2 topics, so ROS2 Humble or newer should be sourced before running:

```bash
source /opt/ros/humble/setup.bash
```

### 3. FoundationPose Dependencies

FoundationPose requires CUDA, PyTorch with matching CUDA, nvdiffrast, PyTorch3D, and pretrained FoundationPose weights.

```bash
bash setup_foundationpose.sh
```

If your environment uses a specific virtualenv Python, pass it explicitly:

```bash
PYTHON_BIN=/path/to/venv/bin/python bash setup_foundationpose.sh
```

The script clones FoundationPose into:

```text
~/FoundationPose
```

If the Google Drive weight download fails, download the official FoundationPose weights manually from the NVlabs FoundationPose repository and place them under:

```text
~/FoundationPose/weights/
```

### 4. CAD Meshes

The 6D node expects one mesh per class:

```text
CAD/
├── cross.stl
├── cylinder.stl
└── hole.stl
```

Default mesh scale is `0.001`, assuming STL files are in millimeters. If your CAD files are already in meters:

```bash
export CAD_MESH_SCALE=1.0
```

---

## Run

### 1. 2.5D Baseline

```bash
python3 detect_3d_pose.py
```

Output example:

```text
[cylinder] conf=0.92 | 3D=(+0.031, -0.012, 0.423) m | angle=87.3 deg
[cross   ] conf=0.88 | 3D=(-0.104, +0.021, 0.381) m | angle=12.1 deg
```

### 2. FoundationPose 6D Node

```bash
source /opt/ros/humble/setup.bash
python3 foundation_pose_node.py
```

Check output:

```bash
ros2 topic echo /object_poses
ros2 topic echo /object_pose_stamped
```

### Useful Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL_PATH` | `best.pt` | YOLOv8 segmentation model path |
| `FOUNDATIONPOSE_DIR` | `~/FoundationPose` | FoundationPose source directory |
| `CAD_MESH_SCALE` | `0.001` | Mesh unit scale before pose estimation |
| `CONF_THRESH` | `0.4` | YOLO confidence threshold |
| `FP_REGISTER_ITER` | `5` | FoundationPose initial registration iterations |
| `FP_TRACK_ITER` | `2` | FoundationPose tracking iterations |
| `FP_ALWAYS_REGISTER` | `0` | Set to `1` to run register every frame |
| `POSE_FRAME_ID` | `camera_color_optical_frame` | ROS pose frame ID |

---

## Training

Place Roboflow exports under `cross/`, `cylinder/`, and `hole/`, each with `train/`, `valid/`, and `test/` folders.

```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"
python3 train_yolo.py
```

Training output:

```text
runs/segment/objects_seg/weights/best.pt
```

The realtime scripts load the root-level `best.pt` by default. After retraining, copy the new checkpoint:

```bash
cp runs/segment/objects_seg/weights/best.pt best.pt
```

---

## Result Visualization

```bash
python3 analyze_results.py
python3 eval_plot.py
```

These scripts consume `runs/segment/train/results.csv` and generate summary plots/tables.

---

## Project Structure

```text
rgbd-object-pose-estimation/
├── foundation_pose_node.py        # FoundationPose CAD-aligned 6D pose ROS2 node
├── setup_foundationpose.sh        # FoundationPose/CUDA dependency setup helper
├── requirements-foundationpose.txt
├── CAD/
│   ├── cross.stl
│   ├── cylinder.stl
│   └── hole.stl
├── detect_3d_pose.py              # 2.5D baseline: mask + depth + 2D angle
├── pose_publisher.py              # ROS2 JSON publisher for baseline pose
├── train_yolo.py                  # YOLOv8-seg training
├── data_collector.py              # RealSense image capture
├── analyze_results.py             # Training result analysis
├── eval_plot.py                   # Metric table/plot generation
├── test_3d_pose.py                # Pixel-to-3D test utility
├── test_d455.py                   # RealSense stream test
├── object_cropper.py              # Color/depth crop helper
├── best.pt                        # Trained YOLOv8n-seg model
├── training_curves.png
├── training_best_metrics.png
└── training_summary_table.png
```

---

## Portfolio Takeaways

- Built an end-to-end RGB-D perception pipeline from data collection to deployment.
- Trained a custom YOLOv8 instance segmentation model for small industrial parts.
- Extended a 2.5D depth baseline into CAD-based 6D pose estimation with FoundationPose.
- Designed ROS2 outputs that are directly usable by a robot manipulation stack.
- Preserved fallback behavior and explicit pose provenance for safer debugging.

---

## Limitations and Next Steps

- Add calibrated 6D ground-truth evaluation with ADD/ADD-S, translation error, and rotation error.
- Add robot hand-eye calibration and transform publication from camera frame to robot base frame.
- Benchmark FPS for `register()` and `track_one()` separately on RTX 4060 Laptop.
- Improve multi-instance tracking if several objects of the same class are visible at once.

---

## English Summary

This project estimates object pose from an Intel RealSense D455 RGB-D camera. A YOLOv8 segmentation model detects `cross`, `cylinder`, and `hole` objects. The baseline estimates 3D position and a 2D mask orientation from depth. The FoundationPose extension uses RGB-D, object masks, camera intrinsics, and CAD meshes to publish CAD-aligned 6D pose as ROS2 JSON and PoseStamped messages.
