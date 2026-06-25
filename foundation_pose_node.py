"""
FoundationPose-based 6D pose node for RGB-D object pose estimation.

Pipeline:
  RealSense D455 RGB-D
    -> YOLOv8 instance segmentation
    -> nearest valid masked object selection
    -> FoundationPose CAD alignment when available
    -> ROS2 JSON + PoseStamped publish

The node publishes a real CAD-aligned 6D pose only when FoundationPose,
the class mesh, RGB, depth, and instance mask are all available. If that
path fails, it falls back to depth PCA so the system can still report an
approximate position and orientation for debugging.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import trimesh
from scipy.spatial.transform import Rotation
from ultralytics import YOLO

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String


# ---------------------------------------------------------------------------
# FoundationPose import
# ---------------------------------------------------------------------------

FOUNDATIONPOSE_DIR = Path(os.environ.get("FOUNDATIONPOSE_DIR", Path.home() / "FoundationPose"))
if str(FOUNDATIONPOSE_DIR) not in sys.path:
    sys.path.insert(0, str(FOUNDATIONPOSE_DIR))

try:
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
    import nvdiffrast.torch as dr

    FOUNDATIONPOSE_AVAILABLE = True
except ImportError as exc:
    print(f"[WARN] FoundationPose import failed: {exc}")
    print("       Run setup_foundationpose.sh or set FOUNDATIONPOSE_DIR.")
    FOUNDATIONPOSE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("YOLO_MODEL_PATH", BASE_DIR / "best.pt"))
MESH_DIRS = [BASE_DIR / "CAD", BASE_DIR / "meshes"]
MESH_EXTS = [".stl", ".obj", ".ply"]

# Many hobby/desktop CAD exports are in millimeters. Use CAD_MESH_SCALE=1.0
# when your mesh is already in meters.
MESH_SCALE = float(os.environ.get("CAD_MESH_SCALE", "0.001"))

CONF_THRESH = float(os.environ.get("CONF_THRESH", "0.4"))
FP_REGISTER_ITER = int(os.environ.get("FP_REGISTER_ITER", "5"))
FP_TRACK_ITER = int(os.environ.get("FP_TRACK_ITER", "2"))
TRACK_LOSS_THR = float(os.environ.get("TRACK_LOSS_THR", "0.2"))
FP_ALWAYS_REGISTER = os.environ.get("FP_ALWAYS_REGISTER", "0") == "1"

CLASS_COLORS = {
    "cross": (0, 220, 0),
    "cylinder": (0, 140, 255),
    "hole": (220, 0, 220),
}

JSON_TOPIC = "/object_poses"
POSE_TOPIC = "/object_pose_stamped"
FRAME_ID = os.environ.get("POSE_FRAME_ID", "camera_color_optical_frame")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def build_camera_matrix(intrinsics) -> np.ndarray:
    """Convert RealSense intrinsics to a 3x3 camera matrix."""
    return np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def mask_from_polygon(mask_xy: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    """Rasterize an Ultralytics segmentation polygon to a binary mask."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if len(mask_xy) > 0:
        cv2.fillPoly(mask, [mask_xy.astype(np.int32)], 255)
    return mask


def depth_median_in_mask(depth_image: np.ndarray, mask: np.ndarray, depth_scale: float) -> float:
    """Return median valid depth in meters for mask pixels."""
    z = depth_image[mask > 0].astype(np.float32) * depth_scale
    z = z[z > 0]
    return float(np.median(z)) if len(z) > 0 else float("inf")


def load_mesh(class_name: str) -> trimesh.Trimesh:
    """Load a class mesh from CAD/ or meshes/ and apply unit scaling."""
    candidates = [mesh_dir / f"{class_name}{ext}" for mesh_dir in MESH_DIRS for ext in MESH_EXTS]
    mesh_path = next((path for path in candidates if path.exists()), None)
    if mesh_path is None:
        searched = "\n  ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Mesh not found for class '{class_name}'. Searched:\n  {searched}")

    mesh = trimesh.load(str(mesh_path), force="mesh")
    if MESH_SCALE != 1.0:
        mesh.apply_scale(MESH_SCALE)
    return mesh


def pose_to_dict(
    pose_mat: np.ndarray,
    class_name: str,
    confidence: float,
    pose_source: str,
    depth_median_m: float,
    detected_count: int,
) -> dict:
    """Convert a 4x4 pose matrix to a JSON-serializable payload."""
    translation = pose_mat[:3, 3]
    rotation = pose_mat[:3, :3]
    quat = Rotation.from_matrix(rotation).as_quat()  # [x, y, z, w]

    return {
        "class": class_name,
        "confidence": round(float(confidence), 3),
        "pose_source": pose_source,
        "is_cad_aligned_6d": pose_source == "foundationpose",
        "frame_id": FRAME_ID,
        "position": {
            "x": round(float(translation[0]), 4),
            "y": round(float(translation[1]), 4),
            "z": round(float(translation[2]), 4),
        },
        "orientation": {
            "x": round(float(quat[0]), 6),
            "y": round(float(quat[1]), 6),
            "z": round(float(quat[2]), 6),
            "w": round(float(quat[3]), 6),
        },
        "pose_matrix": pose_mat.tolist(),
        "priority": {
            "selected": True,
            "reason": "nearest_valid_mask_depth",
            "depth_median_m": round(float(depth_median_m), 4),
            "detected_count": int(detected_count),
        },
    }


def make_pose_stamped(pose_mat: np.ndarray) -> PoseStamped:
    """Convert a 4x4 pose matrix to geometry_msgs/PoseStamped."""
    translation = pose_mat[:3, 3]
    rotation = pose_mat[:3, :3]
    quat = Rotation.from_matrix(rotation).as_quat()

    msg = PoseStamped()
    msg.header.frame_id = FRAME_ID
    msg.pose.position.x = float(translation[0])
    msg.pose.position.y = float(translation[1])
    msg.pose.position.z = float(translation[2])
    msg.pose.orientation.x = float(quat[0])
    msg.pose.orientation.y = float(quat[1])
    msg.pose.orientation.z = float(quat[2])
    msg.pose.orientation.w = float(quat[3])
    return msg


def draw_pose_axis(image: np.ndarray, pose_mat: np.ndarray, camera_matrix: np.ndarray, axis_len: float = 0.05) -> None:
    """Project a 3D pose axis triad onto the color image."""
    translation = pose_mat[:3, 3]
    rotation = pose_mat[:3, :3]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    def project(point_3d: np.ndarray):
        x, y, z = point_3d
        if z <= 0:
            return None
        return int(x * fx / z + cx), int(y * fy / z + cy)

    origin = project(translation)
    if origin is None:
        return

    axes = [
        (rotation[:, 0], (0, 0, 255)),
        (rotation[:, 1], (0, 255, 0)),
        (rotation[:, 2], (255, 0, 0)),
    ]
    for axis_vec, color in axes:
        endpoint = project(translation + axis_vec * axis_len)
        if endpoint is not None:
            cv2.arrowedLine(image, origin, endpoint, color, 2, tipLength=0.2)


# ---------------------------------------------------------------------------
# FoundationPose wrapper
# ---------------------------------------------------------------------------

class FPEstimator:
    """Per-class FoundationPose estimator with register/track switching."""

    def __init__(self, class_name: str, mesh: trimesh.Trimesh, glctx, scorer, refiner):
        self.class_name = class_name
        self.estimator = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=0,
        )
        self.registered = False
        self.last_score = 1.0

    def estimate(
        self,
        bgr_image: np.ndarray,
        depth_image: np.ndarray,
        depth_scale: float,
        mask: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> np.ndarray:
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.uint8)
        depth_m = depth_image.astype(np.float32) * depth_scale

        should_register = FP_ALWAYS_REGISTER or not self.registered or self.last_score < TRACK_LOSS_THR
        if should_register:
            pose = self.estimator.register(
                K=camera_matrix,
                rgb=rgb_image,
                depth=depth_m,
                ob_mask=mask,
                iteration=FP_REGISTER_ITER,
            )
            self.registered = True
        else:
            pose = self.estimator.track_one(
                rgb=rgb_image,
                depth=depth_m,
                K=camera_matrix,
                iteration=FP_TRACK_ITER,
            )

        if hasattr(self.estimator, "last_score"):
            self.last_score = float(self.estimator.last_score)
        return np.array(pose, dtype=np.float64)

    def reset(self) -> None:
        self.registered = False
        self.last_score = 1.0


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class FoundationPoseNode(Node):
    """Real-time RGB-D segmentation and FoundationPose 6D pose publisher."""

    def __init__(self):
        super().__init__("foundation_pose_node")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

        self.model = YOLO(str(MODEL_PATH))
        self.get_logger().info(f"[YOLO] loaded: {MODEL_PATH}")

        self.fp_available = FOUNDATIONPOSE_AVAILABLE
        if self.fp_available:
            try:
                self.glctx = dr.RasterizeCudaContext()
                self.scorer = ScorePredictor()
                self.refiner = PoseRefinePredictor()
                self.get_logger().info("[FoundationPose] GPU context initialized")
            except Exception as exc:
                self.get_logger().error(f"[FoundationPose] GPU initialization failed: {exc}")
                self.fp_available = False

        self._fp_estimators: dict[str, FPEstimator] = {}
        self._last_target_class: str | None = None

        self.pub_json = self.create_publisher(String, JSON_TOPIC, 10)
        self.pub_pose = self.create_publisher(PoseStamped, POSE_TOPIC, 10)

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self.pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.camera_matrix = build_camera_matrix(color_profile.get_intrinsics())
        self.align = rs.align(rs.stream.color)

        self.timer = self.create_timer(0.1, self._timer_cb)
        self.get_logger().info(
            f"ready: JSON={JSON_TOPIC}, PoseStamped={POSE_TOPIC}, "
            f"FP={'ON' if self.fp_available else 'OFF'}"
        )

    def _get_fp_estimator(self, class_name: str) -> FPEstimator | None:
        if class_name in self._fp_estimators:
            return self._fp_estimators[class_name]

        try:
            mesh = load_mesh(class_name)
            estimator = FPEstimator(class_name, mesh, self.glctx, self.scorer, self.refiner)
            self._fp_estimators[class_name] = estimator
            self.get_logger().info(f"[FoundationPose] estimator created: {class_name}")
            return estimator
        except FileNotFoundError as exc:
            self.get_logger().error(str(exc))
            return None

    def _timer_cb(self) -> None:
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        display = color.copy()

        results = self.model(color, conf=CONF_THRESH, verbose=False)[0]
        detections = self._collect_detections(results, color, depth)
        detections.sort(key=lambda det: det["depth_median_m"])
        target = detections[0] if detections else None

        target_payload = None
        if target is not None:
            target_payload = self._estimate_and_publish_target(target, color, depth, display, len(detections))

        cv2.putText(
            display,
            f"FoundationPose: {'ON' if self.fp_available else 'OFF'} | target-only nearest depth",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
        )
        cv2.imshow("FoundationPose 6D Pose (ESC to quit)", display)
        if cv2.waitKey(1) == 27:
            rclpy.shutdown()

        msg = String()
        msg.data = json.dumps(
            {
                "target": target_payload,
                "objects": [target_payload] if target_payload is not None else [],
                "detected_count": len(detections),
                "notes": "is_cad_aligned_6d=true means FoundationPose produced the pose.",
            },
            ensure_ascii=False,
        )
        self.pub_json.publish(msg)

    def _collect_detections(self, results, color: np.ndarray, depth: np.ndarray) -> list[dict]:
        detections = []
        if results.masks is None:
            return detections

        for i, mask_xy in enumerate(results.masks.xy):
            cls_id = int(results.boxes.cls[i])
            class_name = results.names[cls_id]
            confidence = float(results.boxes.conf[i])
            mask = mask_from_polygon(mask_xy, color.shape)
            depth_median_m = depth_median_in_mask(depth, mask, self.depth_scale)
            if not np.isfinite(depth_median_m):
                continue

            bbox = results.boxes.xyxy[i].cpu().numpy().astype(int)
            detections.append(
                {
                    "class_name": class_name,
                    "confidence": confidence,
                    "mask_xy": mask_xy,
                    "mask": mask,
                    "bbox": bbox,
                    "depth_median_m": depth_median_m,
                }
            )
        return detections

    def _estimate_and_publish_target(
        self,
        target: dict,
        color: np.ndarray,
        depth: np.ndarray,
        display: np.ndarray,
        detected_count: int,
    ) -> dict | None:
        class_name = target["class_name"]
        mask = target["mask"]
        pose_mat = None
        pose_source = "none"

        if self._last_target_class != class_name:
            for estimator in self._fp_estimators.values():
                estimator.reset()
            self._last_target_class = class_name

        if self.fp_available:
            estimator = self._get_fp_estimator(class_name)
            if estimator is not None:
                try:
                    pose_mat = estimator.estimate(color, depth, self.depth_scale, mask, self.camera_matrix)
                    pose_source = "foundationpose"
                except Exception as exc:
                    self.get_logger().warn(f"[FoundationPose] {class_name} failed: {exc}")
                    estimator.reset()

        if pose_mat is None:
            pose_mat = self._depth_pca_fallback_pose(depth, mask)
            pose_source = "depth_pca_fallback" if pose_mat is not None else "none"

        if pose_mat is None:
            return None

        payload = pose_to_dict(
            pose_mat,
            class_name,
            target["confidence"],
            pose_source,
            target["depth_median_m"],
            detected_count,
        )

        pose_msg = make_pose_stamped(pose_mat)
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_pose.publish(pose_msg)

        self._draw_target(display, target, pose_mat, pose_source)
        return payload

    def _depth_pca_fallback_pose(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        """Approximate pose: masked point-cloud centroid plus PCA axes."""
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

        rows, cols = np.where(mask > 0)
        z = depth[rows, cols].astype(np.float32) * self.depth_scale
        valid = z > 0
        rows, cols, z = rows[valid], cols[valid], z[valid]
        if len(z) < 10:
            return None

        z_med = np.median(z)
        valid = np.abs(z - z_med) < 0.05
        rows, cols, z = rows[valid], cols[valid], z[valid]
        if len(z) < 10:
            return None

        x = (cols - cx) * z / fx
        y = (rows - cy) * z / fy
        points = np.stack([x, y, z], axis=1)

        centroid = np.median(points, axis=0)
        covariance = np.cov((points - centroid).T)
        _, eigvecs = np.linalg.eigh(covariance)
        order = np.argsort(np.linalg.eigvalsh(covariance))[::-1]
        rotation = eigvecs[:, order]
        if np.linalg.det(rotation) < 0:
            rotation[:, 2] *= -1

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = centroid
        return pose

    def _draw_target(self, display: np.ndarray, target: dict, pose_mat: np.ndarray, pose_source: str) -> None:
        class_name = target["class_name"]
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        mask_xy = target["mask_xy"]
        x1, y1, x2, y2 = target["bbox"]

        overlay = display.copy()
        if len(mask_xy) > 0:
            cv2.fillPoly(overlay, [mask_xy.astype(np.int32)], color)
        display[:] = cv2.addWeighted(overlay, 0.4, display, 0.6, 0)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        draw_pose_axis(display, pose_mat, self.camera_matrix)

        translation = pose_mat[:3, 3]
        label = (
            f"{class_name} {target['confidence']:.2f} | {pose_source} | "
            f"X:{translation[0]:+.3f} Y:{translation[1]:+.3f} Z:{translation[2]:.3f}m"
        )
        cv2.putText(display, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    def destroy_node(self) -> None:
        self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = FoundationPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
