import json
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

MODEL_PATH = "runs/segment/train/weights/best.pt"
TOPIC_NAME = "/object_poses"
CONF_THRESH = 0.3
CLASS_COLORS = {
    "cylinder":  (0, 140, 255),
    "hole":      (220, 0, 220),
    "cross":     (0, 220, 0),
}


class PosePublisher(Node):
    def __init__(self):
        super().__init__("pose_publisher")
        self.publisher = self.create_publisher(String, TOPIC_NAME, 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz

        self.model = YOLO(MODEL_PATH)
        self.get_logger().info(f"Loaded model: {MODEL_PATH}")

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)

        profile = self.pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(
            profile.get_stream(rs.stream.depth)
        )
        self.intrinsics = depth_profile.get_intrinsics()

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.get_logger().info("PosePublisher ready")

    def get_point_cloud(self, depth_image, mask_img):
        fx = self.intrinsics.fx
        fy = self.intrinsics.fy
        ppx = self.intrinsics.ppx
        ppy = self.intrinsics.ppy

        rows, cols = np.where(mask_img > 0)
        z_vals = depth_image[rows, cols].astype(float) * 0.001  # mm → m
        valid = z_vals > 0
        rows, cols, z_vals = rows[valid], cols[valid], z_vals[valid]

        if len(z_vals) < 10:
            return None

        z_med = np.median(z_vals)
        valid = np.abs(z_vals - z_med) < 0.05  # keep within 5cm of median
        rows, cols, z_vals = rows[valid], cols[valid], z_vals[valid]

        if len(z_vals) < 10:
            return None

        x = (cols - ppx) * z_vals / fx
        y = (rows - ppy) * z_vals / fy
        return np.stack([x, y, z_vals], axis=1)  # (N, 3)

    def estimate_pose(self, points):
        centroid = np.median(points, axis=0)
        centered = points - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # eigh returns ascending order → reverse for descending
        order = np.argsort(eigenvalues)[::-1]
        axes = eigenvectors[:, order]  # columns are principal axes
        return centroid, axes

    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        objects = []
        display = color_image.copy()

        results = self.model(color_image, conf=CONF_THRESH, verbose=False)[0]

        n_det = len(results.boxes) if results.boxes is not None else 0
        self.get_logger().info(f"detections: {n_det}", throttle_duration_sec=1.0)

        if results.masks is not None:
            for i, mask_xy in enumerate(results.masks.xy):
                cls_id = int(results.boxes.cls[i])
                class_name = results.names[cls_id]
                confidence = float(results.boxes.conf[i])
                color = CLASS_COLORS.get(class_name, (255, 255, 255))

                mask_img = np.zeros(depth_image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask_img, [mask_xy.astype(np.int32)], 255)

                points = self.get_point_cloud(depth_image, mask_img)
                if points is None:
                    continue

                centroid, axes = self.estimate_pose(points)

                objects.append({
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "position": {
                        "x": round(float(centroid[0]), 4),
                        "y": round(float(centroid[1]), 4),
                        "z": round(float(centroid[2]), 4),
                    },
                    "orientation": {
                        "axis_x": [round(float(v), 4) for v in axes[:, 0]],
                        "axis_y": [round(float(v), 4) for v in axes[:, 1]],
                        "axis_z": [round(float(v), 4) for v in axes[:, 2]],
                    },
                })

                # 시각화
                overlay = display.copy()
                if len(mask_xy) > 0:
                    cv2.fillPoly(overlay, [mask_xy.astype(np.int32)], color)
                display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

                x1, y1, x2, y2 = map(int, results.boxes.xyxy[i])
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                cx = int(centroid[0] * self.intrinsics.fx / centroid[2] + self.intrinsics.ppx)
                cy = int(centroid[1] * self.intrinsics.fy / centroid[2] + self.intrinsics.ppy)
                cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

                label = f"{class_name} {confidence:.2f} | X:{centroid[0]:+.2f} Y:{centroid[1]:+.2f} Z:{centroid[2]:.2f}m"
                cv2.putText(display, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("PosePublisher - ESC to quit", display)
        cv2.waitKey(1)

        self._publish(objects)

    def _publish(self, objects):
        msg = String()
        msg.data = json.dumps({"objects": objects})
        self.publisher.publish(msg)

    def destroy_node(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    node = PosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
