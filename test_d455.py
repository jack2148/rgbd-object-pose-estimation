import pyrealsense2 as rs
import numpy as np
import cv2

# pipeline 생성
pipeline = rs.pipeline()

# 스트림 설정
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 카메라 시작
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # numpy 변환
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 중앙 픽셀 depth
        h, w = depth_image.shape
        center_depth = depth_frame.get_distance(w//2, h//2)

        print("Center depth:", round(center_depth, 3), "m")

        # depth 시각화
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        images = np.hstack((color_image, depth_colormap))

        cv2.imshow("RGB | Depth", images)

        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()