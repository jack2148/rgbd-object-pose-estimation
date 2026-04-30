"""
detect_3d_pose.py
RealSense D455로부터 컬러+뎁스 스트림을 받아
YOLOv8-seg 모델로 cross / cylinder / hole을 감지하고
각 물체의 3D 위치(X, Y, Z)와 2D 방향각(angle)을 실시간으로 출력합니다.

학습 후 실행:
    python detect_3d_pose.py

종료: ESC
"""

import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
from pathlib import Path

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
MODEL_PATH   = str(Path(__file__).parent / 'best.pt')
CLASS_NAMES  = ['cross', 'cylinder', 'hole']
CLASS_COLORS = [(0, 220, 0), (0, 140, 255), (220, 0, 220)]  # BGR
CONF_THRESH  = 0.4


# ---------------------------------------------------------------------------
# 유틸 함수
# ---------------------------------------------------------------------------

def median_depth_from_mask(depth_image, mask_bin, depth_scale):
    """마스크 영역의 유효 뎁스 중앙값(미터)을 반환합니다."""
    raw = depth_image[mask_bin > 0].astype(np.float32) * depth_scale
    valid = raw[raw > 0]
    if len(valid) == 0:
        return 0.0
    return float(np.median(valid))


def pixel_to_3d(u, v, depth_m, intr):
    """픽셀 (u, v) + 뎁스 -> 카메라 좌표계 3D 점 (X, Y, Z) [미터]."""
    X = (u - intr.ppx) * depth_m / intr.fx
    Y = (v - intr.ppy) * depth_m / intr.fy
    Z = depth_m
    return X, Y, Z


def mask_orientation(mask_bin):
    """
    마스크의 주축 방향각(도)을 반환합니다.
    - minAreaRect 로 외접 직사각형을 구하고 긴 변 방향을 반환합니다.
    - 반환값: (angle_deg, contour) | angle_deg 는 -90 ~ 90 범위
    """
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None, contour

    rect = cv2.minAreaRect(contour)   # (center, (w, h), angle)
    w, h = rect[1]
    angle = rect[2]
    # OpenCV minAreaRect 각도 규칙: 긴 쪽을 기준으로 통일
    if w < h:
        angle += 90
    return angle, contour


def draw_arrow(img, cx, cy, angle_deg, length=35, color=(0, 255, 255)):
    angle_rad = np.radians(angle_deg)
    ex = int(cx + length * np.cos(angle_rad))
    ey = int(cy + length * np.sin(angle_rad))
    cv2.arrowedLine(img, (cx, cy), (ex, ey), color, 2, tipLength=0.3)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    if not Path(MODEL_PATH).exists():
        print(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print("먼저 train_yolo.py 를 실행해 학습을 완료하세요.")
        return

    model = YOLO(MODEL_PATH)

    # RealSense 파이프라인 설정
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)

    profile = pipeline.start(config)
    align   = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()

    print("카메라 내부 파라미터:")
    print(f"  fx={intr.fx:.2f}  fy={intr.fy:.2f}  cx={intr.ppx:.2f}  cy={intr.ppy:.2f}")
    print("ESC 키를 누르면 종료합니다.\n")

    try:
        while True:
            frames         = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_img   = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            display     = color_img.copy()

            # YOLO 추론
            results = model(color_img, conf=CONF_THRESH, verbose=False)[0]

            if results.masks is not None:
                for mask_pts, box in zip(results.masks.xy, results.boxes):
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    color  = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                    name   = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)

                    # 바이너리 마스크 생성
                    mask_bin = np.zeros(color_img.shape[:2], dtype=np.uint8)
                    if len(mask_pts) > 0:
                        cv2.fillPoly(mask_bin, [mask_pts.astype(np.int32)], 255)

                    # 무게중심
                    M = cv2.moments(mask_bin)
                    if M['m00'] == 0:
                        continue
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # 뎁스 (마스크 영역 중앙값)
                    depth_m = median_depth_from_mask(depth_image, mask_bin, depth_scale)
                    if depth_m <= 0:
                        continue

                    # 3D 위치
                    X, Y, Z = pixel_to_3d(cx, cy, depth_m, intr)

                    # 방향각
                    angle, contour = mask_orientation(mask_bin)

                    # ── 시각화 ──────────────────────────────────────────
                    # 마스크 반투명 오버레이
                    overlay = display.copy()
                    cv2.fillPoly(overlay, [mask_pts.astype(np.int32)], color)
                    display = cv2.addWeighted(display, 0.55, overlay, 0.45, 0)

                    # 외곽선
                    if contour is not None:
                        cv2.drawContours(display, [contour], -1, color, 2)

                    # 무게중심 점
                    cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

                    # 방향 화살표
                    if angle is not None:
                        draw_arrow(display, cx, cy, angle)

                    # 바운딩 박스 + 텍스트
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                    texts = [
                        f"{name}  {conf:.2f}",
                        f"X:{X:+.3f} Y:{Y:+.3f} Z:{Z:.3f} m",
                        f"Angle: {angle:.1f} deg" if angle is not None else "Angle: N/A",
                    ]
                    for i, txt in enumerate(texts):
                        cv2.putText(display, txt,
                                    (x1, y1 - 10 - 18 * (len(texts) - 1 - i)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

                    # 터미널 출력
                    ang_str = f"{angle:.1f}°" if angle is not None else "N/A"
                    print(f"[{name:8s}] conf={conf:.2f} | "
                          f"3D=({X:+.3f}, {Y:+.3f}, {Z:.3f}) m | angle={ang_str}")

            cv2.imshow("3D Pose Detection  (ESC to quit)", display)
            if cv2.waitKey(1) == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
