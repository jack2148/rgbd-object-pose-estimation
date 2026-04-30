import pyrealsense2 as rs
import numpy as np
import cv2

# -----------------------------
# Global variables
# -----------------------------
clicked_pixel = None
clicked_result = None  # (u, v, depth, X, Y, Z)

# -----------------------------
# Mouse callback
# -----------------------------
def mouse_callback(event, x, y, flags, param):
    global clicked_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pixel = (x, y)

# -----------------------------
# Main
# -----------------------------
def main():
    global clicked_pixel, clicked_result

    pipeline = rs.pipeline()
    config = rs.config()

    # Stream configuration
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start pipeline
    profile = pipeline.start(config)

    # Align depth to color frame
    align = rs.align(rs.stream.color)

    # Get depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale} m/unit")

    # Get color intrinsics
    color_stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream_profile.get_intrinsics()

    fx, fy = intr.fx, intr.fy
    cx, cy = intr.ppx, intr.ppy

    print("Camera intrinsics:")
    print(f"fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")

    cv2.namedWindow("Color")
    cv2.setMouseCallback("Color", mouse_callback)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Depth colormap for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # If pixel clicked, compute 3D coordinate
            if clicked_pixel is not None:
                u, v = clicked_pixel

                # Bounds check
                h, w, _ = color_image.shape
                if 0 <= u < w and 0 <= v < h:
                    depth = depth_frame.get_distance(u, v)  # meters

                    if depth > 0:
                        X = (u - cx) * depth / fx
                        Y = (v - cy) * depth / fy
                        Z = depth
                        clicked_result = (u, v, depth, X, Y, Z)

                        print("-" * 50)
                        print(f"Pixel (u, v) = ({u}, {v})")
                        print(f"Depth Z      = {depth:.4f} m")
                        print(f"3D point     = ({X:.4f}, {Y:.4f}, {Z:.4f}) m")
                    else:
                        clicked_result = (u, v, 0.0, None, None, None)
                        print("-" * 50)
                        print(f"Pixel (u, v) = ({u}, {v})")
                        print("Invalid depth at this pixel.")

                clicked_pixel = None

            # Draw clicked result on color image
            display_color = color_image.copy()

            if clicked_result is not None:
                u, v, depth, X, Y, Z = clicked_result
                cv2.circle(display_color, (u, v), 5, (0, 255, 0), -1)

                if X is not None:
                    text1 = f"Pixel: ({u},{v})"
                    text2 = f"Depth: {depth:.3f} m"
                    text3 = f"3D: ({X:.3f}, {Y:.3f}, {Z:.3f}) m"
                else:
                    text1 = f"Pixel: ({u},{v})"
                    text2 = "Depth: invalid"
                    text3 = "3D: unavailable"

                cv2.putText(display_color, text1, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)
                cv2.putText(display_color, text2, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
                cv2.putText(display_color, text3, (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

            # Show images
            cv2.imshow("Color", display_color)
            cv2.imshow("Depth", depth_colormap)

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()