import cv2
import numpy as np
import pyrealsense2 as rs
import time
import os

def main():
    # 저장할 폴더 생성
    save_dir = "dataset_2/images"
    os.makedirs(save_dir, exist_ok=True)

    # 파이프라인 설정
    pipeline = rs.pipeline()
    config = rs.config()
    
    # YOLO 학습용 고화질 RGB 프레임
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # 추후 뎁스가 필요할 수 있으므로 뎁스 센서 역시 활성화
    try:
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    except Exception as e:
        print("뎁스 센서 초기화 실패 (무시하고 RGB만 진행):", e)

    # 파이프라인 시작
    profile = pipeline.start(config)

    # ---------------------------------------------
    # [설정] 리얼센스 카메라 화이트 밸런스 끄기
    # ---------------------------------------------
    for sensor in profile.get_device().query_sensors():
        if sensor.is_color_sensor():
            # 자동 화이트밸런스를 강제로 끕니다 (0: Off, 1: On)
            if sensor.supports(rs.option.enable_auto_white_balance):
                sensor.set_option(rs.option.enable_auto_white_balance, 0)
            break

    # 상태 관리 변수
    is_recording = False
    last_save_time = 0
    save_interval = 0.5  # 0.5초마다 1장씩 자동 저장 (초당 2프레임)
    saved_count = 0

    print("=" * 40)
    print("🤖 YOLO 학습 데이터(이미지) 수집기 실행")
    print(f"저장 경로: {os.path.abspath(save_dir)}")
    print("-" * 40)
    print("단축키 안내:")
    print("  [s] 키: 현재 프레임 1장만 수동 저장")
    print("  [r] 키: 자동 연속 저장(0.5초 간격) 시작/중지 토글")
    print("  [q] 키: 프로그램 종료")
    print("=" * 40)

    try:
        while True:
            # 프레임 대기
            frames = pipeline.wait_for_frames()
            
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Numpy 배열로 변환
            color_image = np.asanyarray(color_frame.get_data())
            display_image = color_image.copy()

            # 화면 좌측 상단에 상태 표시 (YOLO 모델엔 영향을 주지 않기 위해 display_image에만 그립니다)
            status_text = "RECORDING (Auto)" if is_recording else "Standby (Manual)"
            status_color = (0, 0, 255) if is_recording else (0, 255, 0)
            
            cv2.putText(display_image, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display_image, f"Saved: {saved_count} imgs", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # 수집 대상 물체를 조준할 수 있도록 중앙 십자선 추가
            h, w = display_image.shape[:2]
            cv2.drawMarker(display_image, (w//2, h//2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)

            cv2.imshow("Dataset Collector", display_image)

            # --- 저장 처리 로직 ---
            current_time = time.time()
            save_flag = False

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_flag = True
            elif key == ord('r'):
                is_recording = not is_recording
                print(f"▶ 자동 저장 상태 변경: {'[ON] 수집중' if is_recording else '[OFF] 정지'}")

            # 자동 저장 상태이면서 지정한 쿨타임(0.5초)이 지난 경우
            if is_recording and (current_time - last_save_time >= save_interval):
                save_flag = True

            # 실제 저장 수행 (텍스트나 십자선이 없는 원본 color_image 저장)
            if save_flag:
                # 파일명이 겹치지 않도록 시간 추가
                filename = os.path.join(save_dir, f"frame_{int(current_time * 1000)}.jpg")
                cv2.imwrite(filename, color_image) 
                
                saved_count += 1
                last_save_time = current_time
                
                if not is_recording:
                    print(f"📸 [{saved_count}] 수동 저장 완료: {filename}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("=" * 40)
        print(f"🎉 수집 완료! 총 {saved_count}장의 원본 이미지가 {save_dir}/ 에 확보되었습니다.")

if __name__ == "__main__":
    main()
