import cv2
import numpy as np
import pyrealsense2 as rs
import time

def main():
    # 파이프라인 설정
    pipeline = rs.pipeline()
    config = rs.config()

    # 해상도 및 프레임 레이트 설정
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 파이프라인 시작
    profile = pipeline.start(config)

    # Depth와 Color를 정렬(Align)하기 위한 객체 (Depth를 Color 프레임에 맞춤)
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        while True:
            cropped_object = None  # 매 프레임마다 크롭 객체 초기화 (저장 방어용)
            # 프레임 대기
            frames = pipeline.wait_for_frames()
            
            # Depth 프레임을 Color 프레임 해상도에 맞게 정렬
            aligned_frames = align.process(frames)

            # 정렬된 프레임 가져오기
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 인텔 리얼센스 데이터 배열로 변환
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 깊이 스케일 정보 가져오기 (미터 단위)
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

            #---------- 1. Depth & Color Thresholding (거리 + 색상으로 물체 분리) ----------
            # [수정할 파라미터 1] 거리 임계값 (현재 0.3m ~ 1.0m)
            min_dist = 0.2 / depth_scale
            max_dist = 0.6 / depth_scale
            depth_mask = cv2.inRange(depth_image, min_dist, max_dist)

            # [수정할 파라미터 2] 네이비(Navy) 색상 범위 추출 (HSV 변환)
            hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            
            # 주변 조명에 따라 아래 OpenCV HSV 수치를 미세조정 해야합니다.
            # H(색상): 파란색 대역(100~130) 범위
            # S(채도): 50~255 (물가짐 정도)
            # V(명도): 네이비는 어두우므로 높지 않게(20~150) 설정
            lower_navy = np.array([80, 50, 120])
            upper_navy = np.array([100, 255, 160])
            color_mask = cv2.inRange(hsv_image, lower_navy, upper_navy)

            # Depth(거리) 마스크와 Color(네이비) 마스크의 교집합 연산
            # => "0.3~1.0m 사이에 있으면서, 동시에 네이비 색상인 픽셀"만 남김
            combined_mask = cv2.bitwise_and(depth_mask, color_mask)

            #---------- 2. Morphological 연산 (노이즈 제거) ----------
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

            #---------- 3. 윤곽선(Contour) 검출로 물체 형상 인식 ----------
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 면적이 가장 큰 외곽선을 대상 물체로 간주
                largest_contour = max(contours, key=cv2.contourArea)

                # 노이즈를 걸러내기 위한 면적 조건
                if cv2.contourArea(largest_contour) > 2000:
                    # 해당 형상(Contour)의 Bounding Box 구하기 (x 시작, y 시작, 가로길이, 세로길이)
                    x, y, w, h = cv2.boundingRect(largest_contour)

                    # 바운딩 박스를 그려 시각적으로 확인
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    #---------- 4. 이미지 크롭 (물체 잘라내기) ----------
                    # RGB Color 이미지에서 해당 부분만 크롭
                    cropped_object = color_image[y:y+h, x:x+w]
                    
                    # 크롭된 이미지 화면에 출력
                    cv2.imshow('Cropped Object', cropped_object)

            # 시각화를 위한 Depth 마스크 출력
            cv2.imshow('Depth Mask', depth_mask)
            cv2.imshow('Original Image', color_image)

            # 키보드 입력 대기 (1ms 단위)
            key = cv2.waitKey(1) & 0xFF
            
            # 'q' 키를 누르면 프로그램 종료
            if key == ord('q'):
                break
            
            # 's' 키를 누르면 크롭된 이미지(바운딩박스 영역) 파일로 저장
            elif key == ord('s'):
                if cropped_object is not None and cropped_object.size > 0:
                    filename = f"navy_cylinder_{int(time.time())}.jpg"
                    cv2.imwrite(filename, cropped_object)
                    print(f"✅ [저장 성공] 파일명: {filename} 로 현재 물체가 저장되었습니다!")
                else:
                    print("❌ [저장 실패] 화면에 지정한 밝은 네이비 원기둥이 인식되지 않았습니다.")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
