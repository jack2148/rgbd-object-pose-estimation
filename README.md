# RGB-D Object Pose Estimation

[KR] Intel RealSense D455와 YOLOv8 인스턴스 세그멘테이션을 활용한 실시간 3D 물체 자세 추정 시스템입니다. 산업용 부품 3종(실린더, 홀, 크로스)을 실시간으로 검출하고 각 물체의 **3D 좌표(X, Y, Z)**와 **방향각**을 출력합니다.

[EN] A real-time 3D object pose estimation system using Intel RealSense D455 and YOLOv8 instance segmentation. Detects three industrial parts (cylinder, hole, cross) and outputs their **3D coordinates (X, Y, Z)** and **orientation angle** in real time.

---

## 주요 기능

- YOLOv8n-seg 기반 인스턴스 세그멘테이션 (mAP50 평균 0.929)
- RGB-D 정합(`rs.align`)을 통한 픽셀 단위 깊이 매핑
- 마스크 중심점 역투영으로 카메라 좌표계 3D 위치 추정
- 마스크 중앙값 깊이 샘플링으로 노이즈 강건성 확보
- minAreaRect 기반 방향각 추정
- 3종 데이터셋 병합 및 클래스 ID 자동 리매핑
- ROS2 Humble 연동 — JSON 포맷 `/object_poses` 토픽 발행 (10 Hz)

---

## 시스템 구조

### 하드웨어

Intel RealSense D455, NVIDIA GeForce RTX 4060 Laptop

### 소프트웨어

Python 3.10, YOLOv8n-seg (Ultralytics), pyrealsense2, OpenCV, ROS2 Humble (선택)

### 데이터 흐름

```
데이터 수집            라벨링           학습               추론
data_collector.py  →  Roboflow  →  train_yolo.py  →  detect_3d_pose.py
RealSense D455        polygon seg    YOLOv8n-seg       3D position + angle
```

### 3D 자세 추정 파이프라인 (detect_3d_pose.py)

1. RGB-D 프레임 정합 (`rs.align`) — 컬러-깊이 픽셀 1:1 대응
2. YOLOv8-seg 추론 — 인스턴스 마스크 획득
3. 마스크 중심점 `(cx, cy)` 계산
4. 마스크 영역 깊이 중앙값 샘플링 → `depth_m`
5. 카메라 내부 파라미터로 역투영 → `(X, Y, Z)` [m]
6. 마스크 윤곽에 `minAreaRect` 피팅 → `angle` [deg]

```
X = (cx - ppx) * depth_m / fx
Y = (cy - ppy) * depth_m / fy
Z = depth_m
```

---

## 학습 결과

mAP50이 20 에폭 내에 0.93 이상으로 수렴하고 100 에폭까지 Train/Val Loss가 함께 감소 — 과적합 없음.

Cylinder와 Cross는 형상이 뚜렷해 mAP50 0.95 이상을 달성했습니다. Hole은 카메라 각도에 따라 형상이 원형에서 타원형으로 변해 상대적으로 낮지만, Precision은 0.946으로 오탐은 적습니다.

| Class    | mAP50 (Box) | mAP50 (Mask) | Precision | Recall |
|----------|:-----------:|:------------:|:---------:|:------:|
| cylinder |   0.9950    |    0.9950    |   0.9839  | 1.0000 |
| hole     |   0.8473    |    0.8394    |   0.9458  | 0.8571 |
| cross    |   0.9527    |    0.9527    |   0.9062  | 0.9666 |
| **mean** | **0.9317**  |  **0.9290**  | **0.9453**|**0.9412**|

**Validation predictions**

| Labels | Predictions |
|--------|-------------|
| ![val labels](runs/segment/train/val_batch0_labels.jpg) | ![val preds](runs/segment/train/val_batch0_pred.jpg) |

**Training curves**

![Training Curves](training_curves.png)

---

## 데이터셋

| Class    | Train | Valid |
|----------|------:|------:|
| cylinder |   112 |    21 |
| hole     |   137 |    25 |
| cross    |   141 |    26 |

RealSense D455로 직접 수집. 자동 화이트밸런스 비활성화로 색상 일관성 유지. Roboflow에서 폴리곤 세그멘테이션 라벨링 후 3개 데이터셋을 병합해 학습.

---

## 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

> `pyrealsense2`는 [Intel RealSense SDK 2.0](https://github.com/IntelRealSense/librealsense) 설치가 필요합니다.
> `pose_publisher.py`는 ROS2 Humble 이상이 추가로 필요합니다.

### 2. 실시간 검출 (RealSense D455 필요)

```bash
python detect_3d_pose.py
```

`ESC`로 종료. 터미널 출력 예시:

```
[cylinder] conf=0.92 | 3D=(+0.031, -0.012, 0.423) m | angle=87.3°
[cross   ] conf=0.88 | 3D=(-0.104, +0.021, 0.381) m | angle=12.1°
```

### 3. 데이터 수집

```bash
python data_collector.py
# r: 자동 캡처 토글 (0.5초 간격)
# s: 수동 단일 촬영
# q: 종료
```

### 4. 학습

`cross/`, `cylinder/`, `hole/` 폴더에 Roboflow export를 배치한 뒤 (각 폴더 내 `train/`, `valid/`, `test/` 구조):

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"
python train_yolo.py
```

학습 결과 → `runs/segment/objects_seg/weights/best.pt`

### 5. 결과 시각화

```bash
python analyze_results.py   # runs/segment/train/results.csv 필요
python eval_plot.py         # 상세 요약 테이블
```

### 6. ROS2 자세 발행 (선택)

```bash
# /object_poses 토픽으로 JSON 발행 (std_msgs/String)
python pose_publisher.py
```

---

## 프로젝트 구조

```
rgbd-object-pose-estimation/
├── detect_3d_pose.py      # 메인 — 실시간 마스크 + 3D 자세 출력
├── train_yolo.py          # 데이터셋 병합 + YOLOv8 학습
├── data_collector.py      # RealSense 이미지 수집 도구
├── analyze_results.py     # 학습 결과 시각화
├── eval_plot.py           # 평가 요약 테이블
├── pose_publisher.py      # ROS2 노드 — JSON 토픽 발행
├── test_3d_pose.py        # 클릭으로 픽셀 3D 좌표 확인 유틸
├── test_d455.py           # RealSense 기본 스트림 테스트
├── object_cropper.py      # 색상+깊이 임계값 기반 크롭 도구
└── best.pt                # 학습된 YOLOv8n-seg 모델
```

---

## 기술 스택

| 분류 | 내용 |
|------|------|
| OS | Ubuntu 22.04 |
| Language | Python 3.10 |
| Model | YOLOv8n-seg (Ultralytics) |
| Camera | Intel RealSense D455 |
| GPU | NVIDIA GeForce RTX 4060 Laptop |
| Libraries | pyrealsense2, OpenCV, NumPy, Matplotlib |
| ROS2 | Humble (pose_publisher.py only) |

---

## English Summary

Real-time 3D object pose estimation using Intel RealSense D455 and YOLOv8 instance segmentation. Detects three industrial parts (cylinder, hole, cross) and outputs 3D position and orientation angle per object.

Cylinder and cross achieve mAP50 above 0.95 due to their distinctive geometry. Hole scores lower (0.847) as its shape varies with camera angle, but Precision stays at 0.946 — few false positives. mAP50 converges above 0.93 within 20 epochs with no overfitting across 100 epochs.
