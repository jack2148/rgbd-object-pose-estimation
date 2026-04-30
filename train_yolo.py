import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# 클래스 정의: 폴더명 -> class_id
CLASSES = {
    'cross':    0,
    'cylinder': 1,
    'hole':     2,
}

BASE_DIR    = Path(__file__).parent
MERGED_DIR  = BASE_DIR / 'dataset_merged'
SPLITS      = ['train', 'valid', 'test']


def merge_datasets():
    """3개의 개별 데이터셋을 class ID를 remapping해 하나로 합칩니다."""
    print("=== 데이터셋 병합 시작 ===")

    # 이전 병합 결과 제거
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)

    for split in SPLITS:
        (MERGED_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

    total = {s: 0 for s in SPLITS}

    for class_name, class_id in CLASSES.items():
        class_dir = BASE_DIR / class_name
        for split in SPLITS:
            img_src = class_dir / split / 'images'
            lbl_src = class_dir / split / 'labels'
            img_dst = MERGED_DIR / split / 'images'
            lbl_dst = MERGED_DIR / split / 'labels'

            if not img_src.exists():
                print(f"  [경고] {img_src} 없음, 건너뜀")
                continue

            for img_file in sorted(img_src.glob('*.jpg')):
                # 파일명 충돌 방지: 클래스 이름을 prefix로 추가
                dst_stem = f"{class_name}_{img_file.stem}"
                shutil.copy2(img_file, img_dst / f"{dst_stem}.jpg")

                lbl_file = lbl_src / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    lines = lbl_file.read_text().splitlines()
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            parts[0] = str(class_id)   # class ID remapping
                            new_lines.append(' '.join(parts))
                    (lbl_dst / f"{dst_stem}.txt").write_text('\n'.join(new_lines))

                total[split] += 1

    # data.yaml 생성 (절대 경로 사용)
    yaml_content = (
        f"train: {(MERGED_DIR / 'train' / 'images').as_posix()}\n"
        f"val:   {(MERGED_DIR / 'valid' / 'images').as_posix()}\n"
        f"test:  {(MERGED_DIR / 'test'  / 'images').as_posix()}\n"
        f"\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {list(CLASSES.keys())}\n"
    )
    yaml_path = MERGED_DIR / 'data.yaml'
    yaml_path.write_text(yaml_content)

    print(f"  train: {total['train']}장 | valid: {total['valid']}장 | test: {total['test']}장")
    print(f"  data.yaml -> {yaml_path}")
    print("=== 데이터셋 병합 완료 ===\n")
    return str(yaml_path)


def main():
    yaml_path = merge_datasets()

    # YOLOv8 segmentation 모델 (라벨이 polygon 형식이므로 -seg 사용)
    model = YOLO('yolov8n-seg.pt')

    print("=== 학습 시작 ===")
    model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,           # CUDA GPU 0
        name='objects_seg',
        patience=20,        # 20 epoch 동안 개선 없으면 조기 종료
        workers=4,
    )

    best_pt = BASE_DIR / 'runs' / 'segment' / 'objects_seg' / 'weights' / 'best.pt'
    print(f"\n학습 완료! 모델 저장 위치: {best_pt}")


if __name__ == '__main__':
    main()
