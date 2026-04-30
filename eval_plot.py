import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager
import numpy as np

# 한글 폰트 설정
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()

CSV_PATH = "runs/segment/train/results.csv"
OUT_PATH = "runs/segment/train/training_summary_table.png"

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

best_idx = df["metrics/mAP50-95(M)"].idxmax()
best = df.loc[best_idx]
final = df.iloc[-1]

# ── 표 데이터 ────────────────────────────────────────────────────────────────
sections = [
    {
        "title": "모델 설정",
        "color": "#1565C0",
        "rows": [
            ("모델",        "YOLOv8n-seg"),
            ("학습 epoch",  "100"),
            ("이미지 크기", "640 × 640"),
            ("배치 크기",   "16"),
            ("클래스 수",   "3  (cylinder / hole / cross)"),
            ("최적 epoch",  f"{int(best['epoch'])}  (mAP50-95(M) 기준)"),
        ],
    },
    {
        "title": "Box Detection 성능",
        "color": "#2E7D32",
        "rows": [
            ("Precision",   f"{final['metrics/precision(B)']:.4f}",   f"(best: {best['metrics/precision(B)']:.4f})"),
            ("Recall",      f"{final['metrics/recall(B)']:.4f}",      f"(best: {best['metrics/recall(B)']:.4f})"),
            ("mAP50",       f"{final['metrics/mAP50(B)']:.4f}",       f"(best: {best['metrics/mAP50(B)']:.4f})"),
            ("mAP50-95",    f"{final['metrics/mAP50-95(B)']:.4f}",    f"(best: {best['metrics/mAP50-95(B)']:.4f})"),
        ],
    },
    {
        "title": "Mask Segmentation 성능",
        "color": "#6A1B9A",
        "rows": [
            ("Precision",   f"{final['metrics/precision(M)']:.4f}",   f"(best: {best['metrics/precision(M)']:.4f})"),
            ("Recall",      f"{final['metrics/recall(M)']:.4f}",      f"(best: {best['metrics/recall(M)']:.4f})"),
            ("mAP50",       f"{final['metrics/mAP50(M)']:.4f}",       f"(best: {best['metrics/mAP50(M)']:.4f})"),
            ("mAP50-95",    f"{final['metrics/mAP50-95(M)']:.4f}",    f"(best: {best['metrics/mAP50-95(M)']:.4f})"),
        ],
    },
    {
        "title": "Validation Loss (최종)",
        "color": "#E65100",
        "rows": [
            ("box_loss",    f"{final['val/box_loss']:.4f}",    f"(초기: {df.iloc[0]['val/box_loss']:.4f})"),
            ("seg_loss",    f"{final['val/seg_loss']:.4f}",    f"(초기: {df.iloc[0]['val/seg_loss']:.4f})"),
            ("cls_loss",    f"{final['val/cls_loss']:.4f}",    f"(초기: {df.iloc[0]['val/cls_loss']:.4f})"),
        ],
    },
    {
        "title": "평가 요약",
        "color": "#37474F",
        "rows": [
            ("과적합 여부",      "없음  (val loss 안정 수렴)"),
            ("Precision > Recall", "보수적 탐지 — 오검출 적음, 미검출 ~8%"),
            ("val/seg_loss 정체",  "마스크 경계 품질이 box 대비 낮음"),
            ("개선 방향",          "yolov8s-seg 업그레이드 또는 conf 낮추기"),
        ],
    },
]

# ── 그리기 ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 16))
ax.axis("off")
fig.patch.set_facecolor("#F5F5F5")

# 타이틀
fig.text(0.5, 0.975, "YOLOv8n-seg  학습 결과 요약",
         ha="center", va="top", fontsize=17, fontweight="bold", color="#212121")
fig.text(0.5, 0.957, "runs/segment/train  |  100 epochs  |  3 classes",
         ha="center", va="top", fontsize=10, color="#757575")

COL_W   = [0.28, 0.28, 0.22]   # 열 너비 비율 (합 < 1)
X_START = 0.05
ROW_H   = 0.032
SEC_GAP = 0.018
HDR_H   = 0.036

y = 0.925

for sec in sections:
    # 섹션 헤더
    header_rect = FancyBboxPatch(
        (X_START, y - HDR_H), sum(COL_W) + 0.12, HDR_H,
        boxstyle="round,pad=0.003", linewidth=0,
        facecolor=sec["color"], transform=fig.transFigure, clip_on=False
    )
    fig.add_artist(header_rect)
    fig.text(X_START + 0.008, y - HDR_H/2, sec["title"],
             ha="left", va="center", fontsize=11, fontweight="bold",
             color="white", transform=fig.transFigure)
    y -= HDR_H

    for i, row in enumerate(sec["rows"]):
        bg_color = "#FFFFFF" if i % 2 == 0 else "#EEEEEE"
        row_rect = FancyBboxPatch(
            (X_START, y - ROW_H), sum(COL_W) + 0.12, ROW_H,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=bg_color, transform=fig.transFigure, clip_on=False
        )
        fig.add_artist(row_rect)

        xs = [X_START + 0.008,
              X_START + COL_W[0] + 0.01,
              X_START + COL_W[0] + COL_W[1] + 0.02]

        # 항목명
        fig.text(xs[0], y - ROW_H/2, row[0],
                 ha="left", va="center", fontsize=10, color="#424242",
                 fontweight="bold", transform=fig.transFigure)
        # 값
        if len(row) >= 2:
            fig.text(xs[1], y - ROW_H/2, row[1],
                     ha="left", va="center", fontsize=10, color="#212121",
                     transform=fig.transFigure)
        # 서브값(best 등)
        if len(row) == 3:
            fig.text(xs[2], y - ROW_H/2, row[2],
                     ha="left", va="center", fontsize=9, color="#757575",
                     transform=fig.transFigure)
        y -= ROW_H

    y -= SEC_GAP

# 하단 구분선
fig.add_artist(plt.Line2D(
    [X_START, X_START + sum(COL_W) + 0.12], [y + 0.005, y + 0.005],
    color="#BDBDBD", linewidth=1, transform=fig.transFigure
))
fig.text(0.5, y - 0.01, f"* best epoch = {int(best['epoch'])}  |  기준: mAP50-95(Mask) 최대값",
         ha="center", va="top", fontsize=8.5, color="#9E9E9E")

plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {OUT_PATH}")
