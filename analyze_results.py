"""
analyze_results.py
학습 결과 시각화 스크립트

생성 파일:
  training_curves.png         -- train/val loss + mAP 학습 곡선
  training_best_metrics.png   -- per-class 최고 성능 막대그래프 (YOLO val 기반)
  training_summary_table.png  -- 최종 지표 요약 표
"""

import csv
import subprocess
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ultralytics import YOLO

CSV_PATH    = "runs/segment/train/results.csv"
MODEL_PATH  = "runs/segment/train/weights/best.pt"
DATA_YAML   = "object/data.yaml"
CLASS_NAMES = ["cylinder", "hole", "cross"]
COLORS      = {"cylinder": "#55A868", "hole": "#C44E52", "cross": "#4C72B0"}

# ── CSV 로드 ──────────────────────────────────────────────────────────────────
def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [{k.strip(): v.strip() for k, v in r.items()} for r in rows]

def get(rows, key):
    return [float(r.get(key, 0) or 0) for r in rows]

rows = load_csv(CSV_PATH)
epochs = [int(r["epoch"]) for r in rows]

# ── 1. 학습 곡선 ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("YOLOv8n-seg Training Curves", fontsize=15, fontweight="bold")

plots = [
    ("metrics/mAP50(M)",     "mAP50 (Mask)",    axes[0, 0], (0, 1.05)),
    ("metrics/mAP50-95(M)",  "mAP50-95 (Mask)", axes[0, 1], (0, 1.05)),
    ("val/seg_loss",         "Val Seg Loss",     axes[1, 0], None),
    ("val/box_loss",         "Val Box Loss",     axes[1, 1], None),
]

train_color = "#2196F3"
val_color   = "#F44336"

for key, title, ax, ylim in plots:
    vals = get(rows, key)
    is_loss = "loss" in title.lower()

    if is_loss:
        # train/val 비교
        train_key = key.replace("val/", "train/")
        train_vals = get(rows, train_key)
        ax.plot(epochs, train_vals, color=train_color, linewidth=2, label="train")
        ax.plot(epochs, vals,       color=val_color,   linewidth=2, label="val", linestyle="--")
        ax.legend(fontsize=9)
    else:
        ax.plot(epochs, vals, color=train_color, linewidth=2)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)

fig.tight_layout()
fig.savefig("training_curves.png", dpi=150, bbox_inches="tight")
print("저장: training_curves.png")

# ── 2. YOLO validation → per-class 성능 ──────────────────────────────────────
print("\nYOLO validation 실행 중...")
model = YOLO(MODEL_PATH)
val_results = model.val(data=DATA_YAML, verbose=False)

# per-class metrics 추출
# val_results.box.ap50  shape: (n_classes,)
# val_results.seg.ap50  shape: (n_classes,)
box = val_results.box
seg = val_results.seg

per_class = {}
for i, name in enumerate(CLASS_NAMES):
    per_class[name] = {
        "mAP50 (Box)":    float(box.ap50[i]),
        "mAP50 (Mask)":   float(seg.ap50[i]),
        "Precision":      float(box.p[i]),
        "Recall":         float(box.r[i]),
    }

print("Per-class 결과:")
for name, m in per_class.items():
    print(f"  {name:10s}  mAP50(M)={m['mAP50 (Mask)']:.4f}  P={m['Precision']:.4f}  R={m['Recall']:.4f}")

# ── 3. 막대 그래프 ────────────────────────────────────────────────────────────
metric_keys = ["mAP50 (Box)", "mAP50 (Mask)", "Precision", "Recall"]
x     = np.arange(len(metric_keys))
width = 0.25

fig_bar, ax_b = plt.subplots(figsize=(11, 5))
for i, name in enumerate(CLASS_NAMES):
    vals = [per_class[name][k] for k in metric_keys]
    bars = ax_b.bar(x + i * width, vals, width,
                    label=name, color=COLORS[name], alpha=0.85)
    for bar, val in zip(bars, vals):
        ax_b.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.005,
                  f"{val:.3f}", ha="center", va="bottom", fontsize=8)

ax_b.set_xticks(x + width)
ax_b.set_xticklabels(metric_keys, fontsize=12)
ax_b.set_ylim(0, 1.15)
ax_b.set_ylabel("Score")
ax_b.set_title("Best Metrics per Class", fontsize=13, fontweight="bold")
ax_b.legend()
ax_b.grid(True, axis="y", alpha=0.3)
fig_bar.tight_layout()
fig_bar.savefig("training_best_metrics.png", dpi=150, bbox_inches="tight")
print("저장: training_best_metrics.png")

# ── 4. 요약 표 ────────────────────────────────────────────────────────────────
col_labels = ["Class", "mAP50 (Box)", "mAP50 (Mask)", "Precision", "Recall"]
table_data = [
    [name,
     f"{per_class[name]['mAP50 (Box)']:.4f}",
     f"{per_class[name]['mAP50 (Mask)']:.4f}",
     f"{per_class[name]['Precision']:.4f}",
     f"{per_class[name]['Recall']:.4f}"]
    for name in CLASS_NAMES
]

# 전체 평균 행
means = [
    np.mean([per_class[n][k] for n in CLASS_NAMES])
    for k in ["mAP50 (Box)", "mAP50 (Mask)", "Precision", "Recall"]
]
table_data.append(["mean (all)"] + [f"{v:.4f}" for v in means])

fig_t, ax_t = plt.subplots(figsize=(10, 3))
ax_t.axis("off")
table = ax_t.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.2)

# 헤더 색
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

row_colors = [COLORS[n] for n in CLASS_NAMES] + ["#CCCCCC"]
for i, color in enumerate(row_colors, start=1):
    for j in range(len(col_labels)):
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        light = f"#{min(r+160,255):02x}{min(g+160,255):02x}{min(b+160,255):02x}"
        table[i, j].set_facecolor(light)

fig_t.suptitle("YOLOv8n-seg  Validation Results  (cylinder / hole / cross)",
               fontsize=13, fontweight="bold", y=0.98)
fig_t.tight_layout()
fig_t.savefig("training_summary_table.png", dpi=150, bbox_inches="tight")
print("저장: training_summary_table.png")

plt.show()
