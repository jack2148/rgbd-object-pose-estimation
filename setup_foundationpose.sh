#!/bin/bash
# FoundationPose 환경 설치 스크립트
# RTX 4060 Laptop (8GB VRAM) + Ubuntu + ROS2

set -e

FP_DIR="$HOME/FoundationPose"

if [ -z "${PYTHON_BIN:-}" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN=python
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
    else
        echo "ERROR: python 또는 python3 명령을 찾지 못했습니다."
        exit 1
    fi
fi
PIP_CMD="$PYTHON_BIN -m pip"

echo "Python: $($PYTHON_BIN --version)"

echo "=== [1/5] FoundationPose 클론 ==="
if [ ! -d "$FP_DIR" ]; then
    git clone https://github.com/NVlabs/FoundationPose.git "$FP_DIR"
else
    echo "이미 존재: $FP_DIR"
fi

echo "=== [2/5] Python 의존성 설치 ==="
echo "freeglut은 pip 패키지가 아닙니다. 필요하면 먼저 실행하세요:"
echo "  sudo apt update && sudo apt install -y freeglut3-dev"
$PIP_CMD install -U pip setuptools wheel
if $PYTHON_BIN -c "import torch" >/dev/null 2>&1; then
    echo "기존 PyTorch를 사용합니다."
else
    $PIP_CMD install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi
$PYTHON_BIN -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"
$PIP_CMD install trimesh pillow scipy scikit-learn open3d
$PIP_CMD install einops transformers
$PIP_CMD install PyOpenGL PyOpenGL_accelerate

echo "=== [3/5] nvdiffrast 설치 (FoundationPose 핵심 의존성) ==="
if [ -z "${CUDA_HOME:-}" ]; then
    if [ -d /usr/local/cuda ]; then
        export CUDA_HOME=/usr/local/cuda
    elif compgen -G "/usr/local/cuda-*" > /dev/null; then
        CUDA_HOME="$(compgen -G "/usr/local/cuda-*" | sort -V | tail -n 1)"
        export CUDA_HOME
    elif command -v nvcc > /dev/null 2>&1; then
        CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
        export CUDA_HOME
    fi
fi

if [ -z "${CUDA_HOME:-}" ] || [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "ERROR: CUDA toolkit(nvcc)을 찾지 못했습니다."
    echo "nvdiffrast 빌드에는 NVIDIA driver만으로는 부족하고 CUDA toolkit이 필요합니다."
    echo "확인:"
    echo "  nvidia-smi"
    echo "  nvcc --version"
    echo "설치 후 예시:"
    echo "  export CUDA_HOME=/usr/local/cuda"
    echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
    echo "  export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}"
    exit 1
fi

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "CUDA_HOME: $CUDA_HOME"
nvcc --version
$PYTHON_BIN - <<'PY'
import re
import subprocess
import sys
import torch

torch_cuda = torch.version.cuda
out = subprocess.check_output(["nvcc", "--version"], text=True)
match = re.search(r"release\s+(\d+\.\d+)", out)
nvcc_cuda = match.group(1) if match else None

print(f"PyTorch CUDA: {torch_cuda}")
print(f"nvcc CUDA: {nvcc_cuda}")

if not torch_cuda or not nvcc_cuda:
    sys.exit("ERROR: PyTorch CUDA 또는 nvcc CUDA 버전을 확인하지 못했습니다.")

torch_mm = ".".join(torch_cuda.split(".")[:2])
if torch_mm != nvcc_cuda:
    sys.exit(
        "ERROR: PyTorch CUDA 버전과 nvcc CUDA toolkit 버전이 다릅니다.\n"
        f"  PyTorch CUDA: {torch_cuda}\n"
        f"  nvcc CUDA:    {nvcc_cuda}\n"
        "nvdiffrast 같은 CUDA extension은 이 조합에서 빌드 실패할 가능성이 큽니다.\n"
        "해결: torch CUDA에 맞는 CUDA toolkit을 설치하거나, toolkit에 맞는 torch로 재설치하세요."
    )
PY
if $PYTHON_BIN -c "import pytorch3d" >/dev/null 2>&1; then
    echo "pytorch3d already installed"
else
    $PIP_CMD install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
fi
if $PYTHON_BIN -c "import nvdiffrast.torch" >/dev/null 2>&1; then
    echo "nvdiffrast already installed"
else
    $PIP_CMD install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
fi

echo "=== [4/5] FoundationPose 의존성 및 내부 확장 빌드 ==="
cd "$FP_DIR"
$PIP_CMD install -r requirements.txt
export PYTHONPATH="$FP_DIR:${PYTHONPATH:-}"
$PYTHON_BIN -c "import estimater; print('FoundationPose source import OK')"
# bundlesdf_c 빌드 (없으면 skip)
if [ -d "mycpp" ]; then
    cd mycpp && cmake -B build && cmake --build build -j$(nproc) && cd ..
fi

echo "=== [5/5] FoundationPose 사전학습 가중치 다운로드 ==="
WEIGHTS_DIR="$FP_DIR/weights"
mkdir -p "$WEIGHTS_DIR"

# gdown 설치 후 가중치 다운로드
$PIP_CMD install gdown
cd "$WEIGHTS_DIR"
# FoundationPose 공식 weights (Google Drive)
gdown --fuzzy "https://drive.google.com/drive/folders/1DFezOAD0oD8K0wL2I_SyB--MFKF0Ry3" -O . --folder 2>/dev/null || \
    echo "가중치 수동 다운로드 필요: https://github.com/NVlabs/FoundationPose#model-download"

echo ""
echo "=== 설치 완료 ==="
echo "다음 단계:"
echo "  1. CAD/ 폴더에 CAD 모델 배치 (cross.stl, cylinder.stl, hole.stl)"
echo "     기본 CAD_MESH_SCALE은 0.001입니다. CAD가 미터 단위면: export CAD_MESH_SCALE=1.0"
echo "  2. python3 foundation_pose_node.py 실행"
