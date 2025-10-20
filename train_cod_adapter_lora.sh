#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=80000M
#SBATCH -p gpu --gres=gpu:A100:1
#SBATCH --time=2-00:00:00

echo "================= JOB START ================="
hostname
echo "Hostname: $(hostname)"

echo
echo "===== SLURM GPU ENVIRONMENT ====="
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo
echo "===== NVIDIA SYSTEM INFO ====="
nvidia-smi

echo
echo "===== PYTHON / VENV SETUP ====="
module unload python
module load python/3.11.3
echo "Python module loaded: $(python3 --version)"

rm -rf $TMPDIR/myenv
echo "Old virtual environment removed."

python3 -m venv $TMPDIR/myenv
echo "Virtual environment created."

source $TMPDIR/myenv/bin/activate
echo "Virtual environment activated."
echo "Python executable: $(which python3)"

echo
echo "===== PIP INFO ====="
pip --version
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt -q
echo "Python packages installed."
pip list

echo
echo "===== PYTORCH GPU CHECK ====="
python3 - <<EOF
import torch
print("PyTorch version:", torch.__version__)
print("PyTorch built with CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i} name:", torch.cuda.get_device_name(i))
        print(f"Device {i} capability:", torch.cuda.get_device_capability(i))
EOF


echo
echo "===== START TRAINING ====="
python3 train.py --config ./configs/cod_adapter_lora.yaml --device cuda
echo "Python script finished executing."
echo "================= JOB END ================="