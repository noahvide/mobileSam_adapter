#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=80000M
# we run on the gpu partition and we allocate 1 gpu
#SBATCH -p gpu --gres=gpu:titanrtx:1
# We expect that our program should not run longer than 2 days
# Note that a program will be killed once it exceeds this time!
#SBATCH --time=2-00:00:00

# Print the hostname and the IDs of the chosen GPUs.
hostname
echo "Hostname: $(hostname)"
echo "GPU IDs allocated:"
echo "$CUDA_VISIBLE_DEVICES"
nvidia-smi

# Load the appropriate Python module
echo "Loading Python module..."
module load python/3.11.3
echo "Python module loaded."

# Remove any existing virtual environment
echo "Removing existing virtual environment..."
rm -rf $TMPDIR/myenv
echo "Existing virtual environment removed."

# Create a Python virtual environment in the job's temporary directory
echo "Creating Python virtual environment..."
python3 -m venv $TMPDIR/myenv
echo "Virtual environment created."

# Activate the virtual environment
echo "Activating the virtual environment..."
source $TMPDIR/myenv/bin/activate
echo "Virtual environment activated."

# Install required Python packages from requirements.txt
echo "Installing Python packages..."
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt -q
echo "Python packages installed."

# Execute the Python script
echo "Starting Python script..."
python3 train.py --config ./configs/cod_lora.yaml --device cuda
echo "Python script finished executing."
