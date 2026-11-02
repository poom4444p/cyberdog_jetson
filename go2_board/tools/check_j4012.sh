#!/bin/bash
# check_j4012.sh -- what this J4012 has, read-only. Changes nothing.
#
#   bash check_j4012.sh | tee j4012_report.txt
#
# Paste the output back: it decides whether to flash, and what to install.

section() { echo; echo "== $1"; }

section "board and JetPack"
cat /proc/device-tree/model 2>/dev/null; echo
cat /etc/nv_tegra_release 2>/dev/null || echo "no /etc/nv_tegra_release"
dpkg-query -W -f='${Package} ${Version}\n' nvidia-jetpack nvidia-l4t-core 2>/dev/null

section "OS and kernel"
grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release
uname -r
uname -v | grep -o PREEMPT_RT || echo "kernel: not PREEMPT_RT"

section "CPU, memory, disk"
nproc
free -h | head -2
df -h / | tail -1

section "CUDA and GPU"
ls -d /usr/local/cuda* 2>/dev/null || echo "no /usr/local/cuda"
/usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -1

section "Python"
python3 --version
command -v conda >/dev/null && conda --version || echo "no conda"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>/dev/null \
    || echo "no torch in system python3"

section "ROS 2"
ls /opt/ros 2>/dev/null || echo "no /opt/ros"

section "network (Mid-360 and the motor side go on Ethernet)"
ip -br addr

section "USB devices (camera, ESP32 handle, RS-485 adapters)"
lsusb 2>/dev/null || echo "no lsusb"
ls /dev/video* /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "no video or serial devices now"

section "CAN"
ip -br link show type can 2>/dev/null || true
ls /sys/class/net | grep -i can || echo "no CAN interface"

section "power mode"
nvpmodel -q 2>/dev/null | head -2 || echo "nvpmodel not available"
