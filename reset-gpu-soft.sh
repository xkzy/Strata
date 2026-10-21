#!/usr/bin/env bash
# Soft recovery for a GPU that fell off the bus, for boards with no ACPI slot power control:
# stop GPU users, unload the NVIDIA modules, remove the dead card from the PCI tree, rescan, reload.
# Run from a text console or over SSH:  sudo bash reset-gpu-soft.sh [0000:01:00.0]
# If the card still is not back, only a reboot (or a full power-off) will bring it back.
set -u
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
GPU=${1:-0000:01:00.0}
BASE=${GPU%.*}
echo "== stopping the Strata server and GPU services"
for p in $(ps -eo pid,comm | awk '$2 ~ /^strata/ {print $1}'); do kill "$p" 2>/dev/null; done
sleep 3
for u in display-manager nvidia-persistenced nvidia-powerd; do systemctl stop "$u" 2>/dev/null; done
sleep 2
fuser -k /dev/nvidia* /dev/dri/* 2>/dev/null; sleep 1
echo "== unloading modules"
for m in nvidia_drm nvidia_modeset nvidia_uvm nvidia; do modprobe -r "$m" 2>&1 | sed "s/^/   $m: /"; done
lsmod | grep -q '^nvidia ' && echo "   nvidia is still loaded: a process still holds it ($(fuser -v /dev/nvidia* 2>&1 | head -3 | tr '\n' ' '))"
echo "== removing and rescanning $BASE.*"
for f in /sys/bus/pci/devices/$BASE.*; do [ -e "$f" ] && echo 1 > "$f/remove"; done
sleep 3
echo 1 > /sys/bus/pci/rescan
sleep 5
echo "== link: $(cat /sys/bus/pci/devices/$GPU/current_link_speed 2>/dev/null) x$(cat /sys/bus/pci/devices/$GPU/current_link_width 2>/dev/null)"
modprobe nvidia && modprobe nvidia_uvm && modprobe nvidia_modeset && modprobe nvidia_drm modeset=1
systemctl start nvidia-persistenced 2>/dev/null
systemctl start display-manager 2>/dev/null
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader || echo "GPU still not answering: reboot, or power off for 30 s."
