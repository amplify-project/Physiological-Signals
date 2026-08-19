#!/bin/bash
# GPU-by-user snapshot for sri-gpu-12t4.
# Table 1: per-GPU x user  -> GPU util%, that user's GPU mem, summed CPU%, #procs.
# Table 2: per-user totals across ALL processes -> CPU%, MEM%, #procs.
# NOTE: `ps %cpu` is average-since-start (not instantaneous); good enough for monitoring.

GPU_CSV=$(nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader,nounits)
APP_CSV=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits)

declare -A IDX
while IFS=',' read -r idx uuid util mem; do
    IDX[${uuid// /}]=${idx// /}
done <<< "$GPU_CSV"

# per-process records: idx|user|cpu|gmem
PROC=""
while IFS=',' read -r uuid pid gmem; do
    uuid=${uuid// /}; pid=${pid// /}; gmem=${gmem// /}
    [ -z "$pid" ] && continue
    info=$(ps -o user=,%cpu= -p "$pid" 2>/dev/null)
    [ -z "$info" ] && continue
    user=$(echo "$info" | awk '{print $1}')
    cpu=$(echo "$info" | awk '{print $2}')
    PROC+="${IDX[$uuid]}|$user|$cpu|$gmem"$'\n'
done <<< "$APP_CSV"

echo "=============== GPU x USER  ($(date '+%Y-%m-%d %H:%M:%S')) ==============="
printf "%-4s %-12s %6s %10s %7s %6s\n" GPU USER GPU% GMEM_MiB CPU% PROCS
printf "%-4s %-12s %6s %10s %7s %6s\n" ---- ------------ ------ ---------- ------- ------
while IFS=',' read -r idx uuid util mem; do
    idx=${idx// /}; util=${util// /}
    rows=$(echo "$PROC" | awk -F'|' -v g="$idx" '$1==g{c[$2]+=$3; m[$2]+=$4; n[$2]++} END{for(u in c) printf "%s|%.1f|%d|%d\n",u,c[u],m[u],n[u]}')
    if [ -z "$rows" ]; then
        printf "%-4s %-12s %6s %10s %7s %6s\n" "$idx" "-" "$util" "0" "0" "0"
    else
        echo "$rows" | while IFS='|' read -r u cpu gm n; do
            printf "%-4s %-12s %6s %10s %7s %6s\n" "$idx" "$u" "$util" "$gm" "$cpu" "$n"
        done
    fi
done <<< "$GPU_CSV"

echo
echo "=============== PER-USER TOTALS (all processes) ==============="
printf "%-12s %7s %7s %6s\n" USER CPU% MEM% PROCS
printf "%-12s %7s %7s %6s\n" ------------ ------- ------- ------
ps -eo user=,%cpu=,%mem= | awk '{c[$1]+=$2; m[$1]+=$3; n[$1]++} END{for(u in c) printf "%-12s %7.1f %7.1f %6d\n", u, c[u], m[u], n[u]}' | sort -k2 -nr | head -12
