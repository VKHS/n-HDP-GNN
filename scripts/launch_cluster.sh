#!/usr/bin/env bash
# Distributed runs for Tables 9-12 on a CPU cluster: one process per worker node,
# gloo over Ethernet, static rendezvous.  Runs the given config at every world size.
#
#   HOSTFILE=hosts.txt ./scripts/launch_cluster.sh configs/hpc_products.toml [--depth 30 --model gcnii ...]
#
# hosts.txt: one worker host per line; the first host is rank 0 and the rendezvous
# point.  Every host needs passwordless ssh, the same Python environment (activated
# by ENV_ACTIVATE) and a local copy of the repository and datasets at REPO.
set -euo pipefail

CONFIG=${1:?usage: launch_cluster.sh CONFIG.toml [extra nhdp_gnn.distributed args]}
shift
HOSTFILE=${HOSTFILE:?set HOSTFILE to a file with one worker host per line}
ENV_ACTIVATE=${ENV_ACTIVATE:-/opt/cluster-env/bin/activate}
REPO=${REPO:-$(pwd)}
PORT=${PORT:-29500}
WORLD_SIZES=${WORLD_SIZES:-"1 2 4 8 14"}

mapfile -t ALL_HOSTS < <(grep -v '^\s*$' "$HOSTFILE")
for WORLD in $WORLD_SIZES; do
  if (( WORLD > ${#ALL_HOSTS[@]} )); then
    echo "skip world size $WORLD: only ${#ALL_HOSTS[@]} hosts listed" >&2
    continue
  fi
  MASTER=${ALL_HOSTS[0]}
  echo "=== world size $WORLD (master $MASTER) ==="
  pids=()
  for (( i = 0; i < WORLD; i++ )); do
    ssh "${ALL_HOSTS[$i]}" "source '$ENV_ACTIVATE' && cd '$REPO' && \
      torchrun --nnodes $WORLD --nproc-per-node 1 --node-rank $i \
               --master-addr $MASTER --master-port $PORT \
               -m nhdp_gnn.distributed --config '$CONFIG' $*" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done
echo "results: python -m nhdp_gnn.report runs/hpc --scaling"
