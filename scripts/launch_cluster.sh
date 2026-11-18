#!/usr/bin/env bash
# Distributed runs for Tables 9-12 on a CPU cluster: one process per worker node,
# gloo over Ethernet, static rendezvous.  Runs the given config at every world size.
#
#   HOSTFILE=hosts.txt ./scripts/launch_cluster.sh configs/hpc_products.toml [--depth 30 --model gcnii ...]
#
# hosts.txt: one worker host per line (blank lines and lines starting with # are ignored);
# the first host is rank 0 and the rendezvous point.  Every host needs passwordless ssh, the
# same Python environment (activated by ENV_ACTIVATE) and a local copy of the repository and
# datasets at REPO.  Extra arguments are joined with spaces and parsed again by the shell on
# the node, so an argument containing a space needs an extra level of quotes.
#
# Environment: HOSTFILE (required), ENV_ACTIVATE (/opt/cluster-env/bin/activate), REPO ($PWD),
# PORT (29500), WORLD_SIZES ("1 2 4 8 14"; a size larger than the number of listed hosts is
# skipped), GLOO_SOCKET_IFNAME (unset: gloo picks the interface; set it to the cluster's network
# interface, e.g. eno1, when a node's hostname resolves to a loopback address).  A failing rank
# stops this script at once; the other ranks are left to finish or to time out on their own.
set -euo pipefail

CONFIG=${1:?usage: launch_cluster.sh CONFIG.toml [extra nhdp_gnn.distributed args]}
shift
HOSTFILE=${HOSTFILE:?set HOSTFILE to a file with one worker host per line}
ENV_ACTIVATE=${ENV_ACTIVATE:-/opt/cluster-env/bin/activate}
REPO=${REPO:-$(pwd)}
PORT=${PORT:-29500}
WORLD_SIZES=${WORLD_SIZES:-"1 2 4 8 14"}
NIC_EXPORT=""
if [[ -n "${GLOO_SOCKET_IFNAME:-}" ]]; then
  NIC_EXPORT="export GLOO_SOCKET_IFNAME='$GLOO_SOCKET_IFNAME' && "
fi

[[ -r "$HOSTFILE" ]] || { echo "cannot read HOSTFILE=$HOSTFILE" >&2; exit 2; }
mapfile -t ALL_HOSTS < <(sed -e 's/\r$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' "$HOSTFILE" \
                         | grep -v -e '^$' -e '^#' || true)
(( ${#ALL_HOSTS[@]} > 0 )) || { echo "no hosts listed in $HOSTFILE" >&2; exit 2; }

ran=0
for WORLD in $WORLD_SIZES; do
  if (( WORLD > ${#ALL_HOSTS[@]} )); then
    echo "skip world size $WORLD: only ${#ALL_HOSTS[@]} hosts listed" >&2
    continue
  fi
  MASTER=${ALL_HOSTS[0]}
  echo "=== world size $WORLD (master $MASTER) ==="
  pids=()
  for (( i = 0; i < WORLD; i++ )); do
    ssh "${ALL_HOSTS[$i]}" "source '$ENV_ACTIVATE' && ${NIC_EXPORT}cd '$REPO' && \
      torchrun --nnodes $WORLD --nproc-per-node 1 --node-rank $i \
               --master-addr $MASTER --master-port $PORT \
               -m nhdp_gnn.distributed --config '$CONFIG' $*" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  ran=$((ran + 1))
done
(( ran > 0 )) || { echo "nothing ran: no world size in '$WORLD_SIZES' fits the ${#ALL_HOSTS[@]} listed hosts" >&2; exit 1; }
echo "results: python -m nhdp_gnn.report runs/hpc --scaling"
