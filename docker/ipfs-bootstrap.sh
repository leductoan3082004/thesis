#!/bin/sh

set -eu

export IPFS_PATH="${IPFS_PATH:-/data/ipfs}"

SELF_NODE=${NODE_NAME:?NODE_NAME env must be set}
PEER_REGISTRY=${PEER_REGISTRY:-/peers}
CLUSTER_NODES=${IPFS_CLUSTER_NODES:-}
mkdir -p "$PEER_REGISTRY"

echo "[*] ($SELF_NODE) Preparing IPFS repo at $IPFS_PATH"

if [ ! -f "$IPFS_PATH/config" ]; then
    ipfs init --empty-repo
fi

# Lock down the repo so it stays private.
ipfs config --json AutoConf.Enabled false >/dev/null 2>&1 || true
ipfs config --json Discovery.MDNS.Enabled false >/dev/null 2>&1 || true
ipfs config --json Bootstrap '[]' >/dev/null 2>&1 || true
ipfs config Routing.Type dhtserver >/dev/null 2>&1 || true
ipfs config Reprovider.Interval "5s" >/dev/null 2>&1 || true
ipfs config Reprovider.Strategy "all" >/dev/null 2>&1 || true

# Configure API to listen on all interfaces for external access.
ipfs config Addresses.API "/ip4/0.0.0.0/tcp/5001" >/dev/null 2>&1 || true
ipfs config Addresses.Gateway "/ip4/0.0.0.0/tcp/8080" >/dev/null 2>&1 || true

# Publish this node's multiaddr for other peers.
PEER_ID=$(ipfs config Identity.PeerID | tr -d '"')
SELF_MULTIADDR="/dns4/${SELF_NODE}/tcp/4001/p2p/${PEER_ID}"
echo "$SELF_MULTIADDR" > "${PEER_REGISTRY}/${SELF_NODE}.addr"
echo "[*] ($SELF_NODE) Published peer info: $SELF_MULTIADDR"

# Build bootstrap list from discovered peers.
BOOTSTRAP_JSON="[]"
if [ -n "$CLUSTER_NODES" ]; then
    JSON="["
    FIRST=1
    OLD_IFS=$IFS
    IFS=,
    for NODE in $CLUSTER_NODES; do
        if [ "$NODE" = "$SELF_NODE" ] || [ -z "$NODE" ]; then
            continue
        fi
        PEER_FILE="${PEER_REGISTRY}/${NODE}.addr"
        ATTEMPTS=0
        while [ ! -s "$PEER_FILE" ] && [ $ATTEMPTS -lt 60 ]; do
            echo "[*] ($SELF_NODE) Waiting for peer info from $NODE..."
            sleep 1
            ATTEMPTS=$((ATTEMPTS + 1))
        done
        if [ ! -s "$PEER_FILE" ]; then
            echo "[!] ($SELF_NODE) Peer info for $NODE not found; continuing without it"
            continue
        fi
        ADDR=$(cat "$PEER_FILE")
        if [ $FIRST -eq 1 ]; then
            FIRST=0
        else
            JSON="$JSON,"
        fi
        JSON="$JSON\"$ADDR\""
    done
    IFS=$OLD_IFS
    JSON="$JSON]"
    BOOTSTRAP_JSON="$JSON"
fi

ipfs config --json Bootstrap "$BOOTSTRAP_JSON" >/dev/null 2>&1 || true
echo "[*] ($SELF_NODE) Bootstrap peers set to: $BOOTSTRAP_JSON"

echo "[*] ($SELF_NODE) Starting IPFS daemon..."
exec ipfs daemon --migrate=true
