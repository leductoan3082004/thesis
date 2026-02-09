#!/bin/sh

set -eu

export IPFS_PATH="${IPFS_PATH:-/data/ipfs}"

SELF_NODE=${NODE_NAME:?NODE_NAME env must be set}
PEER_REGISTRY=${PEER_REGISTRY:-/peers}
CLUSTER_NODES=${IPFS_CLUSTER_NODES:-}
IPFS_CMD=${IPFS_BINARY:-ipfs}
API_HOST=${API_HOST:-0.0.0.0}
API_PORT=${API_PORT:-5001}
GATEWAY_HOST=${GATEWAY_HOST:-0.0.0.0}
GATEWAY_PORT=${GATEWAY_PORT:-8080}
SWARM_HOST=${SWARM_HOST:-0.0.0.0}
SWARM_PORT=${SWARM_PORT:-4001}
SWARM_IPV6_HOST=${SWARM_IPV6_HOST:-}
NODE_ADVERTISE_PROTO=${NODE_ADVERTISE_PROTO:-dns4}
NODE_ADVERTISE_HOST=${NODE_ADVERTISE_HOST:-$SELF_NODE}
NODE_ADVERTISE_ADDR=${NODE_ADVERTISE_ADDR:-}
mkdir -p "$PEER_REGISTRY"

echo "[*] ($SELF_NODE) Preparing IPFS repo at $IPFS_PATH"

if [ ! -f "$IPFS_PATH/config" ]; then
    "$IPFS_CMD" init --empty-repo
fi

# Lock down the repo so it stays private.
"$IPFS_CMD" config --json AutoConf.Enabled false >/dev/null 2>&1 || true
"$IPFS_CMD" config --json Discovery.MDNS.Enabled false >/dev/null 2>&1 || true
"$IPFS_CMD" config --json Bootstrap '[]' >/dev/null 2>&1 || true
"$IPFS_CMD" config Routing.Type dhtserver >/dev/null 2>&1 || true
"$IPFS_CMD" config Provide.Strategy "all" >/dev/null 2>&1 || true
"$IPFS_CMD" config Provide.DHT.Interval "5s" >/dev/null 2>&1 || true
# Clear legacy Reprovider keys so new Kubo builds do not abort.
"$IPFS_CMD" config --json Reprovider '{}' >/dev/null 2>&1 || true

# Configure API to listen on all interfaces for external access.
"$IPFS_CMD" config Addresses.API "/ip4/${API_HOST}/tcp/${API_PORT}" >/dev/null 2>&1 || true
"$IPFS_CMD" config Addresses.Gateway "/ip4/${GATEWAY_HOST}/tcp/${GATEWAY_PORT}" >/dev/null 2>&1 || true
SWARM_ENTRIES=""
if [ -n "$SWARM_HOST" ]; then
    SWARM_ENTRIES="\"/ip4/${SWARM_HOST}/tcp/${SWARM_PORT}\""
fi
if [ -n "$SWARM_IPV6_HOST" ]; then
    if [ -n "$SWARM_ENTRIES" ]; then
        SWARM_ENTRIES="${SWARM_ENTRIES},"
    fi
    SWARM_ENTRIES="${SWARM_ENTRIES}\"/ip6/${SWARM_IPV6_HOST}/tcp/${SWARM_PORT}\""
fi
if [ -z "$SWARM_ENTRIES" ]; then
    SWARM_ENTRIES="\"/ip4/0.0.0.0/tcp/${SWARM_PORT}\""
fi
"$IPFS_CMD" config --json Addresses.Swarm "[${SWARM_ENTRIES}]" >/dev/null 2>&1 || true

# Publish this node's multiaddr for other peers.
PEER_ID=$("$IPFS_CMD" config Identity.PeerID | tr -d '"')
if [ -n "$NODE_ADVERTISE_ADDR" ]; then
    SELF_MULTIADDR="${NODE_ADVERTISE_ADDR}/tcp/${SWARM_PORT}/p2p/${PEER_ID}"
else
    SELF_MULTIADDR="/${NODE_ADVERTISE_PROTO}/${NODE_ADVERTISE_HOST}/tcp/${SWARM_PORT}/p2p/${PEER_ID}"
fi
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
"$IPFS_CMD" config --json Bootstrap "$BOOTSTRAP_JSON" >/dev/null 2>&1 || true
echo "[*] ($SELF_NODE) Bootstrap peers set to: $BOOTSTRAP_JSON"

echo "[*] ($SELF_NODE) Starting IPFS daemon..."
exec "$IPFS_CMD" daemon --migrate=true
