#!/bin/sh

set -eu

export IPFS_PATH="${IPFS_PATH:-/data/ipfs}"

SELF_NODE=${NODE_NAME:?NODE_NAME env must be set}
BOOTSTRAP_PEERS=${BOOTSTRAP_PEERS:-}

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

# Fill Bootstrap list with static peers if provided.
if [ -n "$BOOTSTRAP_PEERS" ]; then
    JSON="["
    FIRST=1
    OLD_IFS=$IFS
    IFS=,
    for ADDR in $BOOTSTRAP_PEERS; do
        if [ -z "$ADDR" ]; then
            continue
        fi
        if [ $FIRST -eq 1 ]; then
            FIRST=0
        else
            JSON="$JSON,"
        fi
        JSON="$JSON\"$ADDR\""
    done
    IFS=$OLD_IFS
    JSON="$JSON]"
    ipfs config --json Bootstrap "$JSON" >/dev/null 2>&1 || true
    echo "[*] ($SELF_NODE) Configured bootstrap peers: $BOOTSTRAP_PEERS"
else
    echo "[*] ($SELF_NODE) No static bootstrap peers configured."
fi

echo "[*] ($SELF_NODE) Starting IPFS daemon..."
exec ipfs daemon --migrate=true
