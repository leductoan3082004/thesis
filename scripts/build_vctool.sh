#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLOCKCHAIN_DIR="${BLOCKCHAIN_DIR:-$ROOT_DIR/../thesis-blockchain/api-gateway}"
API_DIR="$BLOCKCHAIN_DIR/api"
OUTPUT="${API_DIR}/vctool"

err() {
	echo "[build-vctool] $*" >&2
}

require_cmd() {
	if ! command -v "$1" >/dev/null 2>&1; then
		err "Missing dependency: $1"
		exit 1
	fi
}

detect_goos() {
	case "$(uname -s | tr '[:upper:]' '[:lower:]')" in
		linux*) echo "linux" ;;
		darwin*) echo "darwin" ;;
		*)
			err "Unsupported platform: $(uname -s)"
			exit 1
			;;
	esac
}

detect_goarch() {
	case "$(uname -m)" in
		x86_64|amd64) echo "amd64" ;;
		arm64|aarch64) echo "arm64" ;;
		*)
			err "Unsupported architecture: $(uname -m)"
			exit 1
			;;
	esac
}

main() {
	require_cmd go
	if [ ! -d "$API_DIR" ]; then
		err "Blockchain api-gateway repo not found at $BLOCKCHAIN_DIR"
		exit 1
	fi
	local goos goarch
	goos="${GOOS:-$(detect_goos)}"
	goarch="${GOARCH:-$(detect_goarch)}"
	err "Building vctool for ${goos}/${goarch}..."
	GOOS="$goos" GOARCH="$goarch" go build -o "$OUTPUT" ./cmd/vctool
	err "vctool binary written to $OUTPUT"
}

(
	cd "$API_DIR"
	main "$@"
)
