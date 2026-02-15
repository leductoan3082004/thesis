#!/usr/bin/env bash
set -euo pipefail

VERSION="${IPFS_VERSION:-v0.30.0}"
INSTALL_PREFIX="${IPFS_INSTALL_PREFIX:-$HOME/.local}"
BIN_DIR="${IPFS_BIN_DIR:-$INSTALL_PREFIX/bin}"

err() {
	echo "[install-ipfs] $*" >&2
}

require_cmd() {
	if ! command -v "$1" >/dev/null 2>&1; then
		err "Missing dependency: $1"
		exit 1
	fi
}

detect_platform() {
	case "$(uname -s | tr '[:upper:]' '[:lower:]')" in
		linux*) echo "linux" ;;
		darwin*) echo "darwin" ;;
		*)
			err "Unsupported platform: $(uname -s)"
			exit 1
			;;
	esac
}

detect_arch() {
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
	require_cmd curl
	require_cmd tar

	local platform arch tarball url tmpdir
	platform="$(detect_platform)"
	arch="$(detect_arch)"
	tarball="kubo_${VERSION}_${platform}-${arch}.tar.gz"
	url="https://dist.ipfs.tech/kubo/${VERSION}/${tarball}"
	tmpdir="$(mktemp -d)"
	trap 'rm -rf "$tmpdir"' EXIT

	err "Downloading ${url}"
	curl -L "$url" -o "${tmpdir}/${tarball}"
	tar -xzf "${tmpdir}/${tarball}" -C "$tmpdir"

	mkdir -p "$BIN_DIR"
	cp "${tmpdir}/kubo/ipfs" "${BIN_DIR}/ipfs"
	chmod +x "${BIN_DIR}/ipfs"

	err "Installed ipfs -> ${BIN_DIR}/ipfs"
	err "Make sure ${BIN_DIR} is on your PATH."
	"${BIN_DIR}/ipfs" --version
}

main "$@"
