#!/usr/bin/env bash
set -euo pipefail

FABRIC_VERSION="${FABRIC_VERSION:-2.5.6}"
FABRIC_CA_VERSION="${FABRIC_CA_VERSION:-1.5.9}"
INSTALL_PREFIX="${FABRIC_INSTALL_PREFIX:-$HOME/.local}"
BIN_DIR="${FABRIC_BIN_DIR:-$INSTALL_PREFIX/bin}"

err() {
	echo "[install-fabric] $*" >&2
}

require_cmd() {
	if ! command -v "$1" >/dev/null 2>&1; then
		err "Missing dependency: $1"
		exit 1
	}
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

download_and_extract() {
	local url="$1"
	local archive="$2"
	local dest="$3"
	err "Downloading ${url}"
	curl -L "$url" -o "$archive"
	tar -xzf "$archive" -C "$dest"
}

main() {
	require_cmd curl
	require_cmd tar

	local platform arch tmpdir fabric_tar fabric_url ca_tar ca_url
	platform="$(detect_platform)"
	arch="$(detect_arch)"
	tmpdir="$(mktemp -d)"
	trap 'rm -rf "$tmpdir"' EXIT

	fabric_tar="${tmpdir}/fabric.tgz"
	fabric_url="https://github.com/hyperledger/fabric/releases/download/v${FABRIC_VERSION}/hyperledger-fabric-${platform}-${arch}-${FABRIC_VERSION}.tar.gz"
	ca_tar="${tmpdir}/fabric-ca.tgz"
	ca_url="https://github.com/hyperledger/fabric-ca/releases/download/v${FABRIC_CA_VERSION}/hyperledger-fabric-ca-${platform}-${arch}-${FABRIC_CA_VERSION}.tar.gz"

	download_and_extract "$fabric_url" "$fabric_tar" "$tmpdir"
	download_and_extract "$ca_url" "$ca_tar" "$tmpdir"

	if [ ! -d "${tmpdir}/bin" ]; then
		err "Extracted archive is missing bin/ directory"
		exit 1
	fi

	mkdir -p "$BIN_DIR"
	for binary in "${tmpdir}"/bin/*; do
		[ -f "$binary" ] || continue
		cp "$binary" "$BIN_DIR/"
	done
	chmod +x "${BIN_DIR}/"*

	err "Installed Hyperledger Fabric binaries -> ${BIN_DIR}"
	err "Make sure ${BIN_DIR} is on your PATH."
	"${BIN_DIR}/cryptogen" version
	"${BIN_DIR}/fabric-ca-client" version
}

main "$@"
