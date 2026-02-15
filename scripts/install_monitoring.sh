#!/usr/bin/env bash
# Portable installer for monitoring tools: Loki, Promtail, Prometheus, Grafana.
# Works on macOS (Darwin) and Linux, for amd64 and arm64 architectures.
# Installs binaries to $HOME/.local/bin and Grafana assets to $HOME/.local/share/grafana.
set -euo pipefail

# ---------------------------------------------------------------------------
# Version pins — update these to upgrade.
# ---------------------------------------------------------------------------
LOKI_VERSION="3.3.2"
PROMETHEUS_VERSION="2.55.1"
GRAFANA_VERSION="11.4.0"

INSTALL_BIN="${INSTALL_BIN:-$HOME/.local/bin}"
INSTALL_SHARE="${INSTALL_SHARE:-$HOME/.local/share}"
TMPDIR_BASE="${TMPDIR:-/tmp}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info()  { printf "\033[1;34m==>\033[0m %s\n" "$*"; }
ok()    { printf "\033[1;32m  ✓\033[0m %s\n" "$*"; }
warn()  { printf "\033[1;33m  !\033[0m %s\n" "$*"; }
fail()  { printf "\033[1;31mERROR:\033[0m %s\n" "$*" >&2; exit 1; }

cleanup() { rm -rf "${WORK_DIR:-}"; }
trap cleanup EXIT

detect_platform() {
    local kernel arch
    kernel="$(uname -s)"
    arch="$(uname -m)"

    case "${kernel}" in
        Darwin) OS="darwin" ;;
        Linux)  OS="linux"  ;;
        *)      fail "Unsupported OS: ${kernel}" ;;
    esac

    case "${arch}" in
        x86_64|amd64)   ARCH="amd64" ;;
        aarch64|arm64)  ARCH="arm64" ;;
        *)              fail "Unsupported architecture: ${arch}" ;;
    esac

    info "Detected platform: ${OS}-${ARCH}"
}

download() {
    local url="$1" dest="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fSL --retry 3 --retry-delay 2 -o "${dest}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "${dest}" "${url}"
    else
        fail "Neither curl nor wget found. Install one of them first."
    fi
}

# Return 0 if the binary at $1 reports a version containing $2.
version_matches() {
    local bin="$1" expected="$2"
    if [ ! -x "${bin}" ]; then return 1; fi
    local actual
    actual="$("${bin}" --version 2>&1 || true)"
    case "${actual}" in
        *"${expected}"*) return 0 ;;
    esac
    return 1
}

# ---------------------------------------------------------------------------
# Installers
# ---------------------------------------------------------------------------

install_loki() {
    local ver="${LOKI_VERSION}"
    local bin_path="${INSTALL_BIN}/loki"
    if version_matches "${bin_path}" "${ver}"; then
        ok "Loki ${ver} already installed"
        return
    fi

    info "Installing Loki ${ver}..."
    local url="https://github.com/grafana/loki/releases/download/v${ver}/loki-${OS}-${ARCH}.zip"
    local archive="${WORK_DIR}/loki.zip"
    download "${url}" "${archive}"
    unzip -qo "${archive}" -d "${WORK_DIR}/loki-extract"
    # The zip contains a single binary named loki-<os>-<arch>.
    local extracted
    extracted="$(find "${WORK_DIR}/loki-extract" -type f -name 'loki*' | head -1)"
    if [ -z "${extracted}" ]; then fail "Loki binary not found in archive"; fi
    cp "${extracted}" "${bin_path}"
    chmod +x "${bin_path}"
    ok "Loki ${ver} installed to ${bin_path}"
}

install_promtail() {
    local ver="${LOKI_VERSION}"
    local bin_path="${INSTALL_BIN}/promtail"
    if version_matches "${bin_path}" "${ver}"; then
        ok "Promtail ${ver} already installed"
        return
    fi

    info "Installing Promtail ${ver}..."
    local url="https://github.com/grafana/loki/releases/download/v${ver}/promtail-${OS}-${ARCH}.zip"
    local archive="${WORK_DIR}/promtail.zip"
    download "${url}" "${archive}"
    unzip -qo "${archive}" -d "${WORK_DIR}/promtail-extract"
    local extracted
    extracted="$(find "${WORK_DIR}/promtail-extract" -type f -name 'promtail*' | head -1)"
    if [ -z "${extracted}" ]; then fail "Promtail binary not found in archive"; fi
    cp "${extracted}" "${bin_path}"
    chmod +x "${bin_path}"
    ok "Promtail ${ver} installed to ${bin_path}"
}

install_prometheus() {
    local ver="${PROMETHEUS_VERSION}"
    local bin_path="${INSTALL_BIN}/prometheus"
    if version_matches "${bin_path}" "${ver}"; then
        ok "Prometheus ${ver} already installed"
        return
    fi

    info "Installing Prometheus ${ver}..."
    local url="https://github.com/prometheus/prometheus/releases/download/v${ver}/prometheus-${ver}.${OS}-${ARCH}.tar.gz"
    local archive="${WORK_DIR}/prometheus.tar.gz"
    download "${url}" "${archive}"
    tar -xzf "${archive}" -C "${WORK_DIR}"
    local extract_dir="${WORK_DIR}/prometheus-${ver}.${OS}-${ARCH}"
    if [ ! -d "${extract_dir}" ]; then fail "Prometheus extract dir not found: ${extract_dir}"; fi
    cp "${extract_dir}/prometheus" "${bin_path}"
    chmod +x "${bin_path}"
    # Also copy promtool if present.
    if [ -f "${extract_dir}/promtool" ]; then
        cp "${extract_dir}/promtool" "${INSTALL_BIN}/promtool"
        chmod +x "${INSTALL_BIN}/promtool"
    fi
    ok "Prometheus ${ver} installed to ${bin_path}"
}

install_grafana() {
    local ver="${GRAFANA_VERSION}"
    local grafana_home="${INSTALL_SHARE}/grafana"
    local bin_path="${INSTALL_BIN}/grafana-server"
    if version_matches "${bin_path}" "${ver}"; then
        ok "Grafana ${ver} already installed"
        return
    fi

    info "Installing Grafana ${ver}..."
    local url="https://dl.grafana.com/oss/release/grafana-${ver}.${OS}-${ARCH}.tar.gz"
    local archive="${WORK_DIR}/grafana.tar.gz"
    download "${url}" "${archive}"
    tar -xzf "${archive}" -C "${WORK_DIR}"
    # The tarball extracts to grafana-v${ver}/ or grafana-${ver}/.
    local extract_dir
    extract_dir="$(find "${WORK_DIR}" -maxdepth 1 -type d -name 'grafana*' | grep -v '\.tar' | head -1)"
    if [ -z "${extract_dir}" ] || [ ! -d "${extract_dir}/public" ]; then
        fail "Grafana public/ assets not found in archive"
    fi

    # Replace the previous install.
    rm -rf "${grafana_home}"
    mkdir -p "${grafana_home}"
    cp -R "${extract_dir}/"* "${grafana_home}/"

    # Symlink grafana-server (legacy) into the bin directory.
    local server_bin="${grafana_home}/bin/grafana-server"
    if [ ! -f "${server_bin}" ]; then
        server_bin="${grafana_home}/bin/grafana"
    fi
    if [ ! -f "${server_bin}" ]; then fail "grafana-server binary not found in distribution"; fi
    ln -sf "${server_bin}" "${bin_path}"
    # Symlink the modern 'grafana' binary (not the deprecated wrapper).
    local grafana_main="${grafana_home}/bin/grafana"
    if [ -f "${grafana_main}" ]; then
        ln -sf "${grafana_main}" "${INSTALL_BIN}/grafana"
    else
        ln -sf "${server_bin}" "${INSTALL_BIN}/grafana"
    fi
    ok "Grafana ${ver} installed to ${grafana_home}"
}

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

verify_all() {
    info "Verifying installations..."
    local failed=0

    for tool in loki promtail prometheus grafana-server; do
        local p="${INSTALL_BIN}/${tool}"
        if [ -x "${p}" ]; then
            local ver
            ver="$("${p}" --version 2>&1 | head -1 || true)"
            ok "${tool}: ${ver}"
        else
            warn "${tool}: NOT FOUND at ${p}"
            failed=1
        fi
    done

    if [ "${failed}" -eq 1 ]; then
        fail "Some tools failed to install. Check the output above."
    fi

    info "All monitoring tools installed successfully."
    echo ""
    echo "  Binaries:      ${INSTALL_BIN}"
    echo "  Grafana home:  ${INSTALL_SHARE}/grafana"
    echo ""
    echo "  Make sure ${INSTALL_BIN} is in your PATH:"
    echo "    export PATH=\"${INSTALL_BIN}:\${PATH}\""
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    WORK_DIR="$(mktemp -d "${TMPDIR_BASE}/monitoring-install.XXXXXX")"
    detect_platform
    mkdir -p "${INSTALL_BIN}" "${INSTALL_SHARE}"

    install_loki
    install_promtail
    install_prometheus
    install_grafana
    verify_all
}

main "$@"
