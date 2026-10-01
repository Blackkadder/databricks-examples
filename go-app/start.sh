#!/bin/bash
set -e

GO_VERSION="1.22.4"
GO_TAR="go${GO_VERSION}.linux-amd64.tar.gz"
GO_URL="https://go.dev/dl/${GO_TAR}"
INSTALL_DIR="/tmp/go-install"

# Download and install Go to a temp directory
echo "Downloading Go ${GO_VERSION}..."
mkdir -p "${INSTALL_DIR}"
curl -fsSL "${GO_URL}" -o "/tmp/${GO_TAR}"
tar -C "${INSTALL_DIR}" -xzf "/tmp/${GO_TAR}"
export PATH="${INSTALL_DIR}/go/bin:${PATH}"
export GOPATH="/tmp/gopath"
rm -f "/tmp/${GO_TAR}"

echo "Go installed: $(go version)"

# Build the server
echo "Building Go server..."
go build -o /tmp/server main.go

# Run it
echo "Starting server..."
exec /tmp/server
