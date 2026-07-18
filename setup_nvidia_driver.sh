#!/bin/bash
# ═══════════════════════════════════════════════════════════
# NVIDIA Vulkan Driver Setup for CARLA (No Sudo)
# Extracts display/Vulkan driver libs from .run into user space
# ═══════════════════════════════════════════════════════════
# Run on: BioTuring JupyterHub server terminal
# Driver: 570.195.03 (GeForce RTX 3060, Ubuntu 22.04)
# ═══════════════════════════════════════════════════════════

set -e

DRIVER_VERSION="570.195.03"
RUN_FILE="NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"
DOWNLOAD_URL="https://us.download.nvidia.com/XFree86/Linux-x86_64/${DRIVER_VERSION}/${RUN_FILE}"

echo "╔══════════════════════════════════════════════╗"
echo "║  NVIDIA Vulkan Driver Setup (no sudo)        ║"
echo "║  Driver: ${DRIVER_VERSION}               ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Step 1: Download .run file ──────────────────────────
cd ~
if [ -f "$RUN_FILE" ]; then
    echo "[1/5] .run file already exists, skipping download"
else
    echo "[1/5] Downloading ${RUN_FILE} (~376MB)..."
    wget -q --show-progress "$DOWNLOAD_URL" -O "$RUN_FILE"
fi
chmod +x "$RUN_FILE"
echo ""

# ── Step 2: Extract (no sudo — just unpacks) ───────────
EXTRACT_DIR="$HOME/nvidia-extract"
echo "[2/5] Extracting to ${EXTRACT_DIR}/..."
rm -rf "$EXTRACT_DIR"
sh "$RUN_FILE" --extract-only --target "$EXTRACT_DIR" 2>&1 | tail -3
echo ""

# ── Step 3: Copy Vulkan/GL libs to ~/nvidia-libs/ ───────
LIBS_DIR="$HOME/nvidia-libs"
echo "[3/5] Copying NVIDIA driver libs to ${LIBS_DIR}/..."
mkdir -p "$LIBS_DIR"
cd "$EXTRACT_DIR"

# Copy ALL nvidia .so files (don't miss any dependencies)
copied=0
for lib in libnvidia-*.so.* libGLX_nvidia.so.*; do
    if [ -f "$lib" ]; then
        cp -n "$lib" "$LIBS_DIR/"
        echo "  ✓ $lib"
        ((copied++))
    fi
done
echo "  Copied ${copied} libraries"
echo ""

# ── Step 4: Create symlinks ────────────────────────────
echo "[4/5] Creating symlinks..."
cd "$LIBS_DIR"
ln -sf "libGLX_nvidia.so.${DRIVER_VERSION}" "libGLX_nvidia.so.0"
ln -sf "libnvidia-glcore.so.${DRIVER_VERSION}" "libnvidia-glcore.so"
ln -sf "libnvidia-eglcore.so.${DRIVER_VERSION}" "libnvidia-eglcore.so"
ln -sf "libnvidia-rtcore.so.${DRIVER_VERSION}" "libnvidia-rtcore.so"
echo "  Symlinks created"
echo ""

# ── Step 5: Create Vulkan ICD manifest ──────────────────
echo "[5/5] Creating Vulkan ICD manifest..."

# Resolve HOME to absolute path (needed in JSON)
HOME_ABS="$(cd ~ && pwd)"

cat > "$LIBS_DIR/nvidia_icd.json" <<EOF
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "${HOME_ABS}/nvidia-libs/libGLX_nvidia.so.0",
    "api_version": "1.3"
  }
}
EOF
echo "  ICD manifest: ${LIBS_DIR}/nvidia_icd.json"
echo ""

# ── Verify ─────────────────────────────────────────────
echo "═══════════════════════════════════════════════"
echo "Verifying..."
echo ""

# Check the critical Vulkan SPIR-V lib (was the missing piece)
if [ -f "$LIBS_DIR/libnvidia-glvkspirv.so.${DRIVER_VERSION}" ]; then
    echo "✓ libnvidia-glvkspirv.so.${DRIVER_VERSION} — present"
else
    echo "✗ MISSING: libnvidia-glvkspirv.so.${DRIVER_VERSION} — CARLA will fail!"
fi

# Check libGLX_nvidia.so.0 symlink resolves
if [ -e "$LIBS_DIR/libGLX_nvidia.so.0" ]; then
    echo "✓ libGLX_nvidia.so.0 — symlink OK"
else
    echo "✗ MISSING: libGLX_nvidia.so.0 symlink"
fi

# Test loading the ICD driver
echo ""
echo "Loading test:"
LD_LIBRARY_PATH="$LIBS_DIR:$LD_LIBRARY_PATH" \
VK_ICD_FILENAMES="$LIBS_DIR/nvidia_icd.json" \
python3 -c "
import ctypes
try:
    lib = ctypes.CDLL('${HOME_ABS}/nvidia-libs/libGLX_nvidia.so.0')
    print('✓ libGLX_nvidia.so.0 loaded successfully')
except Exception as e:
    print('✗ FAILED:', e)
"

echo ""
echo "═══════════════════════════════════════════════"
echo "Done! Files in: ${LIBS_DIR}/"
echo ""
echo "To use with CARLA, set these env vars before launching:"
echo ""
echo "  export LD_LIBRARY_PATH=\"${HOME_ABS}/nvidia-libs:\$LD_LIBRARY_PATH\""
echo "  export VK_ICD_FILENAMES=\"${HOME_ABS}/nvidia-libs/nvidia_icd.json\""
echo ""
echo "═══════════════════════════════════════════════"
