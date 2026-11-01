#!/usr/bin/env bash
# Install cuRobo v0.8.0 into .venv without git.
#
# The GitHub tarball has no git metadata, so setuptools (include-package-data) skips
# curobo/content/ (solver configs + robot assets). We copy it in after installing.
set -euo pipefail
VERSION=0.8.0
CUDA_EXTRA=${CUDA_EXTRA:-cu12-torch}   # cu13-torch for a CUDA 13 driver stack
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY="$ROOT/.venv/bin/python"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

curl -sSfL "https://github.com/NVlabs/curobo/archive/refs/tags/v$VERSION.tar.gz" | tar xz -C "$WORK"
SRC="$WORK/curobo-$VERSION"
SETUPTOOLS_SCM_PRETEND_VERSION=$VERSION "$PY" -m pip install -q "$SRC[$CUDA_EXTRA]"

DEST=$("$PY" -c "import curobo, pathlib; print(pathlib.Path(curobo.__file__).parent / 'content')")
cp -r "$SRC/curobo/content/configs" "$SRC/curobo/content/assets" "$DEST/"
"$PY" -c "import curobo; from curobo.content import get_assets_path; print('cuRobo', curobo.__version__, 'content:', get_assets_path())"
