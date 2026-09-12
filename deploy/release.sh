#!/usr/bin/env bash
# Arma los paquetes de Windows y publica el release en GitHub con sus zips (o partes) y sumas.
#
#   deploy/release.sh v0.1.0 runs/frcnn runs/yolo26s_coco          # sólo CPU
#   VARIANTS="cpu cuda" deploy/release.sh v0.1.0 runs/frcnn         # CPU y CUDA
#
# Necesita `gh` autenticado (gh auth login) y el tag sin publicar; lo crea sobre HEAD.
set -euo pipefail

UV=${UV:-uv}
VARIANTS=${VARIANTS:-cpu}
TAG=${1:?uso: deploy/release.sh vX.Y.Z MODELO...}
shift
VERSION=${TAG#v}

for variant in $VARIANTS; do
    "$UV" run python deploy/build_windows.py --variant "$variant" --version "$VERSION" "$@"
done

assets=(dist/*"-$VERSION-win64-"* dist/SHA256SUMS.txt)
files=()
for asset in "${assets[@]}"; do
    [[ -f $asset ]] && files+=("$asset")
done
gh release create "$TAG" "${files[@]}" --title "$TAG" --generate-notes
