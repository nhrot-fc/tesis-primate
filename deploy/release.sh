#!/usr/bin/env bash
# Arma el runtime de Windows y un zip por modelo, y publica el release en GitHub.
#
#   deploy/release.sh v0.1.0 runs/frcnn runs/yolo26s_coco       # runtime cpu + dos modelos
#   VARIANTS="cpu cuda" deploy/release.sh v0.1.0 runs/frcnn      # runtimes cpu y cuda + un modelo
#   deploy/release.sh v0.1.0                                     # sólo el runtime
#
# Necesita `gh` autenticado. Crea un release BORRADOR (PUBLISH=1 lo publica y crea el tag
# sobre HEAD). Para publicar en otro repo (p. ej. uno privado para los pesos): REPO=usuario/repo.
set -euo pipefail

PATH="$HOME/.local/bin:$PATH"  # gh instalado con webi
UV=${UV:-uv}
VARIANTS=${VARIANTS:-cpu}
TAG=${1:?uso: deploy/release.sh vX.Y.Z [MODELO...]}
shift
VERSION=${TAG#v}

for variant in $VARIANTS; do
    "$UV" run python deploy/build_windows.py --version "$VERSION" runtime --variant "$variant"
done
if (($# > 0)); then
    "$UV" run python deploy/build_windows.py --version "$VERSION" models "$@"
fi

# Sumas de lo que se sube: runtimes, modelos
files=()
for asset in dist/*"-$VERSION-win64-"* dist/*"-$VERSION-model-"*; do
    [[ -f $asset ]] && files+=("$asset")
done
(cd dist && sha256sum "${files[@]#dist/}") > dist/SHA256SUMS.txt
files+=(dist/SHA256SUMS.txt)

# Notas en inglés: la plantilla más la tabla de archivos con su tamaño
notes=dist/release_notes.md
sed "s/{version}/$VERSION/g" deploy/release_notes.md > "$notes"
{
    echo
    echo "| File | Size | What |"
    echo "|---|---|---|"
    for f in "${files[@]}"; do
        name=$(basename "$f")
        size=$(( ($(stat -c %s "$f") + 2**19) / 2**20 )); ((size == 0)) && size="<1"
        case $name in
            *-win64-cpu.zip*)  what="runtime, any 64-bit PC" ;;
            *-win64-cuda.zip*) what="runtime with NVIDIA support" ;;
            *-model-*.zip*)    m=${name#*-model-}; what="model \`${m%%.zip*}\`" ;;
            *.unir.bat)        what="joins the parts of a split zip" ;;
            SHA256SUMS.txt)    what="checksums" ;;
            *)                 what="" ;;
        esac
        echo "| \`$name\` | $size MB | $what |"
    done
    echo
    echo "Verify downloads against \`SHA256SUMS.txt\`."
} >> "$notes"

# Borrador salvo PUBLISH=1: se revisa en la web y se publica desde ahí.
gh release create "$TAG" "${files[@]}" --title "$TAG" --notes-file "$notes" \
    ${PUBLISH:+--latest} ${PUBLISH:---draft} ${REPO:+--repo "$REPO"}
