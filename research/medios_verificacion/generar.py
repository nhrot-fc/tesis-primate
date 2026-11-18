"""Construye research/medios_verificacion/ desde las salidas de los cuadernos, las corridas de
runs/, las constantes de curación y las planillas del equipo. Necesita `make` antes (lee los .aux
para los números de sección) y, fuera de git, runs/ y capitulos/06_oe3_revision/team/."""

import hashlib
import json
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from data.annotations import CALL_SYNONYMS, MANUAL_FIXES, MANUAL_SYNONYMS  # noqa: E402
from data.species import CALL_TYPES  # noqa: E402
from models.registry import load_checkpoint  # noqa: E402

RESEARCH = ROOT / "research"
CAP = RESEARCH / "capitulos"
OUT = RESEARCH / "medios_verificacion"
RUNS = ROOT / "runs"
TEAM = CAP / "06_oe3_revision/team"
CH1, CH4, CH5, CH6 = (CAP / c for c in ["01_generalidades", "04_oe1_curacion", "05_oe2_detector", "06_oe3_revision"])
CORRIDAS = ["detr_t10_logmel_birds_v3", "frcnn_v3", "yolo26s_v3", "rtdetr_l_v3"]
FINAL = "yolo26s_v3"
VOLCADOS = {"entrenamiento": RUNS / "yolo26s_v3_kfold5/yolo26s_v3_kfold5_train_predictions.pt",
            "validacion": RUNS / f"{FINAL}/{FINAL}_val_predictions.pt",
            "prueba": RUNS / f"{FINAL}/{FINAL}_test_predictions.pt"}
COMMIT_PROTOCOLO = "63f84a8"


# ---------------------------------------------------------------- LaTeX a Markdown
def macros(path: Path) -> dict[str, str]:
    found = {}
    for line in path.read_text().splitlines():
        if m := re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", line):
            found[m[1]] = m[2]
    return found


MACROS = {k: v for p in sorted(CAP.glob("*/figures/valores.tex")) for k, v in macros(p).items()}
MACROS |= {k: v for k, v in macros(RESEARCH / "estilos.tex").items() if "#" not in v}
REFS = {}
for aux in (RESEARCH / "build/aux").rglob("*.aux"):
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux.read_text(errors="ignore")):
        REFS[m[1]] = m[2]
if not REFS:
    sys.exit("Faltan los .aux: corre `make` antes de `make medios`.")


def bib() -> dict[str, str]:
    entries, text = {}, (RESEARCH / "references.bib").read_text()
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", text, re.S):
        body = m[2]
        author = re.search(r"author\s*=\s*\{(.*?)\},?\s*\n", body, re.S)
        year = re.search(r"(?:date|year)\s*=\s*\{?(\d{4})", body)
        names, crudo = [], author[1].strip() if author else ""
        for name in ([crudo] if crudo.startswith("{{") else re.split(r"\s+and\s+", crudo)):
            grupo, name = name.startswith("{"), name.strip().strip("{}")
            names.append(name if grupo else name.split(",")[0].strip() if "," in name else name.split()[-1])
        names = [n.replace("{", "").replace("}", "") for n in names if n]
        who = names[0] if len(names) == 1 else " y ".join(names) if len(names) == 2 else f"{names[0]} et al."
        entries[m[1].strip()] = (who, year[1] if year else "s. f.")
    return entries


BIB = bib()


def argumento(text: str, i: int) -> tuple[str, int]:
    while i < len(text) and text[i] == "[":
        i = text.index("]", i) + 1
    if i >= len(text) or text[i] != "{":
        return "", i
    depth, j = 0, i
    while j < len(text):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i + 1:j], j + 1
        j += 1
    return text[i + 1:], len(text)


def comandos(text: str, nombre: str, fn, n: int = 1) -> str:
    out, i = [], 0
    pattern = re.compile(r"\\" + nombre + r"(?![a-zA-Z])")
    while m := pattern.search(text, i):
        out.append(text[i:m.start()])
        args, j = [], m.end()
        for _ in range(n):
            a, j = argumento(text, j)
            args.append(a)
        out.append(fn(*args))
        i = j
    return "".join(out) + text[i:]


def cita(keys: str, narrativa: bool = False) -> str:
    partes = [BIB.get(k.strip(), (k.strip(), "")) for k in keys.split(",")]
    if narrativa:
        return "; ".join(f"{w} ({y})" for w, y in partes)
    return "(" + "; ".join(f"{w}, {y}" for w, y in partes) + ")"


def md(text: str, depth: int = 0) -> str:
    t = re.sub(r"(?<!\\)%.*", "", text)
    t = comandos(t, "multicolumn", lambda a, b, c: c, 3)
    t = comandos(t, "thead", lambda a: a.replace("-\\\\", "-").replace("\\\\", " "))
    t = t.replace("\\log", "log")
    t = re.sub(r"_\{([0-9]+)\}", lambda m: m[1].translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")), t)
    t = re.sub(r"\^\{(-?[0-9]+)\}", lambda m: m[1].translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")), t)
    for c in ["code", "file", "texttt"]:
        t = comandos(t, c, lambda a: "`" + a.replace("\\allowbreak", "").replace("\\_", "_")
                     .replace("\\char32", " ").replace("\\", "").replace("{", "").replace("}", "") + "`")
    t = comandos(t, "citep", lambda a: cita(a))
    t = comandos(t, "citealp", lambda a: cita(a)[1:-1])
    t = comandos(t, "citet", lambda a: cita(a, True))
    for c in ["ref", "eqref"]:
        t = comandos(t, c, lambda a: REFS.get(a, "?"))
    t = comandos(t, "url", lambda a: a)
    t = comandos(t, "footnote", lambda a: f" ({a})")
    for c in ["textbf", "rotulo"]:
        t = comandos(t, c, lambda a: f"**{a.strip()}**")
    for c in ["textit", "emph"]:
        t = comandos(t, c, lambda a: f"*{a.strip()}*")
    t = comandos(t, "tfrac", lambda a, b: f"{a}/{b}", 2)
    if depth < 3:
        for name in sorted(MACROS, key=len, reverse=True):
            t = re.sub(r"\\" + name + r"(?![a-zA-Z])(\{\})?", lambda _m, n=name: md(MACROS[n], depth + 1), t)
    for a, b in [("\\times", "×"), ("\\geq", "≥"), ("\\leq", "≤"), ("\\to", "→"), ("\\Rightarrow", "⇒"),
                 ("\\pm", "±"), ("\\neq", "≠"), ("\\notin", "∉"), ("\\ldots", "…"), ("\\,", "\u202f"),
                 ("\\%", "%"), ("\\&", "&"), ("\\_", "_"), ("\\#", "#"), ("---", "—"), ("--", "–"),
                 ("``", "“"), ("''", "”"), ("~", " "), ("$", ""), ("\\\\", " ")]:
        t = t.replace(a, b)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", "", t)
    t = t.replace("{", "").replace("}", "")
    return re.sub(r"[ \t\n]+", " ", t).strip()


def plano(celda: str) -> str:
    texto = celda.replace("`", "").replace("**", "").replace("*", "")
    texto = re.sub(r"(?<=\d)\u202f(?=\d)", "", texto).replace("\u202f", " ")
    return texto.replace(",", ".") if re.fullmatch(r"[+-]?\d+(,\d+)?", texto) else texto


def filas(cuerpo: str, base: Path) -> list[list[str]]:
    cuerpo = re.sub(r"\\input\{\\fig\w+/([^}]+)\}", lambda m: (base / "figures" / f"{m[1]}.tex").read_text(), cuerpo)
    cuerpo = re.sub(r"(?<!\\)%.*", "", cuerpo)
    out = []
    for fila in re.split(r"\\\\\s*(?:\n|$)", cuerpo):
        fila = re.sub(r"\\(midrule|addlinespace|cmidrule(\([^)]*\))?(\{[^}]*\})?|hline)", "", fila).strip()
        fila = comandos(fila, "multicolumn", lambda n, _s, x: " & ".join([x] * int(n)), 3)
        if fila:
            out.append([md(c) for c in re.split(r"(?<!\\)&", fila)])
    return out


def tabla(tex: Path, label: str) -> tuple[str, list[str], list[list[str]]]:
    text = tex.read_text()
    i = text.index(f"\\label{{{label}}}")
    inicio = text.rfind("\\begin{table}", 0, i)
    bloque = text[inicio:text.index("\\end{table}", i)]
    titulo = md(argumento(bloque, bloque.index("\\caption") + len("\\caption"))[0])
    cabeza = bloque[bloque.index("\\toprule") + 8:bloque.index("\\midrule")]
    niveles = [f for f in filas(cabeza, tex.parent) if any(f)]
    cabeza = [" · ".join(n[i].replace("**", "") for n in niveles if i < len(n) and n[i])
              for i in range(len(niveles[-1]))]
    cuerpo = filas(bloque[bloque.index("\\midrule") + 8:bloque.index("\\bottomrule")], tex.parent)
    return titulo, cabeza, cuerpo


def tabla_md(cabeza: list[str], cuerpo: list[list[str]]) -> str:
    ancho = len(cabeza)
    lineas = ["| " + " | ".join(cabeza) + " |", "|" + "---|" * ancho]
    lineas += ["| " + " | ".join((f + [""] * ancho)[:ancho]) + " |" for f in cuerpo]
    return "\n".join(lineas)


def tabla_doc(tex: Path, label: str) -> str:
    titulo, cabeza, cuerpo = tabla(tex, label)
    return f"*Tabla {REFS[label]}. {titulo}*\n\n{tabla_md(cabeza, cuerpo)}"


def parrafo(tex: Path, inicio: str) -> str:
    text = tex.read_text()
    i = text.index(inicio)
    return md(text[i + len(inicio):text.index("\n\n", i)])


def items(tex: Path, label: str) -> str:
    text = tex.read_text()
    i = text.index("\\begin{itemize}", text.index(f"\\label{{{label}}}"))
    bloque = text[i + len("\\begin{itemize}"):text.index("\\end{itemize}", i)]
    return "\n".join("- " + md(x) for x in re.split(r"\\item\s", bloque) if x.strip())


IOV = {f[0]: f for label in ["tab:iov-oe1", "tab:iov-oe2", "tab:iov-oe3"]
       for f in tabla(CH1 / "01_generalidades.tex", label)[2] if re.match(r"RE\d\.\d$", f[0])}


def seccion(label: str) -> str:
    return f"sección {REFS[label]}"


# ---------------------------------------------------------------- escritura
def readme(re_id: str, desarrollo: str, elementos: list[tuple[str, str]], extra: str = "") -> str:
    _, resultado, medio, indicador = IOV[re_id]
    texto = (f"# {re_id}. {resultado.rstrip('.')}\n\n"
             f"**Medio de verificación.** {medio}\n\n**Indicador.** {indicador}\n\n"
             f"**Dónde lo desarrolla el documento.** {desarrollo}\n\n"
             + tabla_md(["Elemento del indicador", "Dónde se comprueba"], [list(e) for e in elementos]) + "\n")
    return texto + (f"\n{extra.strip()}\n" if extra else "")


def escribir(path: Path, contenido: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.write_bytes if isinstance(contenido, bytes) else path.write_text)(contenido)


def csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def copiar(origen: Path, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origen, destino)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def volcado(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------- OE1
def re11(d: Path) -> None:
    sinonimos = [("todas", escrito, codigo) for escrito, codigo in MANUAL_SYNONYMS.items()]
    sinonimos += [(sp, nombre, codigo) for sp, mapa in CALL_SYNONYMS.items() for nombre, codigo in mapa.items()]
    csv(d / "sinonimos.csv", pd.DataFrame(sinonimos, columns=["especie", "escrito", "codigo"]))
    csv(d / "correcciones_pares.csv", pd.DataFrame(
        [(*a, *b) for a, b in MANUAL_FIXES.items()],
        columns=["especie", "tipo", "especie_corregida", "tipo_corregido"]))
    csv(d / "vocabulario.csv", pd.DataFrame(
        [(sp.name.lower(), codigo, nombre) for sp, tipos in CALL_TYPES.items() for codigo, nombre in tipos.items()],
        columns=["especie", "codigo", "nombre"]))
    _, _, reglas = tabla(CH4 / "04_oe1_curacion.tex", "tab:reglas")
    csv(d / "reglas.csv", pd.DataFrame([[plano(c) for c in f[:3]] for f in reglas],
                                       columns=["regla", "filas", "tratamiento"]))
    escribir(d / "README.md", readme("RE1.1", (
        f"{seccion('sec:re11').capitalize()}, que es el documento del protocolo. La "
        f"{seccion('sec:verificacion-re11')} contrasta cada regla con el código."), [
        ("Normalización de columnas y etiquetas", "`reglas.csv`, filas de tipo de llamada y de especie"),
        ("Mapa de sinónimos", "`sinonimos.csv` (los de todas las especies y los nombres legibles por especie)"),
        ("Descarte de ruido y cajas degeneradas", "`reglas.csv`, filas `noise` y caja degenerada"),
        ("Corrección de pares imposibles", "`correcciones_pares.csv`; sus filas, en `reglas.csv`"),
        ("Recorte al límite de Nyquist", "`reglas.csv`, fila de frecuencia superior"),
        ("Tratamiento de registros fuera de vocabulario", "`vocabulario.csv`; las filas marcadas, en `reglas.csv`"),
    ], "Las tablas salen de las constantes del código de curación (`src/data/annotations.py` y "
       "`src/data/species.py`). Los conteos de `reglas.csv` son los de la auditoría del capítulo 4 "
       f"(Tabla {REFS['tab:reglas']}), sobre las filas crudas."))


def re12(d: Path) -> None:
    copiar(CH4 / "figures/meta.json", d / "meta.json")
    registro, conjuntos = [], {}
    esperado = {"entrenamiento": "nWindowsTrain", "validacion": "nWindowsVal", "prueba": "nWindowsTest"}
    for nombre, path in VOLCADOS.items():
        v = volcado(path)
        grabaciones = sorted(n.split("data/raw/")[1] for n in v["recording_names"])
        conjuntos[nombre] = set(grabaciones)
        escribir(d / f"grabaciones_{nombre}.txt", "\n".join(grabaciones) + "\n")
        ventanas = int(v["n_images"])
        assert ventanas == int(re.sub(r"\D", "", MACROS[esperado[nombre]])), nombre
        registro.append(f"{nombre:<14} {ventanas:>6} ventanas leídas, {len(grabaciones):>5} grabaciones "
                        f"({path.relative_to(ROOT)})")
        if nombre == "prueba":
            escribir(d / "labels.json", json.dumps(dict(enumerate(v["labels"])), indent=2, ensure_ascii=False) + "\n")
    pares = [("entrenamiento", "validacion"), ("entrenamiento", "prueba"), ("validacion", "prueba")]
    compartidas = [f"{a} ∩ {b}: {len(conjuntos[a] & conjuntos[b])}" for a, b in pares]
    assert all(not (conjuntos[a] & conjuntos[b]) for a, b in pares)
    log = (RUNS / f"{FINAL}/nohup.log").read_text(errors="ignore")
    log = re.sub(r"\x1b\[[0-9;]*[mK]", "", log).replace("\r", "\n")
    exportacion = sorted({re.sub(r"\s*━.*", "", x).replace("/home/fcandia/tesis-primate/", "")
                          for x in log.splitlines() if "Scanning" in x and "100%" in x})
    escribir(d / "registro_de_carga.txt", "\n".join([
        "Carga del caché de data/processed/ en sus dos usos.", "",
        "1. Exportación de train.pt y val.pt a las imágenes de Ultralytics (registro de la corrida "
        f"{FINAL}):", *[f"   {x}" for x in exportacion], "",
        "2. Ventanas leídas de cada partición al volcar predicciones (validación y prueba, en la evaluación;",
        "   entrenamiento, en la validación cruzada que buscó llamadas sin anotar):",
        *[f"   {x}" for x in registro], "",
        "3. Grabaciones compartidas entre particiones:", *[f"   {x}" for x in compartidas], ""]))
    meta = json.loads((d / "meta.json").read_text())
    escribir(d / "README.md", readme("RE1.2", (
        f"{seccion('sec:re12').capitalize()}; la {seccion('sec:verificacion-re12')} aplica las "
        "tres comprobaciones del indicador."), [
        ("Los archivos procesados cargan sin error", "`registro_de_carga.txt`: 0 imágenes corruptas al "
         "exportar y todas las ventanas leídas al evaluar"),
        ("`meta.json` registra la semilla", f"`meta.json`, clave `seed` ({meta['seed']})"),
        ("… el criterio de selección", f"`meta.json`, clave `min_pair_count` ({meta['min_pair_count']})"),
        ("… las exclusiones", "`meta.json`, claves `excluded_labels` y `joined_labels`"),
        ("… la etiqueta usada", "`labels.json`, índice → clase `especie/llamada`"),
        ("… la normalización", "`meta.json`, clave `db_range` y bloque `params`"),
        ("El código divide por archivo de grabación", "`grabaciones_*.txt`; `registro_de_carga.txt`, "
         "punto 3: ninguna grabación compartida"),
        ("Proporciones 60/22,5/17,5", f"`meta.json`, clave `split_ratios` ({meta['split_ratios']})"),
    ], "El caché (`data/processed/`) no se publica, porque son espectrogramas de grabaciones del "
       "equipo. `meta.json` es el del caché con que se entrenaron los modelos; las listas de grabaciones "
       "y los conteos salen de los volcados de predicciones de esas mismas corridas."))


def re13(d: Path) -> None:
    tex = CH4 / "04_oe1_curacion.tex"
    meta = json.loads((CH4 / "figures/meta.json").read_text())
    partes = [
        ("1. Las ocho especies", tabla_doc(tex, "tab:composicion")),
        ("2. Vocabulario", tabla_doc(tex, "tab:vocabulario") + "\n\n" + tabla_doc(tex, "tab:revision")),
        ("3. Composición", tabla_doc(tex, "tab:particiones") + "\n\n" + tabla_doc(tex, "tab:clases")),
        ("4. Desbalance", parrafo(tex, "\\rotulo{Desbalance.}")),
        ("5. Diccionario de campos", "\n\n".join(tabla_doc(tex, t) for t in ["tab:archivos", "tab:cleaned",
                                                                              "tab:processed"])),
        ("6. Ventaneo", f"Ventanas de {md(MACROS['clipLen'])} s con salto de {md(MACROS['clipHop'])} s. "
         + parrafo(tex, "\\rotulo{Regla de asignación.}") + "\n\n" + tabla_doc(tex, "tab:ventaneo")),
        ("7. Normalización log-mel", tabla_doc(tex, "tab:preproc") + "\n\nRango en dB del caché "
         "(`meta.json`, `db_range`): " + " a ".join(f"{x:.2f}".replace(".", ",") for x in meta["db_range"])
         + " dB."),
        ("8. Exclusiones", tabla_doc(tex, "tab:excluidas")),
        ("9. Limitaciones conocidas", items(tex, "sec:limitaciones")),
    ]
    cuerpo = "\n\n".join(f"## {t}\n\n{c}" for t, c in partes)
    _, resultado, medio, indicador = IOV["RE1.3"]
    escribir(d / "README.md", (
        f"# RE1.3. {resultado.rstrip('.')}\n\n**Medio de verificación.** {medio}\n\n"
        f"**Indicador.** {indicador}\n\n**Dónde lo desarrolla el documento.** "
        f"{seccion('sec:re13').capitalize()}; la {seccion('sec:verificacion-re13')} ubica cada elemento.\n\n"
        "Esta es la ficha en un solo archivo, en el orden de los nueve elementos del indicador. Las tablas y "
        "los textos se toman del capítulo 4 y de sus salidas, sin transcribir ninguna cifra; las tablas, "
        "figuras y secciones que citan son las del documento. Los metadatos "
        "del conjunto (`meta.json` y `labels.json`) están en la carpeta RE1.2.\n\n" + cuerpo + "\n"))


# ---------------------------------------------------------------- OE2
def re21(d: Path) -> None:
    for corrida in CORRIDAS:
        copiar(RUNS / corrida / "config.json", d / "configuraciones" / f"{corrida}.json")
        if (RUNS / corrida / "args.yaml").exists():
            copiar(RUNS / corrida / "args.yaml", d / "configuraciones" / f"{corrida}_ultralytics.yaml")
    escribir(d / "README.md", readme("RE2.1", (
        f"{seccion('sec:re21').capitalize()}, que es el documento de diseño; la "
        f"{seccion('sec:verificacion-re21')} contrasta cada elemento del indicador."), [
        ("Arquitectura AST-Deformable-DETR implementada", "Módulo de arquitecturas del repositorio "
         "(`src/models/`); `configuraciones/detr_t10_logmel_birds_v3.json`, clave `architecture`"),
        ("Todos sus hiperparámetros", "`configuraciones/detr_t10_logmel_birds_v3.json`, bloque `hparams`"),
        ("Configuraciones comparadas", "Los cuatro `configuraciones/*.json`; las dos de Ultralytics, también "
         "con su `*_ultralytics.yaml`"),
        ("Partición", "Bloque `dataset` de cada configuración: el mismo `meta.json` del caché en las cuatro"),
        ("Métricas y regla de selección", f"{seccion('sec:protocolo-oe2').capitalize()} del documento"),
    ]))


def log_ultralytics(path: Path) -> str:
    texto = re.sub(r"\x1b\[[0-9;]*[mK]", "", path.read_text(errors="ignore")).replace("\r", "\n")
    lineas = [x.rstrip() for x in texto.splitlines() if x.strip() and "━" not in x and "──" not in x]
    return "\n".join(x for x in lineas if len(x) < 400) + "\n"


def re22(d: Path) -> None:
    for corrida in CORRIDAS:
        r = RUNS / corrida
        if (r / "metrics.jsonl").exists():
            copiar(r / "train.log", d / "registros" / f"{corrida}_registro.txt")
            copiar(r / "metrics.jsonl", d / "registros" / f"{corrida}_metricas.jsonl")
        else:
            escribir(d / "registros" / f"{corrida}_registro.txt", (r / "train.log").read_text() + log_ultralytics(r / "nohup.log"))
            copiar(r / "results.csv", d / "registros" / f"{corrida}_metricas.csv")
    copiar(ROOT / "pyproject.toml", d / "pyproject.toml")
    texto = (CH5 / "05_oe2_detector.tex").read_text()
    i = texto.index("label={lst:pipeline}")
    listado = texto[texto.index("\n", i) + 1:texto.index("\\end{lstlisting}", i)]
    escribir(d / "README.md", readme("RE2.2", (
        f"{seccion('sec:re22').capitalize()}; la {seccion('sec:verificacion-re22')} contrasta el indicador."), [
        ("Genera los splits procesados", "`prepare_annotations.py` y `prepare_data.py` (listado de abajo); su "
         "resultado, en la carpeta RE1.2"),
        ("Entrena desde un conjunto de configuración", "`train.py --arch`; un registro por corrida en "
         "`registros/`, con sus métricas por época"),
        ("Evalúa un checkpoint", "`dump_predictions.py` y `compare_models.py`; su resultado, en la carpeta RE2.3"),
        ("Ejecuta inferencia sobre un WAV", "`detect.py`; su resultado, en la carpeta RE2.4"),
        ("Dependencias declaradas", "`pyproject.toml` (aquí) y `uv.lock` (raíz del repositorio)"),
        ("Comandos de ejecución declarados", "El listado de abajo, que es el Listado "
         f"{REFS['lst:pipeline']} del documento"),
    ], f"Interfaz del pipeline (scripts de `src/`):\n\n```bash\n{listado.rstrip()}\n```"))


def re23(d: Path) -> None:
    for ext in ["txt", "json"]:
        copiar(RUNS / f"comparacion/comparacion_modelos_v3.{ext}", d / f"comparacion_modelos_v3.{ext}")
    for corrida in CORRIDAS:
        copiar(RUNS / corrida / "operating_point.json", d / "punto_de_operacion" / f"{corrida}.json")
    copiar(CH5 / "figures/matriz_confusion.png", d / "matriz_confusion.png")
    tex = CH5 / "05_oe2_detector.tex"
    for nombre, label in [("resultados_punto_de_operacion", "tab:resultados-op"), ("ejes", "tab:ejes"),
                          ("recall_por_clase", "tab:recall-clase"), ("diferencias_pareadas", "tab:pares")]:
        _, cabeza, cuerpo = tabla(tex, label)
        csv(d / "tablas" / f"{nombre}.csv", pd.DataFrame([[plano(c) for c in f] for f in cuerpo],
                                                          columns=[plano(c) for c in cabeza]))
    csv(d / "tablas" / "iou_medio.csv", pd.DataFrame(
        [(c, plano(md(MACROS[f"rIouMean{k}"]))) for c, k in zip(CORRIDAS, ["Det", "Frc", "Yol", "Rtd"], strict=True)],
        columns=["corrida", "iou_medio"]))
    t = lambda label: f"Tabla {REFS[label]}"  # noqa: E731
    escribir(d / "README.md", readme("RE2.3", (
        f"{seccion('sec:re23').capitalize()}, que es el informe de experimentación; la "
        f"{seccion('sec:verificacion-re23')} contrasta el indicador magnitud por magnitud."), [
        ("Recall, precisión, F1", f"`tablas/resultados_punto_de_operacion.csv` ({t('tab:resultados-op')}); "
         "recall y precisión también en `comparacion_modelos_v3.txt`, punto 1"),
        ("IoU medio", f"`tablas/iou_medio.csv`, sobre los pares emparejados (Figura {REFS['fig:iou']}); el "
         "`.txt` da el mediano, punto 2"),
        ("AP a IoU 0,25 y 0,5", "`tablas/resultados_punto_de_operacion.csv`, columnas mAP"),
        ("Recall por clase", f"`tablas/recall_por_clase.csv` ({t('tab:recall-clase')}); "
         "`comparacion_modelos_v3.txt`, punto 3"),
        ("Matriz de confusión", "`matriz_confusion.png`"),
        ("Separa detección, encuadre y clasificación", f"`tablas/ejes.csv` ({t('tab:ejes')}); "
         "`comparacion_modelos_v3.txt`, punto 2"),
        ("Justifica la selección del modelo", f"`tablas/diferencias_pareadas.csv` ({t('tab:pares')}) y "
         f"{seccion('sec:protocolo-oe2')}"),
        ("Umbral de cada modelo", "`punto_de_operacion/*.json`"),
    ], "Las tablas de `tablas/` son las del documento, que el cuaderno del capítulo 5 calcula a partir de "
       "los mismos volcados que `comparacion_modelos_v3.txt`."))


def re24(d: Path) -> None:
    ckpt = RUNS / FINAL / "best.pt"
    final = load_checkpoint(ckpt, "cpu")
    guardado = torch.load(ckpt, map_location="cpu", weights_only=False)
    claves, modelo = set(guardado["state_dict"]), set(final.model.state_dict())
    escribir(d / "verificacion_carga.txt", "\n".join([
        f"$ load_checkpoint('runs/{FINAL}/best.pt', 'cpu')",
        f"arquitectura:      {final.architecture}",
        f"clases:            {len(final.labels)}",
        f"tensores (archivo): {len(claves)}",
        f"tensores (modelo):  {len(modelo)}",
        f"claves faltantes:  {len(modelo - claves)}",
        f"claves sobrantes:  {len(claves - modelo)}",
        f"carga estricta:    {'sí' if claves == modelo else 'NO'}",
        f"umbral de operación: {final.operating_point}", ""]))
    assert claves == modelo
    op = RUNS / FINAL / "operating_point.json"
    escribir(d / "SHA256_checkpoint.txt", f"{sha256(ckpt)}  models/{FINAL}/best.pt\n"
                                          f"{sha256(op)}  models/{FINAL}/operating_point.json\n")
    copiar(op, d / "operating_point.json")
    copiar(CH5 / "figures/demo.detections.txt", d / "demo.detections.txt")
    copiar(CH5 / "figures/detect_stdout.txt", d / "transcripcion_detect.txt")
    zip_release = f"detector-2.0.0-model-{FINAL}.zip"
    escribir(d / "README.md", readme("RE2.4", (
        f"{seccion('sec:re24').capitalize()}; la demostración está en la {seccion('sec:demo-oe2')}."), [
        ("El checkpoint se carga sin claves incompatibles", "`verificacion_carga.txt`"),
        ("La inferencia sobre un WAV produce un archivo tabulado", "`transcripcion_detect.txt` (la orden y su "
         "salida) y `demo.detections.txt` (la tabla)"),
        ("… con las columnas de Raven, coordenadas tiempo–frecuencia, especie, tipo de llamada y puntuación",
         "Cabecera de `demo.detections.txt`"),
        ("Checkpoint y vocabulario", f"Versión v0.2.0 del repositorio, archivo `{zip_release}`; sus sumas, en "
         "`SHA256_checkpoint.txt`"),
        ("Punto de operación", "`operating_point.json`"),
    ], f"El checkpoint no se guarda aquí: está en {md(MACROS['rRepoUrl'])}/releases/tag/v0.2.0, dentro de "
       f"`{zip_release}`, con las rutas de `SHA256_checkpoint.txt`. El indicador nombra un `.pth` y "
       "`src/infer.py`, nombres del prototipo: el checkpoint final es un `.pt` autocontenido, que lleva su "
       "vocabulario, y la inferencia la ejecuta `src/detect.py`."))


def re25(d: Path) -> None:
    _, cabeza, cuerpo = tabla(CH5 / "05_oe2_detector.tex", "tab:donde")
    url = md(MACROS["rRepoUrl"])
    escribir(d / "README.md", readme("RE2.5", (
        f"{seccion('sec:re25').capitalize()}."), [
        ("URL del repositorio", url),
        ("Código de curación, entrenamiento, evaluación e inferencia", "Tabla de abajo"),
        ("Instrucciones de ejecución", "`README.md` del repositorio y el listado de la carpeta RE2.2"),
        ("Artefactos del modelo final", f"{url}/releases/tag/v0.2.0 (carpeta RE2.4)"),
    ], f"*Tabla {REFS['tab:donde']} del documento.*\n\n{tabla_md(cabeza, cuerpo)}"))


# ---------------------------------------------------------------- OE3
XLSX = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def hoja(path: Path) -> pd.DataFrame:
    book, tag = zipfile.ZipFile(path), f"{{{XLSX['m']}}}t"
    shared = []
    if "xl/sharedStrings.xml" in book.namelist():
        root = ET.fromstring(book.read("xl/sharedStrings.xml"))
        shared = ["".join(t.text or "" for t in si.iter(tag)) for si in root.findall("m:si", XLSX)]
    rows = []
    for row in ET.fromstring(book.read("xl/worksheets/sheet1.xml")).iter(f"{{{XLSX['m']}}}row"):
        cells = {}
        for cell in row.findall("m:c", XLSX):
            column = 0
            for letter in re.match(r"[A-Z]+", cell.get("r")).group(0):
                column = column * 26 + ord(letter) - 64
            if cell.get("t") == "inlineStr":
                cells[column - 1] = "".join(t.text or "" for t in cell.iter(tag))
            elif (value := cell.find("m:v", XLSX)) is not None:
                cells[column - 1] = shared[int(value.text)] if cell.get("t") == "s" else value.text
        rows.append(cells)
    width = max((max(c) for c in rows if c), default=-1) + 1
    header = [rows[0].get(i, f"col{i}") for i in range(width)]
    return pd.DataFrame([[r.get(i, "") for i in range(width)] for r in rows[1:]], columns=header)


def re31(d: Path) -> None:
    copiar(CH6 / "protocolo.md", d / "protocolo.md")
    commit = subprocess.run(["git", "show", "-s", "--format=%H%n%cs", COMMIT_PROTOCOLO], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.split("\n")
    escribir(d / "README.md", readme("RE3.1", (
        f"{seccion('sec:re31').capitalize()}; la {seccion('sec:verificacion-re31')} contrasta el indicador."), [
        ("Subconjunto de prueba", "`protocolo.md`, §2"),
        ("Punto de operación elegido en validación", "`protocolo.md`, §1"),
        ("Umbral de IoU para verdadero positivo", "`protocolo.md`, §3"),
        ("Fijados antes de medir", f"Commit `{COMMIT_PROTOCOLO}` del {commit[1]}, anterior a la sesión "
         "(11–13/11/2026) y a la revisión de cajas; `protocolo.md`, «Cambios después del commit»"),
        ("Métricas", "`protocolo.md`, §4 y §5"),
        ("Emparejamiento", "`protocolo.md`, §3"),
        ("Procedimiento para revisar discrepancias y eventos omitidos", "`protocolo.md`, §5"),
    ], f"Commit completo: `{commit[0]}`. `protocolo.md` es la versión actual; la del commit se ve con "
       f"`git show {COMMIT_PROTOCOLO}`."))


def segundos(reloj: pd.Series) -> pd.Series:
    def uno(t: str) -> float:
        if ":" not in t:
            return float(t) * 86400
        p = [float(x) for x in t.split(":")]
        return p[0] * 3600 + p[1] * 60 + (p[2] if len(p) > 2 else 0.0)
    return reloj.str.strip().map(uno)


def hora(s: float) -> str:
    s = round(s) % 86400
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}"


def re32(d: Path) -> None:
    tiempos = hoja(TEAM / "answers/times.xlsx")[["folder", "start_time", "end_time", "minutes_paused"]]
    tiempos = tiempos[(tiempos.folder != "") & (tiempos.start_time != "")].copy()
    inicio, fin = segundos(tiempos.start_time), segundos(tiempos.end_time)
    pausa = pd.to_numeric(tiempos.minutes_paused, errors="coerce").fillna(0)
    tiempos["trabajo_s"] = (fin - inicio) % 86400 - pausa * 60
    bloques = pd.read_csv(TEAM / "data/timed_session_folders.csv")
    audio = bloques.groupby(["folder", "condition"], as_index=False).agg(
        grabaciones=("recording", "size"), audio_s=("duration_s", "sum"))
    sesion = tiempos.assign(inicio=inicio.map(hora), termino=fin.map(hora), pausa_min=pausa).merge(
        audio, on="folder", validate="one_to_one").sort_values("folder")
    sesion["trabajo_min"] = (sesion.trabajo_s / 60).round(1)
    sesion["min_por_hora_de_audio"] = (sesion.trabajo_s / 60 / (sesion.audio_s / 3600)).round(1)
    sesion["audio_s"] = sesion.audio_s.round(1)
    csv(d / "tiempos_sesion.csv", sesion.rename(columns={"folder": "carpeta", "condition": "condicion"})[
        ["carpeta", "condicion", "grabaciones", "audio_s", "inicio", "termino", "pausa_min", "trabajo_min",
         "min_por_hora_de_audio"]])
    csv(d / "grabaciones_sesion.csv", bloques.rename(columns={"folder": "carpeta", "condition": "condicion",
                                                              "recording": "grabacion"})[
        ["carpeta", "condicion", "grabacion", "duration_s", "n_annotations", "n_proposals"]])
    copiar(TEAM / "data/test_cpu_times.csv", d / "test_cpu_times.csv")
    escribir(d / "README.md", readme("RE3.2", (
        f"{seccion('sec:re32').capitalize()}; la {seccion('sec:verificacion-re32')} contrasta el indicador."), [
        ("Minutos por hora de audio, manual y asistida", "`tiempos_sesion.csv`, columna "
         "`min_por_hora_de_audio`"),
        ("Material comparable", "`grabaciones_sesion.csv`: grabaciones, duración, anotaciones y propuestas "
         "de cada carpeta"),
        ("Orden de revisión declarado", "`tiempos_sesion.csv`: carpetas 1 a 4 en el orden en que se hicieron"),
        ("Tiempo de procesamiento del modelo", "`test_cpu_times.csv`: segundos de CPU y duración de cada "
         "grabación de prueba"),
    ], "Las horas vienen de la planilla que devolvió el equipo, sin sus notas ni la columna de quién anotó, "
       "que traen nombres. El trabajo de cada carpeta es término menos inicio, menos las pausas."))


def lectura(estrato: str, respuesta: str) -> str:
    if respuesta == "?":
        return "incierto"
    if estrato == "match":
        return "acierto confirmado" if respuesta == "correct" else "desacuerdo con la referencia"
    if respuesta == "no":
        return "ruido"
    if estrato.startswith("none"):
        return "evento no anotado"
    return {"correct": "anotación por corregir", "other": "error de clase"}.get(respuesta, "error de encuadre")


def re33(d: Path) -> None:
    clave = pd.read_csv(TEAM / "data/check_boxes_key.csv")
    hojas = hoja(TEAM / "answers/boxes.xlsx")
    hojas = hojas[hojas.ID.str.startswith("C")].assign(answer=lambda h: h.answer.str.strip().str.lower())
    hojas = hojas.assign(answer=hojas.answer.replace({"corect": "correct", "too short": "box"}))
    hojas = hojas[hojas.answer != ""]
    filas_ = clave.merge(hojas[["ID", "answer", "comment"]], on="ID", validate="one_to_one")
    assert set(filas_.stratum) >= {"match"}, "estratos inesperados"
    personas = sorted(p for p in (TEAM / "answers/audio").glob("*") if p.is_dir())
    persona = {w.name: letra for letra, p in zip("ABCDEFGH", personas) for w in p.glob("*.wav")}
    salida = pd.DataFrame({
        "id": filas_.ID, "grabacion": filas_.recording,
        "inicio_s": filas_["Begin Time (s)"].round(3), "fin_s": filas_["End Time (s)"].round(3),
        "frec_min_hz": filas_["Low Freq (Hz)"].round(1), "frec_max_hz": filas_["High Freq (Hz)"].round(1),
        "clase_propuesta": filas_.label, "confianza": filas_.score.round(3), "estrato": filas_.stratum,
        "respuesta": filas_.answer, "comentario": filas_.comment.str.strip(),
        "categoria": [lectura(s, a) for s, a in zip(filas_.stratum, filas_.answer, strict=True)],
        "persona": filas_.recording.map(lambda r: persona.get(Path(r).name, ""))})
    assert len(salida) == int(re.sub(r"\D", "", MACROS["uItems"]))
    csv(d / "respuestas_revision.csv", salida)
    copiar(CH6 / "figures/revision_ejemplos.png", d / "revision_ejemplos.png")
    escribir(d / "README.md", readme("RE3.3", (
        f"{seccion('sec:re33').capitalize()}; la {seccion('sec:verificacion-re33')} contrasta el indicador."), [
        ("Predicciones de alta confianza revisadas", "`respuestas_revision.csv`, estrato `none_alta` "
         "(confianza mayor que 0,5)"),
        ("Clasificadas en evento no anotado, ruido y error de encuadre", "`respuestas_revision.csv`, columna "
         "`categoria`, que aplica la lectura del protocolo a `estrato` y `respuesta`"),
        ("Espectrogramas anotados", "`revision_ejemplos.png`"),
    ], "Una fila por caja revisada. `estrato` es de dónde salió la caja; `respuesta`, lo que contestó el "
       "revisor a ciegas; `persona`, A o B cuando la grabación volvió separada por persona. La caja es la "
       "del modelo, no una anotación del equipo."))


# ---------------------------------------------------------------- índice
CARPETAS = [("04_oe1_curacion", ["RE1.1", "RE1.2", "RE1.3"], [re11, re12, re13]),
            ("05_oe2_detector", ["RE2.1", "RE2.2", "RE2.3", "RE2.4", "RE2.5"], [re21, re22, re23, re24, re25]),
            ("06_oe3_revision", ["RE3.1", "RE3.2", "RE3.3"], [re31, re32, re33])]


def main() -> None:
    for capitulo, _, _ in CARPETAS:
        shutil.rmtree(OUT / capitulo, ignore_errors=True)
    indice = []
    for capitulo, ids, funciones in CARPETAS:
        for re_id, fn in zip(ids, funciones, strict=True):
            d = OUT / capitulo / re_id
            fn(d)
            archivos = sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())
            indice.append([re_id, f"`{capitulo}/{re_id}/`", ", ".join(f"`{a}`" for a in archivos if "/" not in a)
                           + (" y subcarpetas" if any("/" in a for a in archivos) else "")])
            print(f"{re_id}: {len(archivos)} archivos")
    escribir(OUT / "README.md", (
        "# Medios de verificación\n\nLa evidencia de los once resultados esperados de la tesis, por capítulo "
        f"y por resultado (Anexo {REFS['anx:medios']} del documento). Cada carpeta tiene un `README.md` con "
        "el medio de verificación y el indicador copiados del documento, la sección que desarrolla el "
        "resultado y una tabla que lleva cada elemento del indicador a su archivo.\n\n"
        + tabla_md(["RE", "Carpeta", "Archivos"], indice) + "\n\n"
        "No contiene las grabaciones, las tablas de Raven ni el caché de ventanas, que pertenecen al equipo "
        "de investigación, ni los nombres de quienes revisaron.\n\n"
        "Se regenera con `make` y después `make medios`, desde la raíz del repositorio. `generar.py` lee las "
        "salidas de los cuadernos de los capítulos, las constantes del código de curación, las corridas de "
        "`runs/` y las planillas que devolvió el equipo. Las dos últimas no están en git.\n"))


if __name__ == "__main__":
    main()
