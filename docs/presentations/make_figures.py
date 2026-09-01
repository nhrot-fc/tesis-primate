import json, re, numpy as np, matplotlib
from scipy.stats import spearmanr
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = "docs/presentations/figures"
CKPT = "checkpoints"

MODELS = [  # (etiqueta, prefijo de archivo, color)
    ("Faster R-CNN", "frcnn_coco_25cls_best",            "#2a78d6"),
    ("DETR ts5",     "detr_pcen_ts5_frozen_25cls_best",  "#eb6834"),
    ("DETR ts10",    "detr_pcen_ts10_frozen_25cls_best", "#1baf7a"),
    ("YOLO26",       "best_yolo",                        "#4a3aa7"),
]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e2e1dc"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200,
})

def load(pref, split):
    with open(f"{CKPT}/{pref}_{split}_metrics.json") as f:
        return json.load(f)

def clases(pref, split):
    """Nombres de clase en orden, leidos del .txt (el json solo trae indices)."""
    txt = open(f"{CKPT}/{pref}_{split}_metrics.txt").read()
    names = []
    for line in txt.splitlines():
        m = re.match(r"^(\S+/\S+)\s+[\d.]+\s+[\d.]+$", line.strip())
        if m: names.append(m.group(1))
    return names

D = {}
for lbl, pref, col in MODELS:
    D[lbl] = {s: load(pref, s) for s in ("val", "test")}
CLS = clases(MODELS[0][1], "test")
COL = {lbl: c for lbl, _, c in MODELS}
LBLS = [m[0] for m in MODELS]

# soporte por clase (cajas anotadas en test) desde boxes_per_split.json
bps = json.load(open("research/figures/data/boxes_per_split.json"))
SUP = {c: n for c, n in zip(bps["clase"], bps["test"])}

def bar_labels(ax, bars, fmt="{:.3f}", dx=0.006):
    for b in bars:
        w = b.get_width()
        ax.text(w + dx, b.get_y() + b.get_height()/2, fmt.format(w),
                va="center", ha="left", fontsize=9, color=INK2)

# ---------------------------------------------------------------- Figura A
# Small multiples: una metrica por panel, barras horizontales por modelo.
METS = [("recall", "Recall"), ("precision", "Precisión"), ("f_beta", "F3"),
        ("map_50", "mAP@0.5"), ("map_50_95", "mAP@0.5:0.95")]
fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.4))
for ax, (key, name) in zip(axes, METS):
    vals = [D[l]["test"]["metrics"][key] for l in LBLS]
    y = np.arange(len(LBLS))[::-1]
    bars = ax.barh(y, vals, height=0.62, color=[COL[l] for l in LBLS])
    bar_labels(ax, bars, dx=0.02)
    ax.set_yticks(y); ax.set_yticklabels(LBLS if ax is axes[0] else [], fontsize=10)
    ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.5, 1.0])
    ax.set_title(name, fontsize=11, color=INK, pad=8)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(length=0)
fig.suptitle("Split de prueba · punto de operación score ≥ 0,5 · IoU ≥ 0,5",
             fontsize=11, color=INK2, y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/metrics_test.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Figura B
# val -> test, un panel por metrica (slope chart)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
for ax, (key, name) in zip(axes, [("recall", "Recall"), ("f_beta", "F3"), ("map_50", "mAP@0.5")]):
    puntos = []
    for l in LBLS:
        v = D[l]["val"]["metrics"][key]; t = D[l]["test"]["metrics"][key]
        ax.plot([0, 1], [v, t], "-o", color=COL[l], lw=2, ms=7, mec="white", mew=1.5)
        puntos.append([t, l])
    # separa las etiquetas que caerían una encima de otra
    puntos.sort()
    for i in range(1, len(puntos)):
        if puntos[i][0] - puntos[i-1][0] < 0.045:
            puntos[i][0] = puntos[i-1][0] + 0.045
    for ypos, l in puntos:
        ax.text(1.06, ypos, f"{l} {D[l]['test']['metrics'][key]:.3f}",
                va="center", fontsize=9, color=COL[l])
    ax.set_xlim(-0.15, 1.9); ax.set_xticks([0, 1]); ax.set_xticklabels(["val", "test"])
    ax.set_ylim(0, 0.9); ax.set_title(name, fontsize=11, color=INK, pad=8)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=0)
fig.suptitle("Validación frente a prueba: el orden entre modelos no cambia",
             fontsize=11, color=INK2, y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/val_vs_test.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Figura C
# Recall y AP@0.5 por clase (test), clases ordenadas por soporte
order = sorted(range(len(CLS)), key=lambda i: SUP.get(CLS[i], 0))
names = [f"{CLS[i]}  ({SUP.get(CLS[i],0)})" for i in order]
fig, axes = plt.subplots(1, 2, figsize=(13, 8.6), sharey=True)
h = 0.2
for ax, (key, name) in zip(axes, [("recall_per_class", "Recall al punto de operación"),
                                  ("ap_per_class_50", "AP@0.5")]):
    for k, l in enumerate(LBLS):
        v = [D[l]["test"]["metrics"][key][str(i)] for i in order]
        y = np.arange(len(order)) + (1.5 - k) * h
        ax.barh(y, v, height=h * 0.92, color=COL[l], label=l)
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_title(name, fontsize=12, color=INK, pad=10)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=0)
axes[0].set_yticks(np.arange(len(order))); axes[0].set_yticklabels(names, fontsize=9)
axes[0].set_ylim(-0.6, len(order) - 0.4)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=4,
           frameon=False, fontsize=11, handlelength=1.2)
fig.suptitle("Por clase, split de prueba (entre paréntesis: cajas anotadas en test)",
             fontsize=11, color=INK2, y=1.005)
fig.tight_layout(rect=[0, 0, 1, 0.955]); fig.savefig(f"{OUT}/per_class_test.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Figura D
# Recall por clase contra soporte, un panel por modelo
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), sharey=True)
for ax, l in zip(axes, LBLS):
    xs = [SUP.get(c, 0) for c in CLS]
    ys = [D[l]["test"]["metrics"]["recall_per_class"][str(i)] for i in range(len(CLS))]
    ax.scatter(xs, ys, s=44, color=COL[l], alpha=0.85, edgecolor="white", lw=1)
    ax.axvline(100, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=0)
    rho = spearmanr(xs, ys).statistic
    ax.set_xscale("log"); ax.set_xlim(20, 3000); ax.set_ylim(0, 1)
    ax.set_title(f"{l}   ρ = {rho:.2f}", fontsize=11, color=INK, pad=8)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.set_xlabel("cajas en test (log)", fontsize=9)
axes[0].set_ylabel("recall", fontsize=10)
fig.suptitle("Recall por clase contra su soporte (ρ de Spearman; línea: 100 cajas)",
             fontsize=11, color=INK2, y=1.03)
fig.tight_layout(); fig.savefig(f"{OUT}/recall_vs_support.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Figura E
# Plano precision-recall: frontera de Pareto sobre test, con isolineas de F3
fig, ax = plt.subplots(figsize=(7.6, 5.2))
b2 = 9
for f3 in (0.4, 0.5, 0.6, 0.7, 0.8):
    r = np.linspace(0.02, 1, 400)
    p = (f3 * b2 * r) / ((1 + b2) * r - f3)   # despeje de F_beta
    m = (p > 0) & (p <= 1.05)
    ax.plot(r[m], p[m], color=GRID, lw=1, zorder=1)
    xl = 0.435                                 # etiquetas a la izquierda, sin datos
    pl = (f3 * b2 * xl) / ((1 + b2) * xl - f3)
    if 0.36 < pl < 0.94:
        ax.text(xl, pl + 0.008, f"F3={f3:g}", fontsize=8, color=MUTED,
                ha="left", va="bottom")

P = {l: (D[l]["test"]["metrics"]["recall"], D[l]["test"]["metrics"]["precision"])
     for l in LBLS}
def dominado(l):
    x, y = P[l]
    return any(P[o][0] >= x and P[o][1] >= y and P[o] != P[l] for o in LBLS)
frente = sorted([l for l in LBLS if not dominado(l)], key=lambda l: P[l][0])

# escalera de Pareto: la region que ningun modelo alcanza queda a la derecha-arriba
xs, ys = [], []
for i, l in enumerate(frente):
    x, y = P[l]
    if i: xs += [x]; ys += [ys[-1]]
    xs += [x]; ys += [y]
ax.plot(xs, ys, color=MUTED, lw=1.6, ls=(0, (5, 3)), zorder=2)
ax.text(0.706, 0.70, "frontera de Pareto", fontsize=10, color=MUTED, ha="left")
ax.text(0.80, 0.76, "nadie llega acá", fontsize=9.5, color=MUTED, ha="center", style="italic")

for l in LBLS:
    x, y = P[l]
    v = D[l]["val"]["metrics"]
    ax.scatter(v["recall"], v["precision"], s=90, color=COL[l], alpha=0.35,
               edgecolor="white", lw=1.4, zorder=3)
    if dominado(l):                                   # marcador hueco
        ax.scatter(x, y, s=170, facecolor="white", edgecolor=COL[l], lw=2.4, zorder=4)
    else:
        ax.scatter(x, y, s=170, color=COL[l], edgecolor="white", lw=1.6, zorder=4)
    off = {"Faster R-CNN": (-24, 14), "DETR ts5": (-4, 15), "DETR ts10": (-6, -22),
           "YOLO26": (13, 6)}[l]
    txt = l + ("  (dominado)" if dominado(l) else "")
    ax.annotate(txt, (x, y), textcoords="offset points", xytext=off,
                fontsize=10, color=COL[l])

ax.set_xlim(0.4, 0.9); ax.set_ylim(0.35, 0.95)
ax.set_xlabel("recall"); ax.set_ylabel("precisión")
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=0)
ax.set_title("Test (relleno) y val (translúcido) · líneas finas: isolíneas de F3",
             fontsize=10.5, color=INK2, pad=10)
fig.tight_layout(); fig.savefig(f"{OUT}/pr_plane.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Figura F
# Costo de revision: cajas propuestas por caja verdadera recuperada (test)
fig, ax = plt.subplots(figsize=(7.6, 3.2))
GT = 7985
vals, hits = [], []
for l in LBLS:
    m = D[l]["test"]["metrics"]
    tp = m["recall"] * GT
    dets = tp / m["precision"]      # cajas emitidas con score >= 0.5
    vals.append(dets / tp); hits.append((tp, dets))
y = np.arange(len(LBLS))[::-1]
bars = ax.barh(y, vals, height=0.6, color=[COL[l] for l in LBLS])
for b, (tp, dets) in zip(bars, hits):
    ax.text(b.get_width() + 0.03, b.get_y() + b.get_height()/2,
            f"{b.get_width():.2f}   ({dets:,.0f} cajas para {tp:,.0f} aciertos)".replace(",", " "),
            va="center", fontsize=9, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(LBLS, fontsize=10)
ax.set_xlim(0, 3.6); ax.set_xlabel("cajas propuestas por caja verdadera recuperada")
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=0)
ax.set_title("Carga de revisión en test (7 985 cajas anotadas)", fontsize=11, color=INK, pad=10)
fig.tight_layout(); fig.savefig(f"{OUT}/review_cost.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Figura G
# Soporte por clase: cajas de cada split, apiladas (para la lámina de dataset)
tot = [t + v + te for t, v, te in zip(bps["train"], bps["val"], bps["test"])]
idx = sorted(range(len(bps["clase"])), key=lambda i: tot[i])
fig, ax = plt.subplots(figsize=(8.0, 5.4))
y = np.arange(len(idx))
izq = np.zeros(len(idx))
for split, color in (("train", "#2a78d6"), ("val", "#eb6834"), ("test", "#1baf7a")):
    v = np.array([bps[split][i] for i in idx], dtype=float)
    ax.barh(y, v, left=izq, height=0.68, color=color, label=split)
    izq += v + 40                                    # separador de 2 px entre tramos
for i, k in enumerate(idx):
    ax.text(tot[k] + 240, i, f"{tot[k]:,}".replace(",", " "),
            va="center", fontsize=11, color=INK2)
ax.set_yticks(y); ax.set_yticklabels([bps["clase"][i] for i in idx], fontsize=12)
ax.set_ylim(-0.7, len(idx) - 0.3); ax.set_xlim(0, 13200)
ax.set_xlabel("cajas anotadas", fontsize=12)
ax.tick_params(axis="x", labelsize=11)
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=0)
ax.legend(loc="lower right", frameon=False, fontsize=13, handlelength=1.2)
ax.set_title("Cajas por clase y split (45 097 en total)", fontsize=14, color=INK, pad=10)
fig.tight_layout(); fig.savefig(f"{OUT}/dataset_support.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------- cifras sueltas
print("clases:", len(CLS))
for l in LBLS:
    for s in ("val", "test"):
        m = D[l][s]["metrics"]
        print(f"{l:14s} {s:5s} R={m['recall']:.3f} P={m['precision']:.3f} F3={m['f_beta']:.3f} "
              f"mAP50={m['map_50']:.3f} mAP5095={m['map_50_95']:.3f}")
# clases con recall < 0.35 en test para cada modelo
for l in LBLS:
    bad = [CLS[i] for i in range(len(CLS)) if D[l]["test"]["metrics"]["recall_per_class"][str(i)] < 0.35]
    print(l, "recall<0.35:", len(bad), bad)
# cuantas clases gana cada modelo (recall test)
win = {l: 0 for l in LBLS}
for i in range(len(CLS)):
    best = max(LBLS, key=lambda l: D[l]["test"]["metrics"]["recall_per_class"][str(i)])
    win[best] += 1
print("clases ganadas (recall test):", win)
