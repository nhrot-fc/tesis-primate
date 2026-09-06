import numpy as np
import numpy.typing as npt

ALPHA = 0.8  # factor de olvido de la media movil, el del paper

Matrix = npt.NDArray[np.float64]


# Algoritmo 1 del paper: cada grupo de cajas pasa a ser un ejemplo de clasificacion
# multietiqueta con M+1 clases, donde la ultima es el fondo.
def reduce(
    groups: npt.NDArray[np.int64],
    categories: npt.NDArray[np.int64],
    scores: Matrix,
    n_classes: int,
) -> tuple[npt.NDArray[np.bool_], Matrix]:
    n = int(groups.max()) + 1 if len(groups) else 0
    truth = np.zeros((n, n_classes + 1), dtype=bool)
    probs = np.zeros((n, n_classes + 1))
    annotated = np.isnan(scores)  # las anotaciones no traen score, las predicciones si

    np.logical_or.at(truth, (groups[annotated], categories[annotated]), True)
    np.maximum.at(probs, (groups[~annotated], categories[~annotated]), scores[~annotated])
    truth[:, -1] = ~truth[:, :-1].any(axis=1)  # sin anotacion, la etiqueta es fondo
    probs[:, -1] = probs[:, :-1].sum(axis=1) == 0  # sin prediccion, el modelo ve fondo
    return truth, probs


# Confident learning multietiqueta: auto-confianza por clase (uno contra el resto)
# agregada con una media movil exponencial sobre las clases de mejor a peor, de modo
# que la peor clase pesa alpha y domina la calidad del grupo.
def quality(truth: npt.NDArray[np.bool_], probs: Matrix, alpha: float = ALPHA) -> Matrix:
    confidence = np.where(truth, probs, 1.0 - probs)
    ordered = -np.sort(-confidence, axis=1)
    moving = ordered[:, 0]
    for column in ordered.T[1:]:
        moving = alpha * column + (1 - alpha) * moving
    return moving
