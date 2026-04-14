"""Geração e salvamento de matriz de confusão normalizada."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    save_path: str | Path | None = None,
    title: str = "Confusion Matrix (normalizada)",
) -> plt.Figure:
    """
    Plota e salva a matriz de confusão normalizada por linha (recall).

    Args:
        y_true: Rótulos reais.
        y_pred: Predições do modelo.
        label_names: Nomes das classes na ordem dos índices.
        save_path: Caminho para salvar a figura (opcional).
        title: Título do gráfico.

    Returns:
        Figura matplotlib.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    n = len(label_names)
    fig_size = max(10, n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel("Predito", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
