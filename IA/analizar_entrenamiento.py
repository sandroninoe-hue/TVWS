"""
analizar_entrenamiento.py
=========================
Analiza y visualiza las métricas del entrenamiento de SpectralSenseCNN
a partir del historial.json generado por spectral_sense.py.

Uso:
  python analizar_entrenamiento.py
  python analizar_entrenamiento.py --historial ./modelos/historial.json
  python analizar_entrenamiento.py --historial ./modelos/historial.json --salida ./graficas
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

# ─── Estilo visual ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#1a1d27",
    "axes.edgecolor":    "#3a3d4f",
    "axes.labelcolor":   "#c8cad4",
    "axes.titlecolor":   "#e8eaf0",
    "xtick.color":       "#8890a8",
    "ytick.color":       "#8890a8",
    "grid.color":        "#2a2d3f",
    "grid.linewidth":    0.6,
    "text.color":        "#c8cad4",
    "legend.facecolor":  "#1a1d27",
    "legend.edgecolor":  "#3a3d4f",
    "font.family":       "monospace",
})

AZUL    = "#4f9cf9"
VERDE   = "#4fc97e"
NARANJA = "#f9844f"
MORADO  = "#a78bfa"
ROJO    = "#f97070"
GRIS    = "#8890a8"


def cargar_historial(ruta: str) -> list[dict]:
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def resumen_consola(h: list[dict]):
    """Imprime resumen de métricas clave en consola."""
    epocas       = [e["epoca"]          for e in h]
    auc_vals     = [e["auc_media"]      for e in h]
    f1_vals      = [e["f1_beta2"]       for e in h]
    loss_train   = [e["perdida_train"]  for e in h]
    loss_val     = [e["perdida"]        for e in h]

    mejor_epoca  = epocas[np.argmax(auc_vals)]
    mejor_auc    = max(auc_vals)
    mejor_f1     = max(f1_vals)
    n_epocas     = len(h)

    # Detectar overfitting: diferencia train-val en últimas 5 épocas
    if n_epocas >= 5:
        diff_reciente = np.mean([
            abs(loss_val[i] - loss_train[i])
            for i in range(-5, 0)
        ])
        overfitting = diff_reciente > abs(np.mean(loss_train[-5:])) * 0.15
    else:
        overfitting = False

    print("\n" + "═" * 56)
    print("  RESUMEN DEL ENTRENAMIENTO")
    print("═" * 56)
    print(f"  Épocas completadas   : {n_epocas}")
    print(f"  Mejor AUC-ROC        : {mejor_auc:.4f}  (época {mejor_epoca})")
    print(f"  Mejor F1(β=2)        : {mejor_f1:.4f}")
    print(f"  Loss train final     : {loss_train[-1]:.4f}")
    print(f"  Loss val final       : {loss_val[-1]:.4f}")
    print()

    # Diagnóstico
    print("  DIAGNÓSTICO:")
    if mejor_auc >= 0.92:
        print("  ✓ AUC excelente (≥0.92) — modelo listo para pruebas")
    elif mejor_auc >= 0.85:
        print("  ✓ AUC aceptable (≥0.85) — válido para despliegue inicial")
    elif mejor_auc >= 0.70:
        print("  ⚠ AUC moderado — ampliar dataset o ajustar hiperparámetros")
    else:
        print("  ✗ AUC bajo — revisar pipeline de datos y arquitectura")

    if mejor_f1 == 0.0:
        print("  ⚠ F1=0 en todas las épocas — modelo predice todo como 'libre'")
        print("    Causa probable: pocas muestras ocupadas o etiquetas -1 dominantes")
        print("    Solución: ampliar dataset con más muestras de clase ocupada")
    elif mejor_f1 >= 0.70:
        print("  ✓ F1(β=2) bueno — buena detección de canales ocupados")
    else:
        print("  ⚠ F1(β=2) bajo — ajustar umbral de decisión o balanceo de clases")

    if overfitting:
        print("  ⚠ Posible overfitting — gap train/val creciente en últimas épocas")
        print("    Solución: más datos, aumentar dropout, o reducir capacidad del modelo")
    else:
        print("  ✓ Sin señales claras de overfitting")

    print("═" * 56 + "\n")


def graficar_todo(h: list[dict], salida: str):
    """Genera figura con 4 subgráficas de métricas de entrenamiento."""
    epocas     = [e["epoca"]         for e in h]
    auc        = [e["auc_media"]     for e in h]
    f1         = [e["f1_beta2"]      for e in h]
    loss_train = [e["perdida_train"] for e in h]
    loss_val   = [e["perdida"]       for e in h]

    mejor_idx  = int(np.argmax(auc))
    mejor_ep   = epocas[mejor_idx]
    mejor_auc  = auc[mejor_idx]

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("SpectralSense CNN — Métricas de entrenamiento TVWS",
                 fontsize=13, y=0.97, color="#e8eaf0", fontweight="bold")

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.42, wspace=0.30,
                           left=0.07, right=0.97,
                           top=0.91, bottom=0.08)

    # ── Gráfica 1: Loss train vs val ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epocas, loss_train, color=AZUL,    linewidth=1.8,
             label="Train loss", marker="o", markersize=3)
    ax1.plot(epocas, loss_val,   color=NARANJA, linewidth=1.8,
             label="Val loss",   marker="o", markersize=3, linestyle="--")
    ax1.axvline(mejor_ep, color=GRIS, linewidth=0.8, linestyle=":")
    ax1.set_title("Pérdida (BCEWithLogitsLoss)", fontsize=10)
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True)
    _estilo_ax(ax1)

    # ── Gráfica 2: AUC-ROC ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epocas, auc, color=VERDE, linewidth=2,
             marker="o", markersize=4, label="AUC-ROC (val)")

    # Líneas de referencia
    for val, label, color in [(0.92, "Excelente (0.92)", VERDE),
                               (0.85, "Aceptable (0.85)", NARANJA),
                               (0.70, "Mínimo (0.70)",    ROJO)]:
        ax2.axhline(val, color=color, linewidth=0.7,
                    linestyle="--", alpha=0.6, label=label)

    # Marcador del mejor AUC
    ax2.scatter([mejor_ep], [mejor_auc], color=VERDE,
                s=80, zorder=5, label=f"Mejor: {mejor_auc:.4f} (ép.{mejor_ep})")
    ax2.axvline(mejor_ep, color=GRIS, linewidth=0.8, linestyle=":")
    ax2.set_title("AUC-ROC en validación", fontsize=10)
    ax2.set_xlabel("Época")
    ax2.set_ylabel("AUC")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=7)
    ax2.grid(True)
    _estilo_ax(ax2)

    # ── Gráfica 3: F1(β=2) ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.fill_between(epocas, f1, alpha=0.25, color=MORADO)
    ax3.plot(epocas, f1, color=MORADO, linewidth=2,
             marker="o", markersize=4, label="F1(β=2) (val)")
    ax3.axhline(0.70, color=VERDE,   linewidth=0.7, linestyle="--",
                alpha=0.6, label="Meta (0.70)")
    ax3.axvline(mejor_ep, color=GRIS, linewidth=0.8, linestyle=":")
    ax3.set_title("F1(β=2) en validación", fontsize=10)
    ax3.set_xlabel("Época")
    ax3.set_ylabel("F1(β=2)")
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(fontsize=8)
    ax3.grid(True)
    _estilo_ax(ax3)

    # Anotación si F1 es 0 en todo
    if max(f1) == 0.0:
        ax3.text(0.5, 0.5,
                 "F1 = 0\nAmpliar dataset\ncon más muestras\nde clase 'ocupado'",
                 transform=ax3.transAxes, ha="center", va="center",
                 fontsize=9, color=NARANJA,
                 bbox=dict(boxstyle="round,pad=0.4",
                           facecolor="#1a1d27", edgecolor=NARANJA, alpha=0.85))

    # ── Gráfica 4: AUC vs F1 (scatter por época) ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    sc = ax4.scatter(auc, f1,
                     c=epocas, cmap="plasma",
                     s=50, zorder=3, alpha=0.85)
    ax4.scatter([mejor_auc], [auc[mejor_idx] and f1[mejor_idx]],
                color="white", s=100, zorder=5, marker="*")

    # Colorbar con épocas
    cb = plt.colorbar(sc, ax=ax4, pad=0.02)
    cb.set_label("Época", fontsize=8, color="#c8cad4")
    cb.ax.yaxis.set_tick_params(color="#8890a8")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#8890a8", fontsize=7)

    ax4.axvline(0.85, color=NARANJA, linewidth=0.7, linestyle="--",
                alpha=0.6, label="AUC meta (0.85)")
    ax4.set_title("AUC vs F1(β=2) por época", fontsize=10)
    ax4.set_xlabel("AUC-ROC")
    ax4.set_ylabel("F1(β=2)")
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(fontsize=8)
    ax4.grid(True)
    _estilo_ax(ax4)

    # Guardar
    os.makedirs(salida, exist_ok=True)
    ruta_png = os.path.join(salida, "metricas_entrenamiento.png")
    plt.savefig(ruta_png, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Gráfica guardada en: {ruta_png}")
    plt.show()


def graficar_diagnostico_loss(h: list[dict], salida: str):
    """
    Gráfica adicional: diferencia train-val (gap de generalización).
    Un gap creciente indica overfitting.
    """
    epocas     = [e["epoca"]         for e in h]
    loss_train = [e["perdida_train"] for e in h]
    loss_val   = [e["perdida"]       for e in h]
    gap        = [abs(v - t) for t, v in zip(loss_train, loss_val)]

    fig, ax = plt.subplots(figsize=(9, 4),
                           facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")

    ax.fill_between(epocas, gap, alpha=0.20, color=ROJO)
    ax.plot(epocas, gap, color=ROJO, linewidth=2,
            marker="o", markersize=3, label="|Val loss − Train loss|")

    # Tendencia (regresión lineal)
    if len(epocas) >= 4:
        z    = np.polyfit(epocas, gap, 1)
        tend = np.poly1d(z)
        ax.plot(epocas, tend(epocas), color=NARANJA,
                linewidth=1.2, linestyle="--",
                label=f"Tendencia (pendiente={z[0]:+.3f})")
        if z[0] > 0.5:
            ax.text(0.98, 0.90, "⚠ Gap creciente\n(posible overfitting)",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color=NARANJA,
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#1a1d27", edgecolor=NARANJA))

    ax.set_title("Gap de generalización (|Val loss − Train loss|)",
                 fontsize=11, color="#e8eaf0")
    ax.set_xlabel("Época", color="#c8cad4")
    ax.set_ylabel("Gap", color="#c8cad4")
    ax.legend(fontsize=8)
    ax.grid(True, color="#2a2d3f", linewidth=0.6)
    _estilo_ax(ax)
    plt.tight_layout()

    ruta_png = os.path.join(salida, "gap_generalizacion.png")
    plt.savefig(ruta_png, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Gráfica guardada en: {ruta_png}")
    plt.show()


def _estilo_ax(ax):
    """Aplica estilo oscuro consistente a un eje."""
    ax.tick_params(colors="#8890a8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3d4f")


def main():
    parser = argparse.ArgumentParser(
        description="Analiza métricas del entrenamiento SpectralSense"
    )
    parser.add_argument("--historial", default="./modelos/historial.json",
                        help="Ruta al historial.json (default: ./modelos/historial.json)")
    parser.add_argument("--salida",    default="./graficas",
                        help="Directorio donde guardar las gráficas (default: ./graficas)")
    args = parser.parse_args()

    if not os.path.exists(args.historial):
        print(f"ERROR: No se encontró {args.historial}")
        print("Asegúrate de haber corrido primero:")
        print("  python spectral_sense.py --modo entrenar --dataset ./dataset")
        return

    print(f"Cargando historial: {args.historial}")
    h = cargar_historial(args.historial)
    print(f"Épocas en historial: {len(h)}")

    resumen_consola(h)
    graficar_todo(h, args.salida)
    graficar_diagnostico_loss(h, args.salida)

    print(f"\nArchivos generados en: {args.salida}/")
    print("  metricas_entrenamiento.png — Loss, AUC, F1, scatter AUC vs F1")
    print("  gap_generalizacion.png     — Diagnóstico de overfitting")


if __name__ == "__main__":
    main()
