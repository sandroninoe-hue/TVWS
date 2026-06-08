"""
generar_dataset_sintetico.py
============================
Genera un dataset sintético de muestras espectrales TVWS para probar
el pipeline de entrenamiento sin necesidad del RTL-SDR físico.

Las muestras simulan lo que produciría captura_dataset.py + construir_dataset.py
con un RTL-SDR V4 real: vectores PSD de 512 puntos, normalizados [0,1],
con firmas espectrales realistas de canales TVWS ocupados y libres.

Señales simuladas:
  - Canal libre:    ruido térmico gaussiano con piso plano
  - Canal ocupado:  señal ISDB-Tb (lóbulo plano + flancos de rolloff coseno alzado)
  - Artefacto DC:   pico en bin central (LO leakage del RTL-SDR)
  - Variabilidad:   SNR, offset de frecuencia, nivel de ruido aleatorios

Uso:
  python generar_dataset_sintetico.py
  python generar_dataset_sintetico.py --n_muestras 2000 --salida ./mi_dataset
  python generar_dataset_sintetico.py --n_muestras 500 --balance 0.4

Salida:
  dataset/
    train/   muestra_XXXXXX.npz
    val/     muestra_XXXXXX.npz
    test/    muestra_XXXXXX.npz
    metadata.json
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DEL SISTEMA (deben coincidir con spectral_sense.py)
# ─────────────────────────────────────────────────────────────────────────────

N_FFT        = 512        # puntos del vector PSD
N_CANALES    = 9          # canales por sub-banda
SAMPLE_RATE  = 2.4e6      # Hz — simula RTL-SDR V4

POSICIONES_BARRIDO = [
    {"centro_mhz": 498, "canales_globales": list(range(14, 23))},
    {"centro_mhz": 536, "canales_globales": list(range(22, 31))},
    {"centro_mhz": 578, "canales_globales": list(range(30, 39))},
    {"centro_mhz": 626, "canales_globales": list(range(38, 47))},
    {"centro_mhz": 670, "canales_globales": list(range(46, 53))},
]


# ─────────────────────────────────────────────────────────────────────────────
# GENERADORES DE SEÑAL SINTÉTICA
# ─────────────────────────────────────────────────────────────────────────────

def psd_canal_libre(n_bins: int = N_FFT,
                    nivel_ruido_db: float = None,
                    rng: np.random.Generator = None) -> np.ndarray:
    """
    Genera la PSD de un canal libre: solo ruido térmico gaussiano.

    El ruido térmico en frecuencia produce una PSD aproximadamente plana
    con fluctuaciones gaussianas (distribución chi-cuadrado en potencia).
    """
    if rng is None:
        rng = np.random.default_rng()
    if nivel_ruido_db is None:
        nivel_ruido_db = rng.uniform(-85, -75)   # dBm, rango típico RTL-SDR

    # Ruido térmico: potencia chi-cuadrado con 2 grados de libertad
    potencia_lineal = rng.exponential(scale=1.0, size=n_bins)
    psd_db          = 10 * np.log10(potencia_lineal + 1e-12) + nivel_ruido_db

    # Variación suave de la respuesta en frecuencia del RTL-SDR
    # (no es perfectamente plana — hay una leve curvatura)
    respuesta_rtlsdr = _respuesta_frecuencia_rtlsdr(n_bins, rng)
    psd_db          += respuesta_rtlsdr

    return psd_db


def psd_canal_ocupado_isdb(n_bins: int = N_FFT,
                            snr_db: float = None,
                            offset_bins: int = None,
                            nivel_ruido_db: float = None,
                            rng: np.random.Generator = None) -> np.ndarray:
    """
    Genera la PSD de un canal ocupado con señal ISDB-Tb (TV digital peruana).

    ISDB-Tb usa OFDM con ~5.6 MHz de ocupación efectiva dentro de un canal
    de 6 MHz. La forma espectral es un lóbulo rectangular suavizado por
    filtros de coseno alzado en los flancos — característica que el modelo
    debe aprender a reconocer.

    Características simuladas:
      - Lóbulo plano central (~46 bins de los 55 del canal)
      - Flancos de rolloff tipo coseno alzado (~4-5 bins cada flanco)
      - Pilotos continuos OFDM (pequeñas elevaciones periódicas en el lóbulo)
      - Ruido de fase alrededor de la portadora (ruido próximo)
    """
    if rng is None:
        rng = np.random.default_rng()
    if snr_db is None:
        snr_db = rng.uniform(10, 35)             # dB sobre el piso de ruido
    if offset_bins is None:
        offset_bins = rng.integers(-3, 4)        # offset de frecuencia ±3 bins
    if nivel_ruido_db is None:
        nivel_ruido_db = rng.uniform(-85, -75)

    # 1. Base de ruido
    psd_db = psd_canal_libre(n_bins, nivel_ruido_db, rng)

    # 2. Construir la envolvente espectral de ISDB-Tb
    centro      = n_bins // 2 + offset_bins
    ancho_canal = int(round(6e6 / (SAMPLE_RATE / n_bins)))   # ~55 bins para 6 MHz
    ancho_plano = int(ancho_canal * 0.82)                     # ~45 bins lóbulo plano
    ancho_flanco= (ancho_canal - ancho_plano) // 2            # ~5 bins por flanco

    envolvente = np.zeros(n_bins)

    # Lóbulo plano
    inicio_plano = centro - ancho_plano // 2
    fin_plano    = centro + ancho_plano // 2
    inicio_plano = np.clip(inicio_plano, 0, n_bins)
    fin_plano    = np.clip(fin_plano,    0, n_bins)
    envolvente[inicio_plano:fin_plano] = snr_db

    # Flancos de coseno alzado (rolloff)
    for lado, inicio_f in [(-1, inicio_plano - ancho_flanco),
                            ( 1, fin_plano)]:
        for k in range(ancho_flanco):
            idx = inicio_f + k if lado == 1 else inicio_plano - ancho_flanco + k
            if 0 <= idx < n_bins:
                fase = np.pi * k / ancho_flanco
                envolvente[idx] = snr_db * 0.5 * (1 + np.cos(fase + np.pi))

    # 3. Pilotos continuos ISDB-Tb (subportadoras de referencia)
    # Aparecen como pequeñas elevaciones periódicas dentro del lóbulo
    paso_piloto = max(1, ancho_plano // 8)
    for k in range(inicio_plano, fin_plano, paso_piloto):
        if 0 <= k < n_bins:
            envolvente[k] += rng.uniform(1.5, 3.0)   # +2-3 dB sobre el lóbulo

    # 4. Ruido de fase (phase noise) — elevación suave cerca de la portadora
    ruido_fase = _ruido_fase(n_bins, centro, ancho_canal, rng)
    envolvente += ruido_fase

    # 5. Sumar envolvente al piso de ruido
    psd_db += envolvente

    # 6. Variación de respuesta en frecuencia del RTL-SDR
    psd_db += _respuesta_frecuencia_rtlsdr(n_bins, rng)

    return psd_db


def _respuesta_frecuencia_rtlsdr(n_bins: int,
                                  rng: np.random.Generator) -> np.ndarray:
    """
    Simula la respuesta en frecuencia no plana del RTL-SDR:
    leve rolloff en los bordes y curvatura suave en el centro.
    """
    x         = np.linspace(-1, 1, n_bins)
    amplitud  = rng.uniform(1.5, 3.5)    # dB de variación total
    curvatura = rng.uniform(0.3, 0.7)    # qué tan pronunciada es la caída en bordes
    return -amplitud * (np.abs(x) ** curvatura)


def _ruido_fase(n_bins: int, centro: int,
                ancho_canal: int,
                rng: np.random.Generator) -> np.ndarray:
    """
    Simula el ruido de fase (phase noise) de un oscilador real:
    caída de potencia proporcional a 1/f² alrededor de la portadora,
    visible como elevación de ruido próximo al lóbulo de la señal.
    """
    resultado  = np.zeros(n_bins)
    amplitud   = rng.uniform(0.5, 2.0)   # intensidad del ruido de fase
    radio      = ancho_canal * 0.8

    for i in range(n_bins):
        distancia = abs(i - centro)
        if 0 < distancia < radio:
            resultado[i] = amplitud / (1 + (distancia / (radio * 0.1)) ** 2)

    return resultado


def _artefacto_dc(psd_db: np.ndarray,
                  rng: np.random.Generator,
                  intensidad_db: float = None) -> np.ndarray:
    """
    Añade el artefacto de DC offset (LO leakage) del RTL-SDR:
    pico espurio en el bin central que no corresponde a ninguna señal real.
    Está presente en TODAS las capturas del RTL-SDR, libre u ocupado.
    """
    if intensidad_db is None:
        intensidad_db = rng.uniform(8, 20)   # dB sobre el entorno

    psd_resultado = psd_db.copy()
    centro        = len(psd_db) // 2
    n_afectados   = rng.integers(1, 4)       # 1-3 bins afectados

    for k in range(-n_afectados, n_afectados + 1):
        idx = centro + k
        if 0 <= idx < len(psd_db):
            factor = 1.0 - abs(k) / (n_afectados + 1)
            psd_resultado[idx] += intensidad_db * factor

    return psd_resultado


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZACIÓN (idéntica a spectral_sense.py)
# ─────────────────────────────────────────────────────────────────────────────

def normalizar_psd(psd_raw: np.ndarray) -> np.ndarray:
    """
    Normalización por percentiles robustos [0, 1].
    DEBE ser idéntica a la función en spectral_sense.py.
    """
    psd_min  = np.percentile(psd_raw, 5)
    psd_max  = np.percentile(psd_raw, 95)
    psd_norm = (psd_raw - psd_min) / (psd_max - psd_min + 1e-9)
    return np.clip(psd_norm, 0.0, 1.0).astype(np.float32)


def corregir_dc_offset(psd: np.ndarray, n_bins_mascara: int = 3) -> np.ndarray:
    """Enmascara el pico de DC interpolando con bins adyacentes."""
    psd_c  = psd.copy()
    centro = len(psd) // 2
    inicio = centro - n_bins_mascara
    fin    = centro + n_bins_mascara + 1
    inicio = max(0, inicio)
    fin    = min(len(psd), fin)
    izq    = psd[max(0, inicio - 1)]
    der    = psd[min(len(psd) - 1, fin)]
    psd_c[inicio:fin] = np.linspace(izq, der, fin - inicio)
    return psd_c


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR DE MUESTRAS COMPLETAS
# ─────────────────────────────────────────────────────────────────────────────

def generar_muestra(ocupado: bool,
                    canal_global: int,
                    posicion_idx: int,
                    rng: np.random.Generator) -> dict:
    """
    Genera una muestra completa lista para guardar en .npz.

    Simula el pipeline real:
      RTL-SDR capta canal → calcular_psd() → corregir_dc() → normalizar_psd()

    Args:
        ocupado:      True si el canal tiene señal primaria
        canal_global: número de canal TVWS peruano (14–52)
        posicion_idx: índice de posición de barrido (0–4)
        rng:          generador de números aleatorios (para reproducibilidad)

    Returns:
        dict con psd (512,), etiquetas (9,), posicion (int)
    """
    # 1. Generar PSD cruda según ocupación
    if ocupado:
        psd_db = psd_canal_ocupado_isdb(rng=rng)
    else:
        psd_db = psd_canal_libre(rng=rng)

    # 2. Añadir artefacto DC del RTL-SDR (siempre presente)
    psd_db = _artefacto_dc(psd_db, rng)

    # 3. Pipeline de preprocesamiento (igual que en campo)
    psd_db   = corregir_dc_offset(psd_db)
    psd_norm = normalizar_psd(psd_db)

    # 4. Construir vector de etiquetas
    canales_posicion = POSICIONES_BARRIDO[posicion_idx]["canales_globales"]
    etiquetas        = np.full(N_CANALES, -1.0, dtype=np.float32)
    if canal_global in canales_posicion:
        idx_local            = canales_posicion.index(canal_global)
        etiquetas[idx_local] = 1.0 if ocupado else 0.0

    return {
        "psd":       psd_norm,
        "etiquetas": etiquetas,
        "posicion":  posicion_idx
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL DATASET
# ─────────────────────────────────────────────────────────────────────────────

def construir_dataset(n_muestras: int   = 1000,
                      balance_ocupado: float = 0.35,
                      salida: str       = "./dataset",
                      semilla: int      = 42) -> dict:
    """
    Genera el dataset completo y lo guarda en disco.

    Args:
        n_muestras:      número total de muestras a generar
        balance_ocupado: fracción de muestras de clase 'ocupado' (0.0–1.0)
        salida:          directorio raíz donde guardar train/val/test
        semilla:         semilla para reproducibilidad

    Returns:
        dict con estadísticas del dataset generado
    """
    rng = np.random.default_rng(semilla)

    # Crear directorios
    splits     = {"train": 0.70, "val": 0.15, "test": 0.15}
    contadores = {s: 0 for s in splits}
    for split in splits:
        os.makedirs(os.path.join(salida, split), exist_ok=True)

    # Calcular distribución de clases y splits
    n_ocupado = int(n_muestras * balance_ocupado)
    n_libre   = n_muestras - n_ocupado

    # Asignar splits por clase para mantener balance en cada split
    plan = []
    for ocupado, n in [(True, n_ocupado), (False, n_libre)]:
        n_train = int(n * splits["train"])
        n_val   = int(n * splits["val"])
        n_test  = n - n_train - n_val
        for split, cantidad in [("train", n_train),
                                 ("val",   n_val),
                                 ("test",  n_test)]:
            for _ in range(cantidad):
                plan.append((ocupado, split))

    rng.shuffle(plan)

    # Estadísticas
    stats = {"libre": 0, "ocupado": 0}

    print(f"\nGenerando {n_muestras} muestras sintéticas TVWS...")
    print(f"  Ocupado: {n_ocupado} ({100*balance_ocupado:.0f}%)")
    print(f"  Libre:   {n_libre} ({100*(1-balance_ocupado):.0f}%)")
    print(f"  Splits:  train={int(n_muestras*0.70)} "
          f"val={int(n_muestras*0.15)} "
          f"test={n_muestras - int(n_muestras*0.70) - int(n_muestras*0.15)}")
    print()

    for ocupado, split in tqdm(plan, desc="Generando muestras"):
        # Elegir canal y posición aleatoriamente
        posicion_idx  = int(rng.integers(0, 5))
        canales_disp  = POSICIONES_BARRIDO[posicion_idx]["canales_globales"]
        canal_global  = int(rng.choice(canales_disp))

        muestra = generar_muestra(ocupado, canal_global, posicion_idx, rng)

        # Guardar
        idx      = contadores[split]
        ruta_npz = os.path.join(salida, split, f"muestra_{idx:06d}.npz")
        np.savez_compressed(
            ruta_npz,
            psd      = muestra["psd"],
            etiquetas= muestra["etiquetas"],
            posicion = np.array(muestra["posicion"], dtype=np.int8)
        )
        contadores[split] += 1
        stats["ocupado" if ocupado else "libre"] += 1

    # Guardar metadata
    total = sum(contadores.values())
    metadata = {
        "tipo":            "sintetico",
        "n_muestras":      contadores,
        "balance_clases":  {k: round(v / total, 3) for k, v in stats.items()},
        "n_fft":           N_FFT,
        "n_canales":       N_CANALES,
        "semilla":         semilla,
        "normalizacion":   "percentil_5_95",
        "dc_offset_corr":  True,
        "señales_sim":     ["ISDB-Tb (lóbulo plano + flancos coseno alzado)",
                            "ruido térmico gaussiano",
                            "artefacto DC RTL-SDR",
                            "ruido de fase",
                            "respuesta frecuencia no plana RTL-SDR"]
    }
    with open(os.path.join(salida, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICACIÓN VISUAL (opcional, requiere matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def verificar_muestras(salida: str = "./dataset", n: int = 4):
    """
    Grafica n muestras aleatorias de train para verificar que las señales
    sintéticas tienen el aspecto correcto.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no disponible — omitir verificación visual")
        return

    directorio = os.path.join(salida, "train")
    archivos   = [f for f in os.listdir(directorio) if f.endswith(".npz")]
    elegidos   = np.random.choice(archivos, size=min(n, len(archivos)),
                                  replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    axes      = axes.flatten()
    freqs     = np.linspace(-1.2, 1.2, N_FFT)   # MHz relativo al centro

    for ax, archivo in zip(axes, elegidos):
        data      = np.load(os.path.join(directorio, archivo))
        psd       = data["psd"]
        etiquetas = data["etiquetas"]

        # Determinar etiqueta visible (ignorar -1)
        etiq_validas = [(i, e) for i, e in enumerate(etiquetas) if e >= 0]
        if etiq_validas:
            idx_local, etiq = etiq_validas[0]
            estado = "OCUPADO" if etiq == 1.0 else "LIBRE"
            posicion = int(data["posicion"])
            canal = POSICIONES_BARRIDO[posicion]["canales_globales"][idx_local]
            titulo = f"Canal {canal} — {estado}"
        else:
            titulo = "Sin etiqueta"

        ax.plot(freqs, psd, linewidth=0.8, color="#378ADD")
        ax.axhline(0.5, color='r', linestyle='--', linewidth=0.8,
                   label='umbral 0.5')
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("Frecuencia relativa (MHz)", fontsize=8)
        ax.set_ylabel("PSD normalizada [0,1]", fontsize=8)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Muestras sintéticas TVWS — verificación visual", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(salida, "verificacion_visual.png"),
                dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Gráfica guardada en {salida}/verificacion_visual.png")


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Genera dataset sintético TVWS para probar spectral_sense.py"
    )
    parser.add_argument("--n_muestras", type=int,   default=1000,
                        help="Total de muestras a generar (default: 1000)")
    parser.add_argument("--balance",    type=float, default=0.35,
                        help="Fracción de clase 'ocupado' (default: 0.35)")
    parser.add_argument("--salida",     type=str,   default="./dataset",
                        help="Directorio de salida (default: ./dataset)")
    parser.add_argument("--semilla",    type=int,   default=42,
                        help="Semilla para reproducibilidad (default: 42)")
    parser.add_argument("--visualizar", action="store_true",
                        help="Mostrar gráficas de verificación tras generar")
    args = parser.parse_args()

    if not (0.1 <= args.balance <= 0.9):
        print("ERROR: --balance debe estar entre 0.1 y 0.9")
        return

    metadata = construir_dataset(
        n_muestras      = args.n_muestras,
        balance_ocupado = args.balance,
        salida          = args.salida,
        semilla         = args.semilla
    )

    print("\nDataset generado:")
    for split, n in metadata["n_muestras"].items():
        print(f"  {split:6s}: {n:5d} muestras")
    print(f"  Balance: libre={metadata['balance_clases']['libre']:.1%} "
          f"| ocupado={metadata['balance_clases']['ocupado']:.1%}")
    print(f"  Guardado en: {args.salida}/")
    print(f"\nPróximo paso:")
    print(f"  python spectral_sense.py --modo entrenar --dataset {args.salida}")

    if args.visualizar:
        verificar_muestras(args.salida)


if __name__ == "__main__":
    main()
