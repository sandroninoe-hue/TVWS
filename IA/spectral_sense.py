"""
spectral_sense.py
=================
Módulo principal de IA para sensado espectral TVWS — Radio Cognitiva UNI 2024-2026.

Contiene:
  - SpectralSenseCNN      : definición del modelo PyTorch
  - normalizar_psd()      : preprocesamiento de PSD (idéntico en train e inferencia)
  - SpectralSenseInferencia: wrapper de ONNX Runtime para inferencia en campo
  - entrenar()            : bucle de entrenamiento completo
  - exportar_onnx()       : exportación del modelo entrenado a ONNX
  - main()                : punto de entrada para entrenamiento y exportación

Uso:
  # Entrenamiento:
  python spectral_sense.py --modo entrenar --dataset ./dataset --salida ./modelos

  # Exportar a ONNX:
  python spectral_sense.py --modo exportar --checkpoint ./modelos/mejor_modelo.pt

  # Test de inferencia rápida:
  python spectral_sense.py --modo test

Requisitos:
  pip install torch torchvision torchaudio numpy scikit-learn onnx onnxruntime
"""

import os
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, fbeta_score


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DEL SISTEMA
# ─────────────────────────────────────────────────────────────────────────────

# Plan de atribución TVWS peruano: canales 14–52, 6 MHz cada uno
CANALES_TVWS_PE = list(range(14, 53))  # 39 canales en total

# 5 posiciones de sintonización del bladeRF RX2 para cubrir 470–698 MHz
# Cada posición expone entre 7 y 9 canales TVWS en su zona plana
POSICIONES_BARRIDO = [
    {"centro_mhz": 498, "canales_globales": list(range(14, 23))},  # 9 canales
    {"centro_mhz": 536, "canales_globales": list(range(22, 31))},  # 9 canales
    {"centro_mhz": 578, "canales_globales": list(range(30, 39))},  # 9 canales
    {"centro_mhz": 626, "canales_globales": list(range(38, 47))},  # 9 canales
    {"centro_mhz": 670, "canales_globales": list(range(46, 53))},  # 7 canales
]

N_CANALES_MAX      = 9      # máximo de canales visibles en una sub-banda
PSD_LENGTH         = 512    # puntos del vector PSD de entrada
UMBRAL_LIBRE       = 0.20   # prob_ocupado < 0.20 → canal libre (conservador)
UMBRAL_SOSPECHOSO  = 0.65
UMBRAL_EMERGENCIA  = 0.95


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESAMIENTO — idéntico en entrenamiento e inferencia
# ─────────────────────────────────────────────────────────────────────────────

def calcular_psd(iq: np.ndarray, n_fft: int = 512, n_segmentos: int = 20) -> np.ndarray:
    """
    Calcula el PSD de un vector de muestras I/Q usando el método de Welch simplificado.

    Args:
        iq:           array complejo (N,) o real (N, 2) con columnas [I, Q]
        n_fft:        puntos de la FFT (determina la resolución espectral)
        n_segmentos:  número de segmentos a promediar (más = menos varianza)

    Returns:
        psd_db: array (n_fft,) con PSD en dB, centrado en DC (fftshift aplicado)
    """
    # Normalizar formato de entrada
    if iq.ndim == 2:
        iq = iq[:, 0] + 1j * iq[:, 1]

    ventana   = np.hanning(n_fft)
    paso      = max(1, (len(iq) - n_fft) // n_segmentos)
    acumulado = np.zeros(n_fft)
    n_validos = 0

    for k in range(n_segmentos):
        inicio = k * paso
        fin    = inicio + n_fft
        if fin > len(iq):
            break
        segmento     = iq[inicio:fin] * ventana
        espectro     = np.fft.fftshift(np.fft.fft(segmento))
        acumulado   += np.abs(espectro) ** 2
        n_validos   += 1

    if n_validos == 0:
        raise ValueError(f"Vector I/Q demasiado corto: {len(iq)} muestras para n_fft={n_fft}")

    psd_lineal = acumulado / n_validos
    psd_db     = 10 * np.log10(psd_lineal + 1e-12)
    return psd_db


def normalizar_psd(psd_raw: np.ndarray) -> np.ndarray:
    """
    Normaliza el vector PSD al rango [0, 1] usando percentiles robustos.

    CRÍTICO: Esta función debe ser IDÉNTICA en entrenamiento e inferencia.
    Cualquier diferencia invalida el modelo en campo.

    Mitiga el domain mismatch RTL-SDR (8 bits, ~48 dB) → bladeRF (12 bits, ~72 dB).
    Los percentiles 5/95 descartan picos de ruido espurio y el piso de ruido
    variable entre dispositivos, produciendo una distribución comparable.

    Args:
        psd_raw: array (N,) con PSD en cualquier escala (dB o lineal)

    Returns:
        array (N,) float32 normalizado en [0.0, 1.0]
    """
    psd_min  = np.percentile(psd_raw, 5)
    psd_max  = np.percentile(psd_raw, 95)
    psd_norm = (psd_raw - psd_min) / (psd_max - psd_min + 1e-9)
    return np.clip(psd_norm, 0.0, 1.0).astype(np.float32)


def iq_a_psd_normalizado(iq: np.ndarray, n_fft: int = 512) -> np.ndarray:
    """
    Pipeline completo: I/Q crudo → PSD normalizado listo para la CNN.
    Función de conveniencia que encadena calcular_psd() + normalizar_psd().
    """
    psd_raw = calcular_psd(iq, n_fft=n_fft)
    return normalizar_psd(psd_raw)


# ─────────────────────────────────────────────────────────────────────────────
# MODELO — SpectralSenseCNN
# ─────────────────────────────────────────────────────────────────────────────

class SpectralSenseCNN(nn.Module):
    """
    Red Neuronal Convolucional 1D para clasificación de ocupación espectral TVWS.

    Arquitectura de tres bloques con kernels decrecientes:
      Bloque 1 (k=45, ~5 MHz): detecta la silueta espectral del canal
      Bloque 2 (k=9,  ~1 MHz): detecta textura intra-canal (flancos, pilotos OFDM)
      Bloque 3 (k=5, ~0.5 MHz): correlaciones inter-canal en el feature map reducido

    Input:  (batch, 512)  — PSD normalizado [0, 1]
    Output: (batch, n_canales_salida)  — logits (sin Sigmoid)
    """

    def __init__(self, n_canales_salida: int = N_CANALES_MAX, dropout: float = 0.3):
        super().__init__()
        self.n_canales_salida = n_canales_salida

        # Bloque 1: estructura espectral gruesa
        # kernel=45 → campo receptivo ≈ 45 × (56 MHz / 512) ≈ 4.9 MHz
        self.bloque1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=45, padding=22),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)       # 512 → 256
        )

        # Bloque 2: detalle intra-canal
        # kernel=9 sobre feature map post-pool → campo receptivo ≈ 9 × 218 kHz ≈ 1.96 MHz
        self.bloque2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)       # 256 → 128
        )

        # Bloque 3: contexto inter-canal
        # kernel=5 sobre feature map post-pool → campo receptivo ≈ 5 × 437 kHz ≈ 2.2 MHz
        self.bloque3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
            # Sin MaxPool: conservar resolución para GlobalAveragePool
        )

        # Pooling global: 128 × 128 → (128,)
        # Evita la explosión de parámetros que causaría un Flatten
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Cabeza clasificadora multi-etiqueta
        self.cabeza = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, n_canales_salida)
            # Sin Sigmoid: BCEWithLogitsLoss lo incorpora durante entrenamiento
        )

        # Inicialización de pesos (He para ReLU)
        self._inicializar_pesos()

    def _inicializar_pesos(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (batch, 512) — PSD normalizado

        Returns:
            logits: tensor (batch, n_canales_salida) — sin activación
        """
        x = x.unsqueeze(1)          # (batch, 512) → (batch, 1, 512)
        x = self.bloque1(x)         # → (batch, 32, 256)
        x = self.bloque2(x)         # → (batch, 64, 128)
        x = self.bloque3(x)         # → (batch, 128, 128)
        x = self.gap(x).squeeze(-1) # → (batch, 128)
        return self.cabeza(x)       # → (batch, n_canales_salida)

    def predecir(self, x: torch.Tensor, umbral: float = UMBRAL_LIBRE) -> dict:
        """
        Inferencia con decisión de canal libre/ocupado.

        Args:
            x:      tensor (512,) o (1, 512) — PSD normalizado
            umbral: prob_ocupado < umbral → libre

        Returns:
            dict con prob_ocupado (array), libres (índices), ocupados (índices)
        """
        self.eval()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.sigmoid(logits).squeeze().cpu().numpy()

        return {
            "prob_ocupado": probs,
            "libres":       [i for i, p in enumerate(probs) if p < umbral],
            "ocupados":     [i for i, p in enumerate(probs) if p >= umbral]
        }

    def contar_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET — carga de muestras preprocesadas desde disco
# ─────────────────────────────────────────────────────────────────────────────

class TVWSDataset(Dataset):
    """
    Dataset de muestras espectrales TVWS.

    Espera archivos .npz con:
        psd:       array (512,) float32 — PSD normalizado
        etiquetas: array (9,)  float32 — 1.0=ocupado, 0.0=libre por canal
        posicion:  int — índice de posición de barrido (0–4)

    Estructura de directorio esperada:
        dataset/
            train/
                muestra_000001.npz
                muestra_000002.npz
                ...
            val/
                muestra_010001.npz
                ...
    """

    def __init__(self, directorio: str, split: str = "train",
                 augmentar: bool = True):
        self.directorio = os.path.join(directorio, split)
        self.augmentar  = augmentar and (split == "train")
        self.archivos   = sorted([
            f for f in os.listdir(self.directorio) if f.endswith(".npz")
        ])
        if len(self.archivos) == 0:
            raise RuntimeError(f"No se encontraron archivos .npz en {self.directorio}")

    def __len__(self) -> int:
        return len(self.archivos)

    def __getitem__(self, idx: int):
        ruta = os.path.join(self.directorio, self.archivos[idx])
        data = np.load(ruta)

        psd       = data["psd"].astype(np.float32)         # (512,)
        etiquetas = data["etiquetas"].astype(np.float32)   # (9,)

        if self.augmentar:
            psd = self._augmentar(psd)

        return torch.from_numpy(psd), torch.from_numpy(etiquetas)

    def _augmentar(self, psd: np.ndarray) -> np.ndarray:
        """
        Data augmentation físicamente realista para PSD espectral TVWS.
        Ver pipeline_dataset.md para justificación de cada técnica.
        """
        # 1. Ruido gaussiano aditivo (simula variación del piso de ruido)
        if np.random.rand() < 0.5:
            sigma = np.random.uniform(0.005, 0.02)
            psd   = psd + np.random.normal(0, sigma, psd.shape).astype(np.float32)

        # 2. Desplazamiento de amplitud global (simula variación de ganancia del SDR)
        if np.random.rand() < 0.4:
            offset = np.random.uniform(-0.05, 0.05)
            psd    = psd + offset

        # 3. Desplazamiento espectral sub-bin (simula CFO pequeño)
        if np.random.rand() < 0.3:
            desplazamiento = np.random.randint(-3, 4)   # máx ±3 bins ≈ ±327 kHz
            psd = np.roll(psd, desplazamiento)

        # 4. Escalado de amplitud (simula variación de RSSI entre capturas)
        if np.random.rand() < 0.3:
            factor = np.random.uniform(0.9, 1.1)
            psd    = psd * factor

        # Renormalizar tras augmentation y recortar a [0, 1]
        return np.clip(normalizar_psd(psd), 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def calcular_pos_weight(dataset_dir: str) -> torch.Tensor:
    """
    Calcula el peso de clase positiva (pos_weight) para BCEWithLogitsLoss.
    Ignora posiciones con etiqueta -1 (canal desconocido en esa captura).
    """
    suma_ocupado = np.zeros(N_CANALES_MAX)
    suma_libre   = np.zeros(N_CANALES_MAX)
    directorio   = os.path.join(dataset_dir, "train")

    for archivo in os.listdir(directorio):
        if not archivo.endswith(".npz"):
            continue
        data      = np.load(os.path.join(directorio, archivo))
        etiquetas = data["etiquetas"].astype(float)
        for i, e in enumerate(etiquetas):
            if e >= 0:
                suma_ocupado[i] += e
                suma_libre[i]   += (1.0 - e)

    if suma_ocupado.sum() == 0:
        raise RuntimeError("Dataset vacío o sin etiquetas válidas")

    pos_weight = suma_libre / (suma_ocupado + 1e-6)
    # Recortar valores extremos (canales sin muestras ocupadas dan pos_weight muy alto)
    pos_weight = np.clip(pos_weight, 0.5, 10.0)
    print(f"pos_weight calculado: {np.round(pos_weight, 2)}")
    return torch.tensor(pos_weight, dtype=torch.float32)


def evaluar(modelo: nn.Module, loader: DataLoader,
            criterio: nn.Module, device: torch.device) -> dict:
    """Evaluación completa sobre un DataLoader. Retorna métricas."""
    modelo.eval()
    perdida_total = 0.0
    todas_probs   = []
    todas_etiq    = []

    with torch.no_grad():
        for psd, etiquetas in loader:
            psd, etiquetas = psd.to(device), etiquetas.to(device)
            logits = modelo(psd)
            perdida_total += criterio(logits, etiquetas).item()
            probs = torch.sigmoid(logits).cpu().numpy()
            todas_probs.append(probs)
            todas_etiq.append(etiquetas.cpu().numpy())

    todas_probs = np.vstack(todas_probs)   # (N, 9)
    todas_etiq  = np.vstack(todas_etiq)    # (N, 9)

    # AUC-ROC por canal — solo posiciones con etiqueta válida (>= 0) y ambas clases
    aucs = []
    for canal in range(todas_etiq.shape[1]):
        etiq_c = todas_etiq[:, canal]
        prob_c = todas_probs[:, canal]
        mask   = etiq_c >= 0                     # ignorar etiquetas -1
        etiq_c = etiq_c[mask]
        prob_c = prob_c[mask]
        if len(etiq_c) > 0 and len(np.unique(etiq_c)) == 2:
            aucs.append(roc_auc_score(etiq_c, prob_c))

    # F1 con beta=2 — solo sobre posiciones con etiqueta válida (>= 0)
    mascara      = todas_etiq.flatten() >= 0
    etiq_validas = todas_etiq.flatten()[mascara].astype(int)
    pred_validas = (todas_probs.flatten()[mascara] >= UMBRAL_LIBRE).astype(int)
    if len(np.unique(etiq_validas)) > 1:
        f1_beta2 = fbeta_score(etiq_validas, pred_validas,
                               beta=2, average="binary", zero_division=0)
    else:
        f1_beta2 = 0.0   # solo una clase presente en validación

    return {
        "perdida":   perdida_total / len(loader),
        "auc_media": float(np.mean(aucs)) if aucs else 0.0,
        "f1_beta2":  float(f1_beta2)
    }


def entrenar(dataset_dir: str, salida_dir: str,
             epocas: int = 80, batch_size: int = 64,
             lr: float = 1e-3, dispositivo: str = "auto") -> SpectralSenseCNN:
    """
    Bucle de entrenamiento completo con early stopping y guardado del mejor modelo.

    Args:
        dataset_dir: directorio raíz con subcarpetas train/ y val/
        salida_dir:  donde guardar checkpoints y el mejor modelo
        epocas:      máximo de épocas
        batch_size:  tamaño del mini-batch
        lr:          learning rate inicial
        dispositivo: "auto", "cuda", "cpu"

    Returns:
        modelo entrenado (con los mejores pesos cargados)
    """
    os.makedirs(salida_dir, exist_ok=True)

    # Dispositivo
    if dispositivo == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dispositivo)
    print(f"Entrenando en: {device}")

    # Datasets y loaders
    ds_train = TVWSDataset(dataset_dir, split="train", augmentar=True)
    ds_val   = TVWSDataset(dataset_dir, split="val",   augmentar=False)
    usar_pin_memory = device.type != "cpu"
    # num_workers=0 en Windows: evita problemas con multiprocessing en spawn
    n_workers    = 0 if os.name == "nt" else 2
    loader_train = DataLoader(ds_train, batch_size=batch_size,
                               shuffle=True,  num_workers=n_workers,
                               pin_memory=usar_pin_memory)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size,
                               shuffle=False, num_workers=n_workers,
                               pin_memory=usar_pin_memory)
    print(f"Train: {len(ds_train)} muestras | Val: {len(ds_val)} muestras")

    # Modelo
    modelo = SpectralSenseCNN(n_canales_salida=N_CANALES_MAX).to(device)
    print(f"Parámetros entrenables: {modelo.contar_parametros():,}")

    # Función de pérdida con peso de clase
    pos_weight = calcular_pos_weight(dataset_dir).to(device)
    criterio   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizador y scheduler
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizador, T_max=epocas, eta_min=1e-5
    )

    # Early stopping
    mejor_auc        = 0.0
    paciencia        = 10
    epocas_sin_mejora = 0
    ruta_mejor       = os.path.join(salida_dir, "mejor_modelo.pt")

    historial = []

    for epoca in range(1, epocas + 1):
        # ── Fase de entrenamiento ──
        modelo.train()
        perdida_train = 0.0
        t0 = time.time()

        for psd, etiquetas in loader_train:
            psd, etiquetas = psd.to(device), etiquetas.to(device)
            optimizador.zero_grad()
            logits = modelo(psd)
            perdida = criterio(logits, etiquetas)
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            optimizador.step()
            perdida_train += perdida.item()

        perdida_train /= len(loader_train)
        scheduler.step()

        # ── Fase de validación ──
        metricas_val = evaluar(modelo, loader_val, criterio, device)
        duracion     = time.time() - t0

        print(f"Época {epoca:3d}/{epocas} | "
              f"Train loss: {perdida_train:.4f} | "
              f"Val loss: {metricas_val['perdida']:.4f} | "
              f"AUC: {metricas_val['auc_media']:.4f} | "
              f"F1(β=2): {metricas_val['f1_beta2']:.4f} | "
              f"{duracion:.1f}s")

        historial.append({
            "epoca": epoca,
            "perdida_train": perdida_train,
            **metricas_val
        })

        # Guardar mejor modelo
        if metricas_val["auc_media"] > mejor_auc:
            mejor_auc = metricas_val["auc_media"]
            epocas_sin_mejora = 0
            torch.save({
                "epoca":      epoca,
                "model_state": modelo.state_dict(),
                "optim_state": optimizador.state_dict(),
                "auc":         mejor_auc,
                "metricas":   metricas_val
            }, ruta_mejor)
            print(f"  ✓ Mejor modelo guardado (AUC={mejor_auc:.4f})")
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= paciencia:
                print(f"Early stopping en época {epoca} (sin mejora en {paciencia} épocas)")
                break

    # Cargar mejores pesos si se guardaron, si no usar los actuales
    if os.path.exists(ruta_mejor):
        checkpoint = torch.load(ruta_mejor, map_location=device,
                                weights_only=True)
        modelo.load_state_dict(checkpoint["model_state"])
        print(f"\nEntrenamiento completado. Mejor AUC: {mejor_auc:.4f}")
    else:
        print(f"\nEntrenamiento completado (sin checkpoint guardado — AUC no mejoró).")
        print("Consejo: aumentar --n_muestras del dataset para obtener métricas válidas.")

    # Guardar historial
    with open(os.path.join(salida_dir, "historial.json"), "w") as f:
        json.dump(historial, f, indent=2)

    return modelo


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTACIÓN A ONNX
# ─────────────────────────────────────────────────────────────────────────────

def exportar_onnx(modelo: SpectralSenseCNN,
                  ruta_salida: str = "spectral_sense.onnx") -> str:
    """
    Exporta el modelo PyTorch a formato ONNX para inferencia con ONNX Runtime.

    Args:
        modelo:      SpectralSenseCNN con pesos entrenados cargados
        ruta_salida: ruta del archivo .onnx a generar

    Returns:
        ruta_salida (para encadenamiento)
    """
    modelo.eval()
    modelo.cpu()

    dummy_input = torch.randn(1, PSD_LENGTH)

    torch.onnx.export(
        modelo,
        dummy_input,
        ruta_salida,
        export_params=True,
        input_names=["psd_input"],
        output_names=["logits"],
        dynamic_axes={
            "psd_input": {0: "batch_size"},
            "logits":    {0: "batch_size"}
        },
        opset_version=17,
        do_constant_folding=True    # optimización: fusionar operaciones constantes
    )

    tamanio_kb = os.path.getsize(ruta_salida) / 1024
    print(f"Modelo exportado: {ruta_salida} ({tamanio_kb:.1f} KB)")

    # Verificación rápida con ONNX Runtime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(ruta_salida,
                                       providers=["CPUExecutionProvider"])
        inp  = np.random.randn(1, PSD_LENGTH).astype(np.float32)
        out  = session.run(["logits"], {"psd_input": inp})[0]
        print(f"Verificación ONNX OK — output shape: {out.shape}")
    except ImportError:
        print("onnxruntime no instalado — omitir verificación")

    return ruta_salida


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCIA EN CAMPO — wrapper de ONNX Runtime
# ─────────────────────────────────────────────────────────────────────────────

class SpectralSenseInferencia:
    """
    Wrapper de ONNX Runtime para inferencia en campo en el Gateway.

    Uso:
        inferenciador = SpectralSenseInferencia("spectral_sense.onnx")
        resultado = inferenciador.clasificar_subbanda(iq_array, posicion_idx=0)
        print(resultado["libres_globales"])   # ej: [14, 17, 21]
    """

    def __init__(self, ruta_onnx: str, umbral: float = UMBRAL_LIBRE):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Instalar onnxruntime: pip install onnxruntime")

        # Intentar GPU (DirectML en Windows/Linux con Radeon 780M)
        proveedores_disponibles = ort.get_available_providers()
        if "DmlExecutionProvider" in proveedores_disponibles:
            proveedores = ["DmlExecutionProvider", "CPUExecutionProvider"]
            print("ONNX Runtime: usando DirectML (Radeon 780M)")
        else:
            proveedores = ["CPUExecutionProvider"]
            print("ONNX Runtime: usando CPU")

        self.session = ort.InferenceSession(ruta_onnx, providers=proveedores)
        self.umbral  = umbral

    def clasificar_subbanda(self, iq: np.ndarray, posicion_idx: int) -> dict:
        """
        Clasifica los canales de una sub-banda de 56 MHz.

        Args:
            iq:           muestras I/Q crudas — array complejo (N,) o real (N, 2)
            posicion_idx: índice de posición de barrido (0–4), determina el mapeo
                          de índices locales a canales globales del plan peruano

        Returns:
            dict con:
                prob_por_canal (dict canal→prob),
                libres_globales (list),
                ocupados_globales (list),
                mejor_canal (int o None),
                latencia_ms (float)
        """
        t0 = time.perf_counter()

        # 1. Preprocesamiento
        psd = iq_a_psd_normalizado(iq, n_fft=PSD_LENGTH)   # (512,) float32

        # 2. Inferencia ONNX
        inp    = psd[np.newaxis, :].astype(np.float32)      # (1, 512)
        logits = self.session.run(["logits"], {"psd_input": inp})[0]  # (1, 9)
        probs  = 1.0 / (1.0 + np.exp(-logits[0]))           # sigmoid → (9,)

        # 3. Mapeo a canales globales
        canales = POSICIONES_BARRIDO[posicion_idx]["canales_globales"]
        n_vis   = len(canales)   # canales visibles en esta posición (7 o 9)

        prob_por_canal    = {canales[i]: float(probs[i]) for i in range(n_vis)}
        libres_globales   = [c for c, p in prob_por_canal.items() if p < self.umbral]
        ocupados_globales = [c for c, p in prob_por_canal.items() if p >= self.umbral]

        # Ordenar libres por probabilidad de ocupación ascendente
        libres_globales.sort(key=lambda c: prob_por_canal[c])
        mejor_canal = libres_globales[0] if libres_globales else None

        latencia_ms = (time.perf_counter() - t0) * 1000

        return {
            "prob_por_canal":   prob_por_canal,
            "libres_globales":  libres_globales,
            "ocupados_globales": ocupados_globales,
            "mejor_canal":      mejor_canal,
            "latencia_ms":      round(latencia_ms, 2)
        }

    def ciclo_completo(self, capturas_iq: list[np.ndarray]) -> dict:
        """
        Ejecuta la clasificación sobre las 5 sub-bandas del barrido completo.

        Args:
            capturas_iq: lista de 5 arrays I/Q, uno por posición de barrido

        Returns:
            mapa espectral global con timestamp, canales y mejor candidato
        """
        if len(capturas_iq) != 5:
            raise ValueError(f"Se esperan 5 capturas, se recibieron {len(capturas_iq)}")

        mapa = {}
        for posicion_idx, iq in enumerate(capturas_iq):
            resultado = self.clasificar_subbanda(iq, posicion_idx)
            for canal, prob in resultado["prob_por_canal"].items():
                mapa[canal] = {
                    "prob_ocupado": prob,
                    "libre":        prob < self.umbral
                }

        libres_ordenados = sorted(
            [c for c, v in mapa.items() if v["libre"]],
            key=lambda c: mapa[c]["prob_ocupado"]
        )

        return {
            "timestamp":        time.time(),
            "canales":          mapa,
            "libres_ordenados": libres_ordenados,
            "mejor_canal":      libres_ordenados[0] if libres_ordenados else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SpectralSense — IA para sensado espectral TVWS"
    )
    parser.add_argument("--modo", choices=["entrenar", "exportar", "test"],
                        default="test", help="Modo de operación")
    parser.add_argument("--dataset", default="./dataset",
                        help="Directorio raíz del dataset (con train/ y val/)")
    parser.add_argument("--salida", default="./modelos",
                        help="Directorio para guardar modelos y checkpoints")
    parser.add_argument("--checkpoint", default="./modelos/mejor_modelo.pt",
                        help="Ruta del checkpoint .pt para exportar a ONNX")
    parser.add_argument("--epocas", type=int, default=80)
    parser.add_argument("--batch",  type=int, default=64)
    parser.add_argument("--lr",     type=float, default=1e-3)
    args = parser.parse_args()

    if args.modo == "entrenar":
        print("═" * 60)
        print("  SpectralSense CNN — Entrenamiento")
        print("═" * 60)
        modelo = entrenar(
            dataset_dir=args.dataset,
            salida_dir=args.salida,
            epocas=args.epocas,
            batch_size=args.batch,
            lr=args.lr
        )
        # Exportar automáticamente tras entrenamiento
        ruta_onnx = os.path.join(args.salida, "spectral_sense.onnx")
        exportar_onnx(modelo, ruta_onnx)

    elif args.modo == "exportar":
        print(f"Cargando checkpoint: {args.checkpoint}")
        modelo = SpectralSenseCNN()
        ckpt   = torch.load(args.checkpoint, map_location="cpu")
        modelo.load_state_dict(ckpt["model_state"])
        ruta_onnx = args.checkpoint.replace(".pt", ".onnx")
        exportar_onnx(modelo, ruta_onnx)

    elif args.modo == "test":
        print("═" * 60)
        print("  SpectralSense CNN — Test de inferencia con datos sintéticos")
        print("═" * 60)
        modelo = SpectralSenseCNN()
        print(f"Parámetros: {modelo.contar_parametros():,}")

        # Test con PSD sintético
        psd_fake    = torch.randn(4, PSD_LENGTH).clamp(0, 1)
        logits_fake = modelo(psd_fake)
        probs_fake  = torch.sigmoid(logits_fake)
        print(f"Input shape:  {psd_fake.shape}")
        print(f"Output shape: {logits_fake.shape}")
        print(f"Probs (batch 0): {probs_fake[0].detach().numpy().round(3)}")

        # Exportar modelo de prueba
        os.makedirs("./modelos_test", exist_ok=True)
        ruta = exportar_onnx(modelo, "./modelos_test/spectral_sense_test.onnx")

        # Test de inferencia ONNX
        try:
            inferenciador = SpectralSenseInferencia(ruta)
            iq_fake = (np.random.randn(61000) + 1j * np.random.randn(61000)).astype(np.complex64)
            resultado = inferenciador.clasificar_subbanda(iq_fake, posicion_idx=0)
            print(f"\nTest inferencia sub-banda:")
            print(f"  Canales libres:   {resultado['libres_globales']}")
            print(f"  Canales ocupados: {resultado['ocupados_globales']}")
            print(f"  Mejor canal:      {resultado['mejor_canal']}")
            print(f"  Latencia:         {resultado['latencia_ms']} ms")
        except ImportError:
            print("onnxruntime no disponible — instalar con: pip install onnxruntime")

        print("\nTest completado sin errores.")


if __name__ == "__main__":
    main()