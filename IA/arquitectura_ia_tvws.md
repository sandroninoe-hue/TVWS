# Arquitectura del modelo de IA — Radio Cognitiva TVWS
> Documento técnico de referencia para el módulo de inteligencia artificial del proyecto VRI 2024–2026, UNI.
> Última actualización: junio 2026.

---

## 1. Contexto y objetivo del modelo

El modelo de IA tiene una única función operativa: **dado el vector PSD de una sub-banda de 56 MHz capturada por el canal RX2 del bladeRF, determinar qué canales TVWS de 6 MHz dentro de esa sub-banda están libres de usuarios primarios**.

El Gateway ejecuta este modelo en un bucle continuo, barriendo la banda completa 470–698 MHz en 5 posiciones de sintonización. Al término de cada ciclo completo (~137 ms), el árbitro de decisión dispone de un mapa de ocupación actualizado para los 38 canales del plan peruano de atribución TVWS UHF.

### Restricciones de diseño que gobiernan la arquitectura

| Restricción | Valor | Implicación en el diseño |
|---|---|---|
| Latencia máxima de inferencia | < 5 ms por sub-banda | Modelo pequeño, sin atención ni recurrencia |
| Tamaño máximo del modelo | < 4 MB (ONNX FP32) | Máximo ~1M parámetros |
| Hardware de inferencia | Ryzen 9 8945HS + Radeon 780M (ONNX DirectML) | Sin dependencias de CUDA |
| Input | PSD de 512 puntos float32 | Conv1D, no Conv2D |
| Output | Probabilidad de ocupación por canal visible | Cabeza multi-etiqueta, no softmax |
| Domain mismatch | Dataset RTL-SDR 8 bits → inferencia bladeRF 12 bits | Normalización por percentil obligatoria |

---

## 2. Tipo de modelo seleccionado

**Red Neuronal Convolucional 1D (1D-CNN) con cabeza multi-etiqueta.**

### Por qué 1D-CNN y no otras alternativas

**vs. MLP puro:** Un MLP sobre el vector PSD completo no tiene invarianza traslacional. Si el mismo patrón espectral de un canal ISDB-Tb aparece en distintas posiciones del vector (distintas frecuencias), el MLP lo trata como entradas completamente distintas. La CNN lo reconoce en cualquier posición.

**vs. CNN 2D sobre espectrograma:** Requeriría acumular múltiples snapshots temporales para construir el espectrograma, añadiendo latencia. El PSD instantáneo de una sub-banda es suficiente para detectar ocupación en TVWS, donde los primarios son señales continuas (televisión), no transitorias.

**vs. LSTM / GRU:** Las redes recurrentes incorporan memoria temporal, útil si el estado de un canal cambia en escala de milisegundos. Los usuarios primarios de TVWS (emisoras de TV) tienen ciclos de actividad en escala de horas. La memoria temporal añade complejidad sin beneficio real para este problema.

**vs. Transformer:** Excesivo en parámetros y latencia para este input de 512 puntos. El mecanismo de atención no aporta ventaja sobre la convolución para señales espectrales 1D de longitud fija.

---

## 3. Descripción del input

### 3.1 Origen de los datos

Cada inferencia recibe el PSD de **una sola captura de sub-banda**:

- **Hardware:** canal RX2 del bladeRF 2.0 micro xA4
- **Ancho de banda:** 56 MHz (límite físico del transceptor AD9361)
- **Muestras capturadas:** ~1.12M muestras I/Q a 61 MSPS → ~18.4 ms de señal por captura
- **FFT:** 512 puntos con ventana Hann, método Welch (promedio de segmentos solapados)
- **Resultado:** vector PSD de 512 puntos en dB, representando 56 MHz de espectro

### 3.2 Resolución espectral

```
Resolución por bin = 56 MHz / 512 puntos ≈ 109 kHz por bin
Bins por canal de 6 MHz = 6000 kHz / 109 kHz ≈ 55 bins
```

Cada canal TVWS ocupa aproximadamente 55 puntos del vector PSD. Esto es clave para elegir los tamaños de kernel (ver sección 4).

### 3.3 Normalización obligatoria (domain mismatch)

El dataset de entrenamiento se captura con RTL-SDR Blog V4 (ADC de 8 bits, ~48 dB de rango dinámico). La inferencia en campo corre sobre señales del bladeRF (ADC de 12 bits, ~72 dB de rango dinámico). Sin normalización, los vectores PSD de entrenamiento e inferencia tendrían distribuciones incompatibles.

**Protocolo de normalización (idéntico en entrenamiento e inferencia):**

```python
def normalizar_psd(psd_raw: np.ndarray) -> np.ndarray:
    """
    Normaliza el vector PSD al rango [0, 1] usando percentiles robustos.
    Mitiga el domain mismatch RTL-SDR (8 bits) → bladeRF (12 bits).
    Debe aplicarse ANTES de alimentar la CNN, tanto en entrenamiento como en inferencia.
    """
    psd_min = np.percentile(psd_raw, 5)    # piso de ruido robusto
    psd_max = np.percentile(psd_raw, 95)   # techo robusto (excluye picos aislados)
    psd_norm = (psd_raw - psd_min) / (psd_max - psd_min + 1e-9)
    return np.clip(psd_norm, 0.0, 1.0)
```

**Por qué percentiles y no min/max absoluto:** El RTL-SDR tiene picos esporádicos de ruido que inflan el máximo absoluto, comprimiendo toda la señal útil hacia abajo. El percentil 95 descarta esos picos y produce una distribución comparable entre los dos ADC.

---

## 4. Arquitectura de la red

### 4.1 Visión general

```
Input (512,)
    │
    ▼
┌─────────────────────────────────┐
│  Bloque 1 — estructura gruesa   │  Conv1D(32, k=45) → BN → ReLU → MaxPool(2)
│  campo receptivo: ~5 MHz        │  salida: 256 × 32
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Bloque 2 — detalle intra-canal │  Conv1D(64, k=9) × 2 → BN → ReLU → MaxPool(2)
│  campo receptivo: ~1 MHz        │  salida: 128 × 64
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Bloque 3 — contexto inter-canal│  Conv1D(128, k=5) → BN → ReLU
│  campo receptivo: ~545 kHz      │  salida: 128 × 128
└─────────────────────────────────┘
    │
    ▼
GlobalAveragePool1D → (128,)
    │
    ▼
Dense(64) → ReLU → Dropout(0.3)
    │
    ▼
Dense(9) → Sigmoid
    │
    ▼
Output: 9 probabilidades P(ocupado) — una por canal TVWS visible en la sub-banda
```

### 4.2 Bloque 1 — estructura espectral gruesa

**Capa:** `Conv1D(in=1, out=32, kernel_size=45, padding=22)`

**Razonamiento del kernel:**
- 45 bins × 109 kHz/bin ≈ **4.9 MHz** de campo receptivo
- Un canal TVWS de 6 MHz ocupa ~55 bins; un kernel de 45 bins cubre el 82% del canal
- A esta escala, los filtros aprenden a reconocer la "silueta" de una señal de TV: el lóbulo principal plano y los flancos de rolloff característicos de ISDB-Tb
- Kernel impar → `padding = kernel//2 = 22` → salida del mismo largo que la entrada (`same`)

**Salida tras MaxPool(stride=2):** 256 × 32

El MaxPool reduce la dimensión espectral a la mitad. A 256 puntos, cada punto representa ~218 kHz, resolución suficiente para los bloques siguientes.

### 4.3 Bloque 2 — detalle intra-canal

**Capas:** `Conv1D(64, k=9) → BN → ReLU → Conv1D(64, k=9) → BN → ReLU → MaxPool(2)`

**Razonamiento del kernel:**
- 9 bins × 218 kHz/bin (post-pool del bloque 1) ≈ **~1 MHz** de campo receptivo
- A esta escala, los filtros aprenden la textura interna de un canal: pendiente de los flancos espectrales, presencia de pilotos OFDM, nivel del suelo de ruido entre subportadoras
- Dos capas encadenadas con kernel 9 equivalen en campo receptivo a una capa con kernel 17, pero con menor número de parámetros y con no-linealidad intermedia

**Salida tras MaxPool(stride=2):** 128 × 64

### 4.4 Bloque 3 — contexto inter-canal

**Capa:** `Conv1D(128, k=5) → BN → ReLU`

**Razonamiento del kernel:**
- 5 bins sobre el feature map de 128 posiciones cubre ~5 posiciones × (56 MHz / 128) ≈ **2.2 MHz**
- A esta escala, el filtro puede ver la relación entre canales adyacentes: un primario con fuga espectral ocupa parcialmente los bordes de canales vecinos, patrón que este bloque detecta como correlación espacial en el feature map
- No hay MaxPool al final de este bloque: se conservan los 128 puntos para el GlobalAveragePool

**Salida:** 128 × 128

### 4.5 GlobalAveragePool1D

```
128 × 128  →  (128,)
```

**Por qué GlobalAveragePool y no Flatten:**

Flatten sobre 128 × 128 produciría un vector de 16,384 dimensiones. La primera capa densa tendría 16,384 × 64 = ~1M parámetros solo en esa conexión, violando el presupuesto de tamaño del modelo.

GlobalAveragePool promedia cada mapa de características a lo largo de la dimensión espectral, produciendo un vector de 128 valores. Cada valor es el "nivel medio de activación" de ese filtro sobre toda la sub-banda. Además actúa como regularizador implícito: el modelo no puede memorizar posiciones espectrales específicas, solo patrones globales.

### 4.6 Cabeza clasificadora

```python
Dense(128 → 64) → ReLU → Dropout(0.3) → Dense(64 → 9)
```

**Dense(64):** Comprime las 128 activaciones del GAP a 64, forzando al modelo a construir una representación compacta de la calidad espectral de la sub-banda.

**Dropout(0.3):** Regularización estándar. Previene que el clasificador memorice condiciones de captura específicas del dataset de entrenamiento (hora del día, ubicación geográfica, equipo usado).

**Dense(9):** Produce 9 logits, uno por cada canal de 6 MHz visible en la zona plana de la sub-banda. La capa no tiene activación propia porque se usa `BCEWithLogitsLoss` durante el entrenamiento (numéricamente más estable que Sigmoid + BCELoss).

**Durante inferencia:** se aplica Sigmoid explícito a los 9 logits para obtener probabilidades `[0, 1]`.

### 4.7 Resumen de parámetros

| Componente | Parámetros |
|---|---|
| Conv1D bloque 1 (1×32, k=45) | 1,472 |
| BatchNorm bloque 1 | 64 |
| Conv1D bloque 2 (32×64, k=9) × 2 | 36,928 |
| BatchNorm bloque 2 × 2 | 256 |
| Conv1D bloque 3 (64×128, k=5) | 41,088 |
| BatchNorm bloque 3 | 256 |
| Dense(128→64) | 8,256 |
| Dense(64→9) | 585 |
| **Total** | **~88,900 parámetros** |
| Tamaño ONNX (FP32) | ~355 KB |
| Tamaño ONNX (INT8 cuantizado) | ~95 KB |
| Latencia estimada (Ryzen 9 8945HS, CPU) | < 1 ms |
| Latencia estimada (Radeon 780M, ONNX DirectML) | < 0.5 ms |

> **Nota:** El modelo es significativamente más pequeño que los ~290K parámetros estimados inicialmente, gracias al GlobalAveragePool que elimina la mayor parte de los parámetros de la cabeza. Esto es una ventaja: más margen para el barrido de hardware dentro del presupuesto de 137 ms por ciclo.

---

## 5. Implementación en PyTorch

```python
import torch
import torch.nn as nn

class SpectralSenseCNN(nn.Module):
    """
    1D-CNN para clasificación de ocupación espectral en sub-bandas TVWS de 56 MHz.

    Input:  (batch, 512)   — vector PSD normalizado [0, 1]
    Output: (batch, 9)     — logits de ocupación por canal (sin Sigmoid)
                             aplicar Sigmoid para obtener probabilidades en inferencia
    """

    def __init__(self, n_canales_salida: int = 9, dropout: float = 0.3):
        super().__init__()

        # Bloque 1: silueta espectral del canal (~5 MHz de campo receptivo)
        self.bloque1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=45, padding=22),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)   # 512 → 256
        )

        # Bloque 2: textura intra-canal (~1 MHz de campo receptivo)
        self.bloque2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)   # 256 → 128
        )

        # Bloque 3: correlaciones inter-canal (~2 MHz de campo receptivo)
        self.bloque3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
            # Sin MaxPool: conservar resolución para GlobalAveragePool
        )

        # Pooling global: colapsa 128 × 128 → (128,)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Cabeza clasificadora multi-etiqueta
        self.cabeza = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_canales_salida)
            # Sin Sigmoid: BCEWithLogitsLoss lo incorpora durante entrenamiento
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 512) → añadir dimensión de canal para Conv1d
        x = x.unsqueeze(1)          # (batch, 1, 512)
        x = self.bloque1(x)         # (batch, 32, 256)
        x = self.bloque2(x)         # (batch, 64, 128)
        x = self.bloque3(x)         # (batch, 128, 128)
        x = self.gap(x).squeeze(-1) # (batch, 128)
        return self.cabeza(x)       # (batch, 9) — logits

    def predecir(self, x: torch.Tensor, umbral: float = 0.20) -> dict:
        """
        Inferencia: devuelve probabilidades y decisión de canal libre/ocupado.

        Args:
            x:       tensor (1, 512) o (batch, 512), PSD normalizado
            umbral:  prob_ocupado < umbral → canal libre (default: 0.20, conservador)

        Returns:
            dict con 'prob_ocupado' (array), 'libres' (índices locales), 'ocupados' (índices locales)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.sigmoid(logits).squeeze().numpy()

        return {
            "prob_ocupado": probs,
            "libres":       [i for i, p in enumerate(probs) if p < umbral],
            "ocupados":     [i for i, p in enumerate(probs) if p >= umbral]
        }
```

---

## 6. Función de pérdida y entrenamiento

### 6.1 Función de pérdida

```python
# BCEWithLogitsLoss con pesos por clase para manejar desbalance
# En TVWS rural: la mayoría de los canales están libres la mayor parte del tiempo
# pos_weight compensa: si 80% libre / 20% ocupado → pos_weight ≈ 4.0

# Calcular pos_weight desde el dataset:
n_muestras_ocupado = etiquetas.sum(0)           # suma por columna (por canal)
n_muestras_libre   = len(dataset) - n_muestras_ocupado
pos_weight         = n_muestras_libre / (n_muestras_ocupado + 1e-6)  # shape (9,)

criterio = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
```

### 6.2 Optimizador y scheduler

```python
optimizador = torch.optim.AdamW(
    modelo.parameters(),
    lr=1e-3,
    weight_decay=1e-4      # regularización L2 implícita
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizador,
    T_max=50,              # 50 épocas de decaimiento
    eta_min=1e-5
)
```

### 6.3 Configuración de entrenamiento

| Parámetro | Valor |
|---|---|
| Hardware | RTX 4060 local o Google Colab Pro (T4) |
| Épocas | 50–100 (con early stopping, paciencia=10) |
| Batch size | 64 |
| Tiempo estimado por experimento | 3–8 minutos (T4 GPU) |
| Framework | PyTorch 2.x |
| Exportación | `torch.onnx.export()` → ONNX opset 17 |

---

## 7. Exportación a ONNX e inferencia en campo

### 7.1 Exportación

```python
def exportar_onnx(modelo: SpectralSenseCNN, ruta: str = "spectral_sense.onnx"):
    modelo.eval()
    dummy_input = torch.randn(1, 512)   # batch=1, PSD de 512 puntos

    torch.onnx.export(
        modelo,
        dummy_input,
        ruta,
        input_names=["psd_input"],
        output_names=["logits"],
        dynamic_axes={
            "psd_input": {0: "batch_size"},   # batch variable en inferencia
            "logits":    {0: "batch_size"}
        },
        opset_version=17
    )
    print(f"Modelo exportado: {ruta}")
    print(f"Tamaño: {os.path.getsize(ruta) / 1024:.1f} KB")
```

### 7.2 Inferencia con ONNX Runtime en el Gateway

```python
import onnxruntime as ort
import numpy as np

# Cargar modelo una sola vez al inicio del proceso
session = ort.InferenceSession(
    "spectral_sense.onnx",
    providers=["DmlExecutionProvider",    # Radeon 780M via DirectML (preferido)
               "CPUExecutionProvider"]    # fallback a CPU
)

def inferir_subbanda(psd_normalizado: np.ndarray, umbral: float = 0.20) -> dict:
    """
    Inferencia de una sub-banda de 56 MHz.

    Args:
        psd_normalizado: array (512,) float32, ya normalizado con normalizar_psd()
        umbral:          prob_ocupado < umbral → canal libre

    Returns:
        dict con probabilidades y canales libres (índices locales 0–8)
    """
    inp    = psd_normalizado[np.newaxis, :].astype(np.float32)   # (1, 512)
    logits = session.run(["logits"], {"psd_input": inp})[0]      # (1, 9)
    probs  = 1.0 / (1.0 + np.exp(-logits[0]))                   # sigmoid → (9,)

    return {
        "prob_ocupado": probs,
        "libres":       [i for i, p in enumerate(probs) if p < umbral],
        "ocupados":     [i for i, p in enumerate(probs) if p >= umbral]
    }
```

---

## 8. Mapeo de índices locales a canales globales

El modelo produce índices locales (0 a 8, relativos a la sub-banda). El árbitro necesita números de canal globales del plan peruano (14 a 52). La tabla de mapeo depende de la posición de sintonización activa:

```python
# Canales TVWS peruanos: 470 MHz (canal 14) a 698 MHz (canal 52), paso 6 MHz
# Centro del canal N: 470 + (N - 14) * 6 + 3 MHz

POSICIONES_BARRIDO = [
    {"centro_mhz": 498, "canales_globales": list(range(14, 23))},  # 9 canales: 14–22
    {"centro_mhz": 536, "canales_globales": list(range(22, 31))},  # 9 canales: 22–30
    {"centro_mhz": 578, "canales_globales": list(range(30, 39))},  # 9 canales: 30–38
    {"centro_mhz": 626, "canales_globales": list(range(38, 47))},  # 9 canales: 38–46
    {"centro_mhz": 670, "canales_globales": list(range(46, 53))},  # 7 canales: 46–52
]

def mapear_a_globales(resultado_local: dict, posicion_idx: int) -> dict:
    """Convierte índices locales del modelo a números de canal globales."""
    canales = POSICIONES_BARRIDO[posicion_idx]["canales_globales"]
    return {
        "libres_globales":  [canales[i] for i in resultado_local["libres"]
                             if i < len(canales)],
        "ocupados_globales":[canales[i] for i in resultado_local["ocupados"]
                             if i < len(canales)],
        "prob_por_canal":   {canales[i]: float(resultado_local["prob_ocupado"][i])
                             for i in range(min(9, len(canales)))}
    }
```

---

## 9. Umbral de decisión y su calibración

El umbral por defecto en el árbitro es **`UMBRAL_LIBRE = 0.20`**, no 0.5.

### Por qué 0.20 y no 0.5

En radio cognitiva, los costos de error son asimétricos:

- **Falsa alarma** (declara ocupado un canal libre): el sistema usa un canal alternativo. Costo: menor eficiencia espectral.
- **Detección fallida** (declara libre un canal ocupado): el secundario interfiere al primario (emisora de TV). Costo: **violación regulatoria grave**.

Un umbral de 0.20 significa que el canal se declara libre solo si el modelo está muy seguro de que no hay primario. Esto sesga el sistema hacia la protección del primario, que es exactamente lo que exige el marco regulatorio de TVWS.

### Calibración del umbral en Fase 1

Durante la captura del dataset, registrar para cada muestra la etiqueta real y la probabilidad predicha por el modelo. Trazar la curva ROC y seleccionar el umbral que minimiza:

```
Costo = C_FA × FPR + C_MD × FNR
```

Donde `C_MD >> C_FA` (el costo de una detección fallida es mucho mayor que el de una falsa alarma). En la práctica, un umbral entre 0.15 y 0.25 es el rango razonable para TVWS.

---

## 10. Output del modelo por ciclo completo

Al término de las 5 sub-bandas, el árbitro consolida los resultados en el mapa espectral global:

```python
{
    "timestamp": 1748123456.789,

    "canales": {
        14: {"prob_ocupado": 0.03, "libre": True},
        15: {"prob_ocupado": 0.91, "libre": False},
        # ... canales 14 a 52
        52: {"prob_ocupado": 0.11, "libre": True},
    },

    "libres_ordenados": [14, 52, 31, 17],   # ordenados de más limpio a menos
    "mejor_canal":      14,                  # candidato para siguiente TX
    "duracion_ciclo_ms": 137.4               # para monitoreo de performance
}
```

Este dict es consumido exclusivamente por el árbitro de decisión (ver documento `arbitro_decision.md`), que aplica las reglas R1–R7 antes de emitir cualquier orden de salto de canal por el enlace LoRa.

---

## 11. Métricas de evaluación del modelo

| Métrica | Descripción | Relevancia para TVWS |
|---|---|---|
| **Tasa de detección (TPR)** | % de canales ocupados correctamente detectados | Crítica — protección al primario |
| **Tasa de falsa alarma (FPR)** | % de canales libres declarados erróneamente ocupados | Importante — eficiencia espectral |
| **AUC-ROC** | Área bajo la curva ROC, agnostica al umbral | Métrica global de discriminación |
| **F1-score (β=2)** | Penaliza más los falsos negativos que los falsos positivos | Alineada con el costo asimétrico de TVWS |
| **Latencia de inferencia** | Tiempo de `session.run()` en ONNX Runtime | Restricción operacional dura |

La meta del proyecto es **TPR ≥ 0.95** con **FPR ≤ 0.15** sobre el dataset de validación capturado en Lima (Fase 1).

---

*Documento generado a partir de las decisiones de diseño consolidadas en el contexto maestro del proyecto. Actualizar al cierre de Fase 1 con los resultados de entrenamiento y validación del modelo.*
