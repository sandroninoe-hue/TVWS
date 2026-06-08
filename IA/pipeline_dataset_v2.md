# Pipeline de construcción del dataset de entrenamiento — v2
> Documento técnico de referencia — Radio Cognitiva TVWS, UNI 2024–2026.
> Fase 1: 18/05/2026 → 15/06/2026. Responsable: Franco Rafael Espinoza.
> **Revisión v2:** hardware actualizado con equipos reales confirmados del equipo.

---

## 1. Hardware real confirmado para Fase 1

### 1.1 Equipos disponibles en Fase 1

| Equipo | Rol | Estado |
|---|---|---|
| RTL-SDR Blog V4 (×1) | Captura de muestras I/Q espectrales | Disponible desde día 1 |
| Antena discone 25–1300 MHz | Recepción espectral omnidireccional | Disponible desde día 1 |
| PC escritorio Ryzen 7 7700x + RX 580 2048SP 8GB | Captura, preprocesamiento y **entrenamiento (CPU)** | Equipo personal del equipo, no RNP |
| Google Colab Pro (T4 GPU) | Entrenamiento CNN si GPU local no es viable | ~$10 USD/mes |
| bladeRF 2.0 xA4 | SDR Gateway (Fase 2 en adelante) | Pendiente de aduana, 4–8 semanas |
| LimeSDR Mini 2.0 | SDR Cliente (Fase 2 en adelante) | Pendiente de aduana, 4–8 semanas |
| Orange Pi 5 8GB | Nodo cliente (reemplaza Mini PC N100) | Confirmar modelo exacto |
| Mini PC Ryzen 9 8945HS | Nodo Gateway | Pendiente de compra RNP |

### 1.2 Restricción crítica: RX 580 2048SP y PyTorch

La RX 580 2048SP es una GPU basada en el chip **Polaris 20 recortado (gfx803)**. Su soporte en el stack ROCm/PyTorch tiene limitaciones importantes:

| Versión PyTorch | Soporte ROCm para gfx803 | Estado |
|---|---|---|
| PyTorch ≤ 1.13 | ROCm 4.x | Funcional pero obsoleto |
| PyTorch 2.x | ROCm 5.4+ | **gfx803 no está en la lista oficial de targets** |
| PyTorch 2.x | ROCm 6.x | gfx803 eliminado de la lista soportada |

**Conclusión:** No se puede garantizar que PyTorch 2.x con ROCm funcione en la RX 580. El entrenamiento en esta GPU **no es el camino principal**. Ver sección 6 para la estrategia de entrenamiento.

### 1.3 Cambio de nodo cliente: Orange Pi vs Mini PC N100

| Criterio | Mini PC N100 (descartado) | Orange Pi 5 8GB (adoptado) |
|---|---|---|
| Arquitectura | x86_64 | ARM64 (RK3588) |
| gr-limesdr instalación | `apt install` directo | Compilar desde fuente |
| GNU Radio 3.10 | Paquete apt disponible | Compilar desde fuente o PPA ARM |
| Rendimiento SDR (12 MSPS) | Verificado en x86 | Requiere verificación en Fase 2 |
| Consumo energético | ~15–25W | ~8–12W (ventaja en campo) |
| Precio | S/ 550–720 | Menor costo |

**Tarea adicional en Fase 2:** Compilar gr-limesdr y GNU Radio 3.10 en la Orange Pi. Reservar al menos 3–5 días de trabajo para esto. Si la compilación falla o el rendimiento es insuficiente para 12 MSPS sostenidos, escalar a una Orange Pi 5 Pro o reintroducir el N100.

---

## 2. Visión general del pipeline de dataset

```
1. CAPTURA       → RTL-SDR V4 + antena discone, PC Ryzen 7 7700x, Lima
2. ETIQUETADO    → detector de energía automático + verificación visual 10%
3. SEGMENTACIÓN  → ventanas de 512 puntos PSD por canal de 6 MHz
4. BALANCEO      → pos_weight en función de pérdida (no submuestreo físico)
5. EMPAQUETADO   → archivos .npz con estructura lista para DataLoader PyTorch
```

---

## 3. Estrategia de captura

### 3.1 Limitaciones del RTL-SDR V4 en este contexto

El RTL-SDR V4 captura hasta **2.4 MSPS de forma estable** (máximo absoluto ~3.2 MSPS con pérdida de muestras). Con 2.4 MSPS, el ancho de banda instantáneo es **2.4 MHz**, lo que cubre menos de la mitad de un canal TVWS de 6 MHz.

Esto significa que cada captura del RTL-SDR **no cubre un canal completo de 6 MHz**. La estrategia correcta es:

- Sintonizar el RTL-SDR al **centro exacto del canal** (ej. 473 MHz para el canal 14)
- Capturar 2.4 MHz centrados en ese canal
- El vector PSD de 512 puntos representará la parte central del canal
- Los flancos del canal (fuera de los 2.4 MHz centrales) no son visibles — aceptable, porque la mayor parte de la energía de una señal de TV digital está en el lóbulo central

**Resolución espectral con este esquema:**
```
Resolución por bin = 2.4 MHz / 512 puntos ≈ 4.7 kHz por bin
Cobertura total de la captura = 2.4 MHz (centrado en el canal)
```

Esta resolución es suficiente para distinguir un canal ocupado (PSD elevado y plano) de uno libre (PSD con solo ruido), que es exactamente la tarea del modelo.

### 3.2 Ubicaciones de captura

Capturar en al menos **3 ubicaciones en Lima** con características distintas:

| Ubicación | Tipo de entorno | Objetivo |
|---|---|---|
| Azotea UNI, Rímac | Urbano denso | Canales TV activos, alta variedad de señales primarias |
| Zona periurbana (ej. Huachipa o Ate) | Semi-urbano | Mix de canales activos y libres |
| Zona semi-rural (ej. Cieneguilla o Lurigancho) | Semi-rural | Simula condiciones del despliegue final |

**Montaje de la antena discone en cada ubicación:**
- Instalar con ≥60 cm de separación de bordes de azotea o techo
- Orientación vertical, sin obstáculos metálicos en radio de 1 m
- Nunca capturar en interiores: reflexiones contaminan el espectro
- Verificar con `rtl_power` que el espectro de TV es visible antes de iniciar

### 3.3 Sesiones de captura por franja horaria

| Franja | Horario | Condición esperada |
|---|---|---|
| Mañana | 06:00 – 09:00 | Pocos canales activos |
| Mediodía | 12:00 – 14:00 | Actividad media |
| Prime time | 19:00 – 22:00 | Máxima ocupación de TV digital |
| Noche | 23:00 – 01:00 | Mínima ocupación — mayoría libres |

Capturar las 4 franjas en al menos 2 ubicaciones. Prime time es la franja más importante: es cuando más canales están ocupados, lo que provee las muestras de clase "ocupado" que el dataset necesita.

### 3.4 Parámetros fijos de captura

```python
SAMPLE_RATE    = 2.4e6       # 2.4 MSPS — máximo estable RTL-SDR V4
GAIN           = "auto"      # ganancia automática del RTL-SDR
N_MUESTRAS     = 2_400_000   # 1 segundo de captura por canal
FORMATO_IQ     = np.complex64

# Los 38 canales TVWS peruanos: centros de 473 a 695 MHz
CANALES_TVWS = {
    canal_num: 470 + (canal_num - 14) * 6 + 3
    for canal_num in range(14, 53)
}
# Ejemplo: canal 14 → 473 MHz, canal 52 → 695 MHz
```

### 3.5 Script de captura

```python
"""
captura_dataset.py
Captura de muestras I/Q canal por canal con RTL-SDR V4.
Ejecutar en el PC Ryzen 7 7700x con el RTL-SDR conectado.

Dependencias: pip install pyrtlsdr numpy
Uso:
  python captura_dataset.py --ubicacion uni_rimac --sesion prime_time --salida ./raw_iq
"""
import rtlsdr
import numpy as np
import os, time, argparse

SAMPLE_RATE = 2.4e6
N_MUESTRAS  = 2_400_000
CANALES_TVWS = {n: 470 + (n - 14) * 6 + 3 for n in range(14, 53)}

def capturar_canal(sdr: rtlsdr.RtlSdr, centro_mhz: float) -> np.ndarray:
    sdr.center_freq  = centro_mhz * 1e6
    sdr.sample_rate  = SAMPLE_RATE
    sdr.gain         = "auto"
    time.sleep(0.15)   # esperar re-lock del PLL del RTL-SDR
    return sdr.read_samples(N_MUESTRAS).astype(np.complex64)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ubicacion", required=True,
                        help="Identificador de ubicación (ej. uni_rimac)")
    parser.add_argument("--sesion",    required=True,
                        help="Franja horaria (manana/mediodia/prime_time/noche)")
    parser.add_argument("--salida",    default="./raw_iq")
    parser.add_argument("--canales",   default="all",
                        help="'all' o lista separada por comas, ej. '14,15,16'")
    args = parser.parse_args()

    os.makedirs(args.salida, exist_ok=True)

    if args.canales == "all":
        canales = list(CANALES_TVWS.keys())
    else:
        canales = [int(c) for c in args.canales.split(",")]

    sdr = rtlsdr.RtlSdr()
    timestamp = int(time.time())

    for canal_num in canales:
        centro_mhz = CANALES_TVWS[canal_num]
        iq = capturar_canal(sdr, centro_mhz)
        nombre = (f"iq_canal{canal_num:02d}"
                  f"_{args.ubicacion}_{args.sesion}_{timestamp}.npy")
        ruta = os.path.join(args.salida, nombre)
        np.save(ruta, iq)
        print(f"Canal {canal_num:2d} ({centro_mhz} MHz) → {ruta}")

    sdr.close()
    print(f"\nCaptura completada: {len(canales)} canales.")

if __name__ == "__main__":
    main()
```

---

## 4. Etiquetado

### 4.1 Detector de energía con umbral adaptativo

```python
import numpy as np
from spectral_sense import calcular_psd

def etiquetar_canal(iq: np.ndarray,
                    margen_db: float = 10.0) -> tuple[int, float]:
    """
    Detecta si un canal está ocupado usando energía espectral.

    Estrategia: comparar la energía media del canal contra el piso de ruido
    estimado del propio vector PSD (percentil 10 de los bins).

    Args:
        iq:        muestras I/Q complex64
        margen_db: dB sobre el piso de ruido para declarar ocupado (default 10 dB)

    Returns:
        (etiqueta, margen_efectivo_db)
        etiqueta: 1=ocupado, 0=libre, -1=zona gris (descartar)
    """
    psd_db        = calcular_psd(iq, n_fft=512, n_segmentos=20)
    piso_ruido    = np.percentile(psd_db, 10)
    energia_media = np.mean(psd_db)
    umbral        = piso_ruido + margen_db
    margen_efect  = energia_media - umbral

    if margen_efect > 3.0:
        return 1, margen_efect     # claramente ocupado
    elif margen_efect < -3.0:
        return 0, margen_efect     # claramente libre
    else:
        return -1, margen_efect    # zona gris → descartar
```

### 4.2 Verificación visual del 10% de las etiquetas

```python
import matplotlib.pyplot as plt

def visualizar_muestra(ruta_iq: str, etiqueta: int, margen_db: float):
    """Grafica PSD para verificación manual de etiqueta."""
    from spectral_sense import calcular_psd
    iq    = np.load(ruta_iq).astype(np.complex64)
    psd   = calcular_psd(iq)
    freqs = np.linspace(-1.2, 1.2, len(psd))

    piso   = np.percentile(psd, 10)
    umbral = piso + 10.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, psd, linewidth=0.8, label="PSD")
    ax.axhline(umbral, color='r', linestyle='--', label=f'umbral ({umbral:.1f} dB)')
    ax.axhline(piso,   color='gray', linestyle=':', label=f'piso ({piso:.1f} dB)')
    ax.fill_between(freqs, pso, umbral, alpha=0.1, color='orange', label='zona gris')

    estado = {1: "OCUPADO", 0: "LIBRE", -1: "ZONA GRIS"}[etiqueta]
    nombre = os.path.basename(ruta_iq)
    ax.set_title(f"{estado} | margen={margen_db:+.1f} dB | {nombre}")
    ax.set_xlabel("Frecuencia relativa (MHz)")
    ax.set_ylabel("PSD (dB)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
```

### 4.3 Casos especiales de etiquetado

| Caso | Etiqueta | Justificación |
|---|---|---|
| Canal TV digital activo (ISDB-Tb) | 1 | Usuario primario presente |
| Canal vacío (solo ruido térmico) | 0 | Sin primario |
| Señal débil (margen < ±3 dB) | −1 (descartar) | Zona gris — contamina entrenamiento |
| Canal en edge del filtro RTL-SDR | −1 (descartar) | PSD degradado por rolloff |
| Interferencia no-TV de baja potencia | 1 | Proteger cualquier señal existente |
| Canal con artefacto DC (LO leakage del RTL-SDR) | Enmascarar bin central | Ver sección 5.2 |

---

## 5. Preprocesamiento

### 5.1 Pipeline I/Q → PSD normalizado

```python
from spectral_sense import calcular_psd, normalizar_psd

def procesar_muestra(ruta_iq: str, n_fft: int = 512) -> np.ndarray:
    """
    Convierte archivo .npy de I/Q al vector PSD normalizado para la CNN.
    CRÍTICO: Este pipeline debe ser IDÉNTICO al que se usa en inferencia.
    """
    iq       = np.load(ruta_iq).astype(np.complex64)
    psd_db   = calcular_psd(iq, n_fft=n_fft, n_segmentos=20)
    psd_norm = normalizar_psd(psd_db)
    return psd_norm   # array (512,) float32 en [0, 1]
```

### 5.2 Corrección del DC offset del RTL-SDR

El RTL-SDR V4 tiene LO leakage (fuga del oscilador local) que produce un pico espurio en el bin central (DC, bin 256 de 512). Si no se corrige, el modelo aprende a asociar el pico de DC con señales ocupadas.

```python
def corregir_dc_offset(psd: np.ndarray, n_bins_mascara: int = 3) -> np.ndarray:
    """
    Enmascara el bin central de DC y sus vecinos inmediatos
    interpolando con los bins adyacentes.

    Args:
        psd:            array (512,) con PSD en dB o normalizado
        n_bins_mascara: número de bins a enmascarar a cada lado del DC

    Returns:
        psd con DC interpolado
    """
    psd_corr  = psd.copy()
    centro    = len(psd) // 2
    inicio    = centro - n_bins_mascara
    fin       = centro + n_bins_mascara + 1

    # Interpolación lineal entre los bordes de la máscara
    izq  = psd[inicio - 1]
    der  = psd[fin]
    psd_corr[inicio:fin] = np.linspace(izq, der, fin - inicio)
    return psd_corr
```

**Orden correcto del pipeline:**
```python
psd_db   = calcular_psd(iq, n_fft=512)
psd_db   = corregir_dc_offset(psd_db)   # primero corregir en dB
psd_norm = normalizar_psd(psd_db)        # después normalizar
```

### 5.3 Normalización y domain mismatch

El RTL-SDR V4 tiene ADC de 8 bits (~48 dB de rango dinámico). El bladeRF tiene ADC de 12 bits (~72 dB). Sin normalización, los vectores PSD del dataset (RTL-SDR) e inferencia (bladeRF) tendrían distribuciones incompatibles.

```python
def normalizar_psd(psd_raw: np.ndarray) -> np.ndarray:
    """
    Normalización por percentiles robustos.
    IDÉNTICA en entrenamiento (RTL-SDR) e inferencia (bladeRF).

    Percentil 5 como piso: descarta el pico de ruido impulsivo
    Percentil 95 como techo: descarta picos de señal aislados
    """
    psd_min  = np.percentile(psd_raw, 5)
    psd_max  = np.percentile(psd_raw, 95)
    psd_norm = (psd_raw - psd_min) / (psd_max - psd_min + 1e-9)
    return np.clip(psd_norm, 0.0, 1.0).astype(np.float32)
```

---

## 6. Estrategia de entrenamiento — PC Ryzen 7 7700x + RX 580 2048SP

### 6.1 Problema con ROCm en la RX 580 2048SP

La RX 580 2048SP usa el chip Polaris 20 (arquitectura GCN 4.0, target `gfx803`). PyTorch 2.x con ROCm oficialmente **no soporta gfx803** desde ROCm 6.0. Los intentos de forzar el soporte requieren compilar PyTorch desde fuente con targets personalizados, lo que está fuera del alcance de la Fase 1.

### 6.2 Opciones de entrenamiento — orden de prioridad

| Opción | Hardware | Tiempo estimado/experimento | Viabilidad |
|---|---|---|---|
| **1 (preferida)** | Google Colab Pro — T4 GPU | 3–8 minutos | Siempre disponible |
| **2** | PC Ryzen 7 7700x — CPU puro | 15–40 minutos | Viable para modelos pequeños |
| **3 (intentar)** | RX 580 con ROCm 5.x + PyTorch 1.13 | Variable | Probar, no garantizado |
| **4 (Fase 2+)** | Mini PC Ryzen 9 8945HS — ONNX DirectML | Solo inferencia | No para entrenamiento |

**Recomendación práctica para Fase 1:**
- Usar el Ryzen 7 7700x para captura, preprocesamiento y construcción del dataset
- Entrenar en Google Colab Pro (T4): el modelo tiene ~107K parámetros, menos de 10 min/experimento
- Si se quiere intentar la RX 580: instalar ROCm 5.7 y PyTorch 1.13, verificar con `torch.cuda.is_available()`

### 6.3 Verificar si la RX 580 es utilizable

```bash
# Instalar ROCm 5.7 en Ubuntu 22.04
wget https://repo.radeon.com/amdgpu-install/5.7/ubuntu/jammy/amdgpu-install_5.7.50700-1_all.deb
sudo dpkg -i amdgpu-install_5.7.50700-1_all.deb
sudo amdgpu-install --usecase=rocm

# Verificar detección de la GPU
rocm-smi
rocminfo | grep gfx

# Instalar PyTorch con ROCm 5.7 (última versión con soporte gfx803 parcial)
pip install torch==1.13.1+rocm5.2 -f https://download.pytorch.org/whl/rocm5.2/torch_stable.html

# Test rápido
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Si `torch.cuda.is_available()` retorna `False`, usar el CPU del Ryzen 7 o Colab.

### 6.4 Configuración de entrenamiento en CPU (Ryzen 7 7700x)

```python
# En spectral_sense.py, el entrenamiento en CPU funciona sin cambios.
# El Ryzen 7 7700x (8C/16T, 5.4 GHz boost) es suficiente para el modelo de 107K params.

# Ajuste recomendado para CPU:
# - Reducir batch_size a 32 (menos memoria de activaciones en RAM)
# - Aumentar num_workers a 6 (aprovechar los 8 núcleos del 7700x)
# - Usar torch.set_num_threads(14) para saturar los núcleos del 7700x

import torch
torch.set_num_threads(14)   # 7700x tiene 8 núcleos físicos, 16 hilos

modelo = entrenar(
    dataset_dir = "./dataset",
    salida_dir  = "./modelos",
    epocas      = 80,
    batch_size  = 32,        # reducido para CPU
    lr          = 1e-3,
    dispositivo = "cpu"
)
```

---

## 7. Construcción del dataset — script completo

```python
"""
construir_dataset.py
Pipeline completo: I/Q crudos → dataset .npz para entrenamiento.

Uso:
  python construir_dataset.py --raw_dir ./raw_iq --salida ./dataset
"""
import os, json, random, argparse
import numpy as np
from spectral_sense import calcular_psd, normalizar_psd

POSICIONES_BARRIDO = [
    {"centro_mhz": 498, "canales_globales": list(range(14, 23))},
    {"centro_mhz": 536, "canales_globales": list(range(22, 31))},
    {"centro_mhz": 578, "canales_globales": list(range(30, 39))},
    {"centro_mhz": 626, "canales_globales": list(range(38, 47))},
    {"centro_mhz": 670, "canales_globales": list(range(46, 53))},
]

def corregir_dc_offset(psd: np.ndarray, n: int = 3) -> np.ndarray:
    psd_c = psd.copy()
    c = len(psd) // 2
    psd_c[c-n:c+n+1] = np.linspace(psd[c-n-1], psd[c+n+1], 2*n+1)
    return psd_c

def etiquetar_canal(iq: np.ndarray) -> tuple[int, float]:
    psd           = calcular_psd(iq, n_fft=512, n_segmentos=20)
    psd           = corregir_dc_offset(psd)
    piso          = np.percentile(psd, 10)
    energia_media = np.mean(psd)
    margen        = energia_media - (piso + 10.0)
    if margen > 3.0:
        return 1, margen
    elif margen < -3.0:
        return 0, margen
    else:
        return -1, margen    # zona gris — descartar

def canal_a_posicion(canal_global: int) -> tuple[int, int]:
    for pos_idx, pos in enumerate(POSICIONES_BARRIDO):
        if canal_global in pos["canales_globales"]:
            return pos_idx, pos["canales_globales"].index(canal_global)
    raise ValueError(f"Canal {canal_global} fuera del rango 14–52")

def procesar_archivo(ruta_iq: str, n_fft: int = 512) -> dict | None:
    nombre    = os.path.basename(ruta_iq)
    partes    = nombre.split("_")
    canal_num = int(partes[1].replace("canal", ""))

    iq = np.load(ruta_iq).astype(np.complex64)

    etiqueta, margen = etiquetar_canal(iq)
    if etiqueta < 0:
        return None   # zona gris → descartar

    psd_db   = calcular_psd(iq, n_fft=n_fft, n_segmentos=20)
    psd_db   = corregir_dc_offset(psd_db)
    psd_norm = normalizar_psd(psd_db)

    pos_idx, idx_local = canal_a_posicion(canal_num)
    etiquetas = np.full(9, -1.0, dtype=np.float32)
    etiquetas[idx_local] = float(etiqueta)

    return {
        "psd":       psd_norm,
        "etiquetas": etiquetas,
        "posicion":  pos_idx,
        "canal":     canal_num,
        "ocupado":   etiqueta
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="./raw_iq")
    parser.add_argument("--salida",  default="./dataset")
    parser.add_argument("--n_fft",   type=int, default=512)
    args = parser.parse_args()

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.salida, split), exist_ok=True)

    archivos = sorted([f for f in os.listdir(args.raw_dir) if f.endswith(".npy")])

    # División POR SESIÓN, no por muestra individual
    # Extraer sesiones únicas del nombre de archivo
    sesiones = sorted(set("_".join(f.split("_")[2:4]) for f in archivos))
    random.shuffle(sesiones)
    n_s      = len(sesiones)
    train_s  = set(sesiones[:int(n_s * 0.70)])
    val_s    = set(sesiones[int(n_s * 0.70):int(n_s * 0.85)])
    # resto → test

    n_guardado   = {"train": 0, "val": 0, "test": 0}
    n_descartado = 0
    stats        = {"libre": 0, "ocupado": 0}

    for archivo in archivos:
        sesion_key = "_".join(archivo.split("_")[2:4])
        if sesion_key in train_s:
            split = "train"
        elif sesion_key in val_s:
            split = "val"
        else:
            split = "test"

        ruta    = os.path.join(args.raw_dir, archivo)
        muestra = procesar_archivo(ruta, args.n_fft)
        if muestra is None:
            n_descartado += 1
            continue

        idx      = n_guardado[split]
        ruta_npz = os.path.join(args.salida, split, f"muestra_{idx:06d}.npz")
        np.savez_compressed(
            ruta_npz,
            psd      = muestra["psd"],
            etiquetas= muestra["etiquetas"],
            posicion = np.array(muestra["posicion"], dtype=np.int8)
        )
        n_guardado[split] += 1
        stats["ocupado" if muestra["ocupado"] else "libre"] += 1

    total = sum(n_guardado.values())
    print(f"\nDataset construido:")
    for split, n in n_guardado.items():
        print(f"  {split}: {n:,} muestras")
    print(f"  Descartadas (zona gris): {n_descartado:,}")
    if total > 0:
        print(f"  Balance → libre: {100*stats['libre']/total:.1f}% | "
              f"ocupado: {100*stats['ocupado']/total:.1f}%")

    metadata = {
        "n_muestras":     n_guardado,
        "n_descartado":   n_descartado,
        "balance_clases": {k: round(v/total, 3) for k, v in stats.items()} if total else {},
        "n_fft":          args.n_fft,
        "hardware_captura": "RTL-SDR Blog V4",
        "normalizacion":  "percentil_5_95",
        "dc_offset_corr": True
    }
    with open(os.path.join(args.salida, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  metadata.json guardado.")

if __name__ == "__main__":
    main()
```

---

## 8. Estructura del dataset en disco

```
dataset/
├── train/                     # ~70% de muestras válidas
│   ├── muestra_000001.npz     # psd(512,) + etiquetas(9,) + posicion(1,)
│   └── ...
├── val/                       # ~15% — validación durante entrenamiento
│   └── ...
├── test/                      # ~15% — NO tocar hasta evaluación final
│   └── ...
└── metadata.json              # estadísticas del dataset
```

**Regla de partición:** La división se hace **por sesión de captura**, no por muestra individual. Si muestras de una misma sesión quedan en train y val, el modelo aprende las condiciones específicas de esa sesión y la validación no mide generalización real.

---

## 9. Data augmentation

Aplicada solo al split de entrenamiento, en tiempo de carga en `TVWSDataset.__getitem__`.

| Técnica | Parámetros | Justificación física |
|---|---|---|
| Ruido gaussiano aditivo | σ ∈ [0.005, 0.02], p=0.5 | Variación del piso de ruido térmico |
| Desplazamiento de amplitud | offset ∈ [−0.05, 0.05], p=0.4 | Variación de ganancia AGC del RTL-SDR |
| Roll espectral (CFO) | ±3 bins máx, p=0.3 | Error de frecuencia residual del TCXO |
| Escalado de amplitud | factor ∈ [0.9, 1.1], p=0.3 | Variación de RSSI entre capturas |

**Técnicas excluidas:** flip horizontal (PSD invertido no es físicamente realizable), mixup entre canales distintos (mezcla de señales incoherentes), recorte aleatorio (rompe la correspondencia posición-canal).

---

## 10. Metas del dataset y métricas de calidad

| Métrica | Mínimo aceptable | Recomendado |
|---|---|---|
| Total de muestras válidas | 30,000 | ≥ 50,000 |
| Muestras de clase "ocupado" | ≥ 20% | 30–40% |
| Ubicaciones distintas | ≥ 2 | 3 |
| Franjas horarias cubiertas | ≥ 3 | 4 |
| Canales TVWS representados (14–52) | ≥ 20 | ≥ 35 |
| Muestras descartadas (zona gris) | — | < 10% del total capturado |
| AUC-ROC en validación tras entrenamiento | ≥ 0.85 | ≥ 0.92 |

Si el balance de clases resulta inferior al 20% de "ocupado", capturar sesiones adicionales en prime time antes de entrenar. No compensar con augmentation: la augmentation no crea señales de TV donde no las hay.

---

## 11. Checklist de Fase 1

### Hardware y software
- [ ] RTL-SDR V4 verificado con `rtl_test -t` (sin overflow de muestras a 2.4 MSPS)
- [ ] Antena discone instalada correctamente (≥60 cm de bordes, exterior)
- [ ] `pyrtlsdr` instalado y probado en el PC Ryzen 7 7700x
- [ ] `spectral_sense.py` verificado con `python spectral_sense.py --modo test`
- [ ] Intentado ROCm 5.7 en RX 580 — resultado documentado

### Captura
- [ ] Capturas en ≥ 2 ubicaciones completadas
- [ ] Capturas en ≥ 3 franjas horarias completadas
- [ ] Archivos raw I/Q respaldados en Google Drive (nomenclatura: `iq_canal##_ubicacion_sesion_timestamp.npy`)

### Etiquetado y construcción
- [ ] Verificación visual del 10% de etiquetas automáticas completada
- [ ] `construir_dataset.py` ejecutado sin errores
- [ ] `metadata.json` generado con balance ≥ 20% ocupado
- [ ] Dataset versionado en Google Drive con carpeta fechada

### Entrenamiento
- [ ] Primer entrenamiento de prueba (≥5 épocas) sin errores en Colab o CPU
- [ ] AUC > 0.85 en split de validación al cierre de Fase 1
- [ ] Modelo exportado a ONNX y verificado con `onnxruntime`

### Nodo cliente (tarea de Fase 2 — identificada en Fase 1)
- [ ] Orange Pi encendida con Ubuntu 22.04 ARM64
- [ ] Plan de compilación de GNU Radio 3.10 + gr-limesdr documentado
- [ ] Reservar 3–5 días de Fase 2 para compilación y prueba de gr-limesdr en ARM

---

*Documento v2 — actualizado con hardware real confirmado del equipo (junio 2026). La v1 asumía 2× RTL-SDR, Mini PC N100 y entrenamiento con GPU local garantizado. Esta versión corrige esas suposiciones.*
