# Radio Cognitiva TVWS — Documentación Técnica Detallada
**Universidad Nacional de Ingeniería (UNI) — Facultad IITMC**  
**Proyecto VRI 2024–2026 | Fechas: 18 mayo → 15 diciembre 2026**  
**PI: Galvez Legua, Mauricio Pedro**

---

## 1. Identidad del Proyecto

| Campo | Valor |
|-------|-------|
| Nombre completo | Diseño y Validación de un prototipo de Radio Cognitiva basado en Hardware SDR asimétrico y Deep Learning para el acceso dinámico a TVWS en zonas rurales |
| Universidad | Universidad Nacional de Ingeniería (UNI), Lima, Perú |
| Facultad | Ingeniería Eléctrica y Electrónica — IITMC |
| Tipo | Investigación Aplicada — VRI Grupo de Investigación |
| Presupuesto total | S/ 40,000 (S/ 16,000 subvenciones + S/ 24,000 equipamiento) |
| Financiamiento | Fondos estatales — compras por portal RNP sin licitación (< 8 UIT) |

**Equipo:**
- Franco Rafael Espinoza — Fase 1
- Victor Manuel Soto — Fase 2
- Sandro Gonzalo Niño — Fase 3

---

## 2. Arquitectura General del Sistema

### 2.1 Concepto de Operación

Enlace punto a punto NLOS/LOS parcial de **15–20 km** entre dos nodos:

```
[Comunidad rural]                              [Localidad con fibra]
  Teléfono/PC  ←→  Nodo Cliente  ←→  20 km RF  ←→  Nodo Gateway  ←→  Internet
                   LimeSDR Mini 2.0              bladeRF 2.0 micro xA4
                   Mini PC N100                  Mini PC Ryzen 9 8945HS
```

**Principio de radio cognitiva:** El sistema opera en banda TVWS (470–698 MHz) como usuario secundario. La CNN en el Gateway monitoriza el espectro continuamente y ordena saltos de canal cuando detecta un emisor primario (canal de TV activo).

### 2.2 Tres Planos de Comunicación

| Plano | Medio | Dirección | Función |
|-------|-------|-----------|---------|
| **Datos** | OFDM 6 MHz en TVWS | DL: Gateway→Cliente / UL: Cliente→Gateway | Tráfico de usuario (video, voz, datos) |
| **Control** | LoRa SX1262 a 915 MHz | Bidireccional | Señalización cognitiva (next_ch, t_hop, ACK, métricas) |
| **Sensado** | bladeRF RX2 | Gateway escucha espectro | Entrada de datos para la CNN |

### 2.3 Banda de Operación

- **TVWS UHF:** 470–698 MHz (plan de atribución peruano)
- **Canales disponibles:** 39 canales de 6 MHz entre 470 y 698 MHz
- **Canal de control:** 915 MHz (ISM, fuera de banda TVWS)
- **Frecuencia de referencia para cálculos:** 600 MHz (centro de banda)

---

## 3. Hardware por Nodo

### 3.1 Nodo Gateway

| Componente | Especificación | Función |
|------------|---------------|---------|
| SDR | bladeRF 2.0 micro xA4 | TX datos DL + RX sensado espectral |
| PC | Mini PC Ryzen 9 8945HS, 32 GB DDR5 | GNU Radio + inferencia CNN ONNX |
| PA | 400–1000 MHz, 2W (33 dBm), clase AB, 12V | Amplificar TX antes de la antena |
| LNA RX (propuesto) | 400–1000 MHz, NF ≤1 dB, ganancia 20 dB | Bajar figura de ruido del RX uplink |
| Antena enlace | LPDA 400–2700 MHz, 10 dBi, conector N-hembra | TX downlink + RX uplink |
| Módulo LoRa | SX1262, 915 MHz, TX ≥22 dBm, USB-C CDC-ACM | Canal de control bidireccional |
| Antena LoRa | Colineal 900–930 MHz, 5 dBi, N-hembra, IP65 | Enlace de control |
| Cable RF principal | LMR-400 3.5m + conectores N + supresor GDT | Baja pérdida entre PA y antena |
| Pigtail TX | RG-316 0.6m, bladeRF → PA | Tramo corto de baja pérdida |

**Cadena RF TX Gateway:**
```
bladeRF TX1 (+6 dBm)
  → pigtail RG-316 0.6m (−0.47 dB)
  → PA 2W (+27 dB)
  → LMR-400 3.5m + 3 conectores N + supresor GDT (−0.84 dB)
  → LPDA 10 dBi
  → EIRP: +41.7 dBm
```

**Cadena RF RX Gateway (uplink):**
```
LPDA 10 dBi
  → LNA propuesto (NF ≤1 dB, +20 dB)   ← montar lo más cerca de la antena
  → LMR-400 3.5m (−0.84 dB)
  → bladeRF RX1 (NF ~3.5 dB)
  → NF sistema total: ~1.04 dB
```

### 3.2 Nodo Cliente

| Componente | Especificación | Función |
|------------|---------------|---------|
| SDR | LimeSDR Mini 2.0 (chip LMS7002M) | RX datos DL + TX uplink |
| PC | Mini PC Intel N100, 8–16 GB DDR4 | GNU Radio RX/TX (sin IA) |
| LNA RX | 400–1000 MHz, NF ≤1 dB, ganancia 20 dB | Bajar figura de ruido del RX downlink |
| Antena enlace | LPDA 400–2700 MHz, 10 dBi, conector N-hembra | RX downlink + TX uplink |
| Módulo LoRa | SX1262, 915 MHz, TX ≥22 dBm, USB-C CDC-ACM | Canal de control bidireccional |
| Antena LoRa | Colineal 900–930 MHz, 5 dBi, N-hembra, IP65 | Enlace de control |
| Cable RF principal | LMR-400 3.5m | Mismas pérdidas que Gateway |
| Gabinete | IP65 | Protección ambiental en campo |

**Cadena RF RX Cliente (downlink):**
```
LPDA 10 dBi
  → LNA (NF ≤1 dB, +20 dB)   ← montar en el mástil, cerca de la antena
  → LMR-400 3.5m (−0.84 dB)
  → LimeSDR RX (NF ~5 dB, corregido por LNA a ~1 dB efectivo)
  → PRx en QPSK: −78 dBm | margen: +19.7 dB
```

**Cadena RF TX Cliente (uplink):**
```
LimeSDR TX (+10 dBm, límite del chip LMS7002M)
  → LMR-400 3.5m (−0.84 dB)
  → LPDA 10 dBi
  → EIRP: +20.0 dBm   ← SIN PA (el único PA está en el Gateway)
```

**Limitaciones conocidas del LimeSDR Mini 2.0:**
- DC offset / LO leakage en subportadora central → calibrar con LimeSuite antes de captura
- IQ imbalance → usar bloque IQ Corrector de GNU Radio en tiempo real
- Un solo RX externo → sensado time-multiplexado cada ~500 ms (sin barrido continuo)
- Sin timestamping hardware preciso → ventana de guarda de 10 ms en saltos
- Ecosistema de driver distinto al bladeRF (gr-limesdr vs gr-bladeRF) → capa de abstracción Python (clase `RadioInterface`)

---

## 4. Parámetros OFDM del Canal de Datos

### 4.1 Configuración de la Trama

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Ancho de banda | 6 MHz | Canal TVWS estándar peruano |
| Subportadoras totales (FFT) | 512 puntos | Resolución frecuencial adecuada |
| Subportadoras de datos | ~420 | Resto son guardas y pilotos |
| Subportadoras piloto | ~55 dispersas | Estimación de canal + corrección CFO |
| Subportadoras de guarda | ~52 (26 cada extremo) | Separación espectral con canales vecinos |
| Subportadoras de control | 4 (sub #254–257) | Canal in-band propuesto (Opción A) |
| Prefijo cíclico (CP) | 1/4 del símbolo (~56 µs) | Protección contra multipath |
| Duración símbolo OFDM | ~89 µs (CP + FFT) | 640 muestras a 7.68 MSPS |
| Modulaciones soportadas | BPSK / QPSK / 16-QAM | Selección adaptativa por CNN |
| Tasa de código FEC | 1/2 o 3/4 (convolucional o LDPC) | Protección contra errores de canal |

### 4.2 Distribución de Subportadoras (vector IFFT de 512 puntos)

```
Índice 0–25:     Banda de guarda inferior (26 sub → cero, no se transmiten)
Índice 26–253:   Datos + pilotos dispersos (bloque inferior ~228 sub)
Índice 254–257:  Campo de control in-band (next_ch + t_hop + CRC-16) [Opción A]
Índice 258–486:  Datos + pilotos dispersos (bloque superior ~229 sub)
Índice 487–511:  Banda de guarda superior (25 sub → cero)

NOTA: Subportadora #255 (DC) EVITADA — LimeSDR tiene LO leakage en DC
```

### 4.3 Throughput por Modo

| Modulación | Tasa código | Throughput bruto | Factor corrección | Throughput neto |
|-----------|-------------|-----------------|-------------------|----------------|
| BPSK | 1/2 | 6 Mbps | × 0.321 | ~1.9 Mbps |
| BPSK | 3/4 | 6 Mbps | × 0.482 | ~2.9 Mbps |
| QPSK | 1/2 | 12 Mbps | × 0.321 | ~3.9 Mbps |
| QPSK | 3/4 | 12 Mbps | × 0.482 | ~5.8 Mbps |
| 16-QAM | 3/4 | 24 Mbps | × 0.482 | ~11.6 Mbps |

**Factor de corrección = R_FEC × (1 − CP) × (1 − pilotos) × (1 − guardas)**  
= 0.75 × (640/512 × inversión) × 0.893 × 0.898 ≈ 0.482 para R=3/4

---

## 5. Cadena TX Downlink (Gateway → Cliente)

### 5.1 Flujo Completo en el Gateway

```
[Datos usuario / Internet]
    ↓
[1] Capa MAC — empaquetado
    Cabecera: seq_num | next_ch | mod_scheme | CRC
    ↓
[2] Codificación FEC (gr-fec)
    Convolucional o LDPC — añade bits de redundancia
    ↓
[3] Mapeador de constelación
    Bits → símbolos complejos I+jQ
    BPSK: {+1, −1} / QPSK: {±1±j}/√2 / 16-QAM: {±1±3j, ±3±j, ±3±3j}/√10
    ↓
[4] OFDM Carrier Allocator (GNU Radio)
    Distribuye símbolos en las 512 posiciones del vector IFFT:
    · Posiciones de guarda → 0+0j
    · Posiciones piloto → valores BPSK fijos conocidos por ambos extremos
    · Posiciones de control (#254–257) → campo next_ch + t_hop + CRC-16
    · Posiciones de datos → símbolos de constelación
    ↓
[5] IFFT 512 puntos
    Convierte dominio frecuencia → dominio tiempo
    Genera forma de onda OFDM compleja
    ↓
[6] Inserción prefijo cíclico (CP = 128 muestras)
    Copia las últimas 128 muestras al inicio del símbolo
    Protege contra multipath con delay ≤56 µs
    ↓
[7] Flujo de muestras IQ → bladeRF por USB 3.0
    Formato: complejos float32 / int16
    ↓
[8] bladeRF DAC (12 bits, 61 MSPS)
    Convierte muestras digitales → señal analógica de banda base
    ↓
[9] Up-conversion a canal TVWS (ej. 584 MHz)
    VCTCXO ±1 ppm → drift máx: 584 Hz → corregible por CFO
    ↓
[10] Amplificador de Potencia (PA) 2W
    Ganancia: +27 dB | Salida: ~+33 dBm
    ↓
[11] Cable LMR-400 + supresor GDT (−0.84 dB)
    ↓
[12] LPDA 10 dBi → aire
    EIRP final: +41.7 dBm
```

### 5.2 Flujo de Recepción en el Cliente

```
[LPDA 10 dBi recibe señal a ~−78 dBm]
    ↓
[1] LNA externo (NF ≤1 dB, +20 dB)
    Mínimo ruido posible antes del ADC
    ↓
[2] LimeSDR ADC (12 bits, ≥30 MSPS)
    Muestrea la señal en RF → down-convert a banda base
    ↓
[3] Corrección DC offset (LimeSuite al inicio)
    Elimina el LO leakage en subportadora DC
    ↓
[4] IQ Corrector (GNU Radio, tiempo real)
    Corrige desbalance de amplitud y fase entre I y Q
    ↓
[5] Sincronización Schmidl-Cox
    Correlaciona el prefijo cíclico consigo mismo
    Detecta el inicio exacto de cada símbolo OFDM
    Estima el CFO (Carrier Frequency Offset)
    ↓
[6] Corrección CFO
    Multiplica la señal por e^(−j·2π·CFO·t) para compensar el desfase
    ↓
[7] FFT 512 puntos
    Convierte dominio tiempo → dominio frecuencia
    Recupera los 512 símbolos complejos
    ↓
[8] Ecualización del canal
    Pilotos recibidos / pilotos conocidos = H(f) por subportadora
    Divide cada subportadora de datos por H(f) estimado interpolarmente
    ↓
[9] Extracción campo de control (sub #254–257)
    Decodifica next_ch y t_hop antes de los datos
    Verifica CRC-16 — si válido: programa el salto
    ↓
[10] Demapeo de constelación
    Símbolos complejos → bits (decisión de máxima verosimilitud)
    ↓
[11] Decodificación FEC (Viterbi o LDPC)
    Corrige errores usando la redundancia añadida en el transmisor
    ↓
[12] Capa MAC — desencapsulado
    Verifica CRC de la trama → entrega paquetes IP al SO
    ↓
[Usuario recibe datos / Internet]
```

---

## 6. Cadena TX Uplink (Cliente → Gateway)

### 6.1 Flujo TX en el Cliente

```
[Datos usuario hacia Internet]
    ↓
[1] Capa MAC — empaquetado
    Igual que downlink pero sin campo de control
    (el control de retorno viaja por LoRa)
    ↓
[2] Codificación FEC
    Tasa 1/2 preferida (margen más ajustado en uplink)
    ↓
[3] Mapeador BPSK
    Modulación forzada a BPSK en condición NLOS 15 dB y 20 km
    Margen con BPSK (sin LNA en GW): +0.9 dB — al límite
    Margen con BPSK (con LNA en GW): +4.2 dB — funcional
    QPSK posible solo con LNA en Gateway: margen +1.3 dB
    ↓
[4] OFDM Carrier Allocator
    Misma estructura de 512 subportadoras
    Sin campo de control in-band (el control sube por LoRa)
    ↓
[5] IFFT 512 + CP 128 muestras
    ↓
[6] LimeSDR TX (LMS7002M)
    Calibración DC offset automática (~5 ms al activar TX, via LimeSuite)
    PLL ya bloqueado en el canal activo → lock time: ~0 ms adicional
    DAC 12 bits → up-convert al canal TVWS activo
    ↓
[7] Potencia de salida: +10 dBm (límite del chip a 600 MHz)
    SIN amplificador de potencia externo
    ↓
[8] Cable LMR-400 3.5m (−0.84 dB)
    ↓
[9] LPDA 10 dBi → aire
    EIRP: +20.0 dBm (21.7 dB menos que el downlink)
```

### 6.2 Flujo RX en el Gateway (uplink)

```
[LPDA 10 dBi recibe señal a ~−99 dBm (sin LNA) / −99 dBm (con LNA)]
    ↓
[1] LNA propuesto (NF ≤1 dB, +20 dB) — montar cerca de la antena
    Eleva la señal y fija la figura de ruido del sistema en ~1.04 dB
    ↓
[2] LMR-400 3.5m (−0.84 dB)
    ↓
[3] bladeRF RX1 (ADC 12 bits, 61 MSPS)
    RX2 está simultáneamente en barrido espectral (CNN)
    El bladeRF xA4 tiene RX1 y RX2 independientes — no hay conflicto
    ↓
[4–12] Mismo proceso que en el Cliente RX downlink
    Sync Schmidl-Cox → FFT → Ecualización → Demapeo → Viterbi → MAC
    ↓
[Paquetes IP entregados al Mini PC → fibra → Internet]
```

### 6.3 Link Budget Comparativo

| Parámetro | Downlink | Uplink (sin LNA GW) | Uplink (con LNA GW) |
|-----------|----------|---------------------|---------------------|
| EIRP TX | +41.7 dBm | +20.0 dBm | +20.0 dBm |
| FSPL 600 MHz, 20 km | −114 dB | −114 dB | −114 dB |
| Pérdida NLOS estimada | −15 dB | −15 dB | −15 dB |
| Ganancia antena RX | +10 dBi | +10 dBi | +10 dBi |
| Pérdida cable RX | −0.84 dB | −0.84 dB | −0.84 dB |
| **Potencia recibida** | **−78 dBm** | **−99 dBm** | **−99 dBm** |
| NF sistema receptor | ~1.0 dB | ~4.34 dB | ~1.04 dB |
| Sensibilidad BPSK | −100.7 dBm | −97.4 dBm | −100.7 dBm |
| Sensibilidad QPSK | −97.7 dBm | −94.4 dBm | −97.7 dBm |
| **Margen BPSK** | **+22.7 dB ✓** | **+0.9 dB ⚠** | **+4.2 dB ✓** |
| **Margen QPSK** | **+19.7 dB ✓** | **−1.6 dB ✗** | **+1.3 dB ✓** |

---

## 7. Canal de Control — LoRa SX1262

### 7.1 Configuración del Módulo

| Parámetro | Valor |
|-----------|-------|
| Chip | Semtech SX1262 |
| Frecuencia | 902–928 MHz (ISM, banda 915 MHz) |
| Potencia TX | ≥22 dBm |
| Sensibilidad RX | ≤−136 dBm (SF12) |
| Spreading Factor | SF12 (máxima cobertura) |
| Ancho de banda | BW125 kHz |
| Interfaz | USB-C (modo CDC-ACM — puerto serie) |
| Firmware | Puente serie transparente |
| Control desde Python | `pyserial` — 10 líneas de código |
| Antena | Colineal 5 dBi, 915 MHz, N-hembra, IP65 |
| Montaje | Side arm 60 cm bajo la LPDA en el mismo mástil |
| Separación LPDA–LoRa | ~1.83λ a 915 MHz → >20 dB de aislamiento |

### 7.2 Estructura del Mensaje de Control (Gateway → Cliente)

```
Paquete LoRa DL — ~50 bytes totales

Byte 0:    next_ch     [6 bits útiles] — índice del canal TVWS destino (0–38)
Byte 1:    t_hop       [8 bits] — tiempo en slots de 10 ms hasta el salto
Byte 2:    mod_scheme  [2 bits] — 00=BPSK, 01=QPSK, 10=16QAM, 11=reservado
Byte 3:    flags       [8 bits] — bits de estado del sistema
Bytes 4–5: CRC-16      — checksum de bytes 0–3
Bytes 6–49: padding + metadatos opcionales (RSSI del primario, nivel de ocupación)
```

### 7.3 Estructura del Mensaje de Retorno (Cliente → Gateway)

```
Paquete LoRa UL — ~30 bytes totales

Byte 0:    hop_ack     [1 bit] — confirmación de recepción del salto
Byte 0:    ch_actual   [6 bits] — canal actualmente sintonizado en el Cliente
Byte 1:    RSSI_rx     [8 bits] — RSSI medido del downlink (dBm + 128)
Byte 2:    SNR_rx      [8 bits] — SNR estimado del downlink (dB × 4)
Byte 3:    BER_est     [8 bits] — BER estimada (escala logarítmica)
Bytes 4–5: CRC-16
```

### 7.4 Flujo TX en el Gateway (control hacia Cliente)

```
[CNN decide: next_ch=36, t_hop=10ms, mod=QPSK]
    ↓
[1] Python construye paquete de control (50 bytes)
    ↓
[2] pyserial escribe al puerto CDC-ACM del SX1262 Gateway
    /dev/ttyUSB0 (Linux) — 115200 baud
    ↓
[3] SX1262 Gateway: modulación LoRa SF12 + BW125
    Tiempo de transmisión air-time: ~2 s para 50 bytes con SF12
    ↓
[4] Antena colineal 5 dBi → aire (915 MHz)
    EIRP: ≥27 dBm
```

### 7.5 Flujo RX en el Cliente (recepción del control)

```
[Antena colineal 5 dBi recibe señal a ~−87 dBm]
    (Margen: +39.5 dB — enlace extremadamente robusto)
    ↓
[1] SX1262 Cliente desmodula LoRa SF12
    ↓
[2] Puerto CDC-ACM → pyserial en Python del Mini PC N100
    ↓
[3] Parser verifica CRC-16
    Si válido: registra next_ch y t_hop
    Si inválido: descarta (el Gateway reenvía en el próximo ciclo)
    ↓
[4] En T = t_hop × 10 ms:
    GNU Radio recibe señal de resintonización
    LimeSDR PLL configura nueva frecuencia (lock: ~1–3 ms)
    Ventana de silencio: 10 ms
    ↓
[5] Confirma salto vía LoRa UL (hop_ack=1, ch_actual=next_ch)
```

### 7.6 Link Budget del Canal LoRa

| Parámetro | Valor |
|-----------|-------|
| EIRP TX (ambos nodos) | ≥+27 dBm (+22 dBm TX + 5 dBi antena) |
| FSPL a 20 km, 915 MHz | −116.7 dB |
| Ganancia antena RX | +5 dBi |
| Potencia recibida estimada | ≈ −84.7 dBm |
| Sensibilidad SF12 | −136 dBm |
| **Margen del enlace LoRa** | **+51.3 dB — prácticamente irrompible** |

---

## 8. Canal de Control In-Band — Opción A (propuesta)

La Opción A complementa al LoRa añadiendo el campo de control directamente en las subportadoras OFDM, reduciendo la latencia del control de 100 ms a <1 ms para el caso downlink.

### 8.1 Estructura del Campo de Control In-Band

```
4 subportadoras OFDM dedicadas (sub #254–257), valor en cada símbolo:

Sub #254: byte 0 → next_ch (canal destino, 6 bits útiles)
Sub #255: EVITADA — DC offset del LimeSDR
Sub #256: byte 1 → t_hop (slots de 10 ms, 8 bits)
Sub #257: bytes 2–3 → CRC-16 del payload (split en 2 subportadoras)

Cada subportadora lleva un símbolo BPSK de 1 bit efectivo.
El campo completo = 4 bytes por símbolo OFDM (~89 µs de actualización).
```

### 8.2 TX del Campo de Control (en GNU Radio Gateway)

```
[CNN produce: next_ch=36, t_hop=10]
    ↓
[1] Construye paquete de 4 bytes en Python
    next_ch=36 (0b100100) | t_hop=10 | CRC-16
    ↓
[2] Tagged Stream Mux (GNU Radio)
    Inyecta los 4 bytes en las posiciones #254, #256, #257 del vector
    antes de pasar al OFDM Carrier Allocator
    ↓
[3] OFDM Carrier Allocator asigna:
    · Posiciones de datos: símbolos QPSK de los datos de usuario
    · Posiciones piloto: valores BPSK fijos
    · Posiciones control (#254, 256, 257): símbolos del campo de control
    · Posición guarda: 0+0j
    ↓
[4] IFFT + CP → bladeRF → PA → antena
    El campo de control viaja embebido en cada símbolo OFDM
    Sin interrupción del flujo de datos
```

### 8.3 RX del Campo de Control (en GNU Radio Cliente)

```
[FFT recupera los 512 símbolos en frecuencia]
    ↓
[1] Tag Source Block (GNU Radio personalizado)
    Extrae las subportadoras #254, #256, #257 ANTES del demapeador de datos
    ↓
[2] Decodifica 4 bytes → next_ch + t_hop + CRC-16
    ↓
[3] Verifica CRC-16:
    Si válido: programa salto en t_hop × 10 ms
    Si inválido: descarta — espera confirmación en próximo símbolo (cada 89 µs)
    ↓
[4] El proceso de demodulación de datos continúa normalmente en paralelo
```

### 8.4 Redundancia y Robustez

El campo de control se repite en **cada símbolo OFDM** durante el período de pre-anuncio (10–20 tramas = 100–200 ms). Si la tasa de error de paquete (PER) del canal es del 10%, la probabilidad de que fallen 20 intentos independientes es 0.1^20 ≈ 10^-20. En la práctica, el campo de control llega con certeza.

---

## 9. Modelo de IA — CNN de Sensado Espectral

### 9.1 Arquitectura

| Parámetro | Valor |
|-----------|-------|
| Tipo | Red neuronal convolucional 1D (1D-CNN) |
| Entrada | Vector PSD de 1024 puntos (FFT sobre muestras I/Q del RX2) |
| Salida | Clasificación canal libre/ocupado por cada canal TVWS |
| Parámetros | ~350,000–600,000 |
| Tamaño modelo | ~2–4 MB |
| Inferencia | <5 ms en Mini PC con ONNX Runtime |
| Entrenamiento | PyTorch en Google Colab Pro (T4) o RTX 4060 local |
| Despliegue | ONNX exportado con `torch.onnx.export()` |
| Ciclo de sensado | 100–200 ms (RX2 del bladeRF en barrido continuo) |

### 9.2 Dataset y Domain Mismatch

- **Captura:** RTL-SDR (8 bits ADC) en Fase 1 — antena discone 25–1300 MHz
- **Inferencia en campo:** bladeRF (12 bits ADC)
- **Mitigación:** Normalización por rango dinámico antes de la FFT en el preprocesamiento

### 9.3 KPIs del Modelo

| Métrica | Objetivo |
|---------|---------|
| Accuracy en dataset de prueba | >90% |
| Tasa de falsos negativos (libre → ocupado) | <5% |
| Tiempo de reacción al primario (end-to-end) | <300 ms |

---

## 10. Latencias del Sistema

| Componente | Valor | Origen |
|------------|-------|--------|
| Propagación RF (20 km) | 0.067 ms | Física (c = 3×10⁸ m/s) |
| Duración símbolo OFDM | ~89 µs | 640 muestras / 7.68 MSPS |
| Inferencia CNN (ONNX) | <5 ms | Documentado |
| Ventana de guarda en salto | 10 ms | Diseño (PLL lock + margen) |
| Ciclo de sensado CNN | 100–200 ms | bladeRF RX2 |
| Canal LoRa (50 bytes SF12) | ~100 ms | Air-time SX1262 |
| Latencia E2E datos | 5–15 ms | Prop + procesamiento GNU Radio |
| **Tiempo de evacuación de canal** | **<300 ms** | Sensado + CNN + LoRa + guarda |

---

## 11. Modelos de LNA Candidatos para el Gateway

| Modelo | NF @ 600 MHz | Ganancia | Precio aprox. | Alimentación DC directa | Apto campo |
|--------|-------------|----------|---------------|------------------------|------------|
| RTL-SDR Blog Wideband LNA (SPF5189Z) | <1.0 dB | ~18.7 dB | ~S/ 68 | Jumper interno (pierde caja) | Con modificación |
| Nooelec LaNA | <0.8 dB | ~20 dB | ~S/ 270 | Sí — DC barrel incluido | Sí — caja aluminio |
| GPIO Labs Ultra LNA | ~0.8–0.9 dB | ~20 dB | ~S/ 130 | Sí — micro-USB / DC | Requiere enclosure |

**Especificación genérica para RNP:** "Amplificador de bajo ruido (LNA), banda 400–1000 MHz, figura de ruido ≤1.0 dB, ganancia 15–25 dB, impedancia 50 Ω, conector SMA hembra, alimentación DC 5V o bias tee, con enclosure."

---

## 12. Cronograma y Presupuesto

| Fase | Descripción | Inicio | Fin | Presupuesto | Responsable |
|------|-------------|--------|-----|-------------|-------------|
| F1 | Recolección dataset + modelado IA + compras HW | 18/05/2026 | 15/06/2026 | S/ 19,037 | Franco Rafael Espinoza |
| F2 | Integración arquitectura SDR + capa MAC | 15/06/2026 | 15/08/2026 | S/ 7,085 | Victor Manuel Soto |
| F3 | Enlace piloto urbano (azotea UNI) | 16/08/2026 | 15/10/2026 | S/ 5,300 | Sandro Gonzalo Niño |
| F4 | Despliegue rural + pruebas TVWS | 15/10/2026 | 01/12/2026 | S/ 5,378 | — |
| F5 | Análisis resultados + informe final | 02/12/2026 | 15/12/2026 | S/ 3,200 | — |

**Alertas críticas:**
- F1: Publicar TODO en RNP el 18/05 — bladeRF y LimeSDR importados tardan 4–8 semanas en aduana
- F5: Solo 13 días — iniciar redacción del informe final desde F4 en paralelo
- Uplink a 20 km: evaluar adición de LNA en Gateway (S/ 400) antes de Fase 4

---

## 13. Inconsistencias Identificadas y Estado

| # | Inconsistencia | Estado |
|---|----------------|--------|
| 1 | PlutoSDR (USB 2.0, incapaz para OFDM 6 MHz) | ✅ Reemplazado por LimeSDR Mini 2.0 |
| 2 | Rango TVWS declarado "400–800 MHz" | ✅ Corregido a 470–698 MHz (normativa peruana) |
| 3 | Mástiles 3m insuficientes para zona de Fresnel a 20 km | ✅ Solución: elevación natural + mástil 10m en F4 |
| 4 | MIMO descartado (LimeSDR tiene 1 solo RX externo) | ✅ Operación SISO confirmada |
| 5 | CNN entrenada con RTL-SDR (8b) → inferencia en bladeRF (12b) | ✅ Mitigado con normalización |
| 6 | Solo 1 PA en el sistema (Gateway) — uplink sin amplificar | ⚠ Margen BPSK = +0.9 dB — añadir LNA en GW |
| 7 | Side arm Fase 2: figura 1 unidad, posiblemente necesarias 2 | ⚠ Verificar |
| 8 | Pigtail MCX→SMA para RTL-SDRs no listado | ⚠ Añadir al kit cables Fase 1 |
| 9 | F5 tiene solo 13 días para informe final | ⚠ Redactar desde F4 |

---

## 14. Software

| Herramienta | Uso |
|-------------|-----|
| Ubuntu 22.04 LTS | SO en ambos nodos |
| GNU Radio 3.10.x | Procesamiento de señal SDR |
| gr-bladeRF | Driver Gateway |
| gr-limesdr | Driver Cliente |
| LimeSuite | Calibración DC offset y IQ del LimeSDR |
| PyTorch | Entrenamiento CNN (Google Colab / RTX 4060) |
| ONNX Runtime | Inferencia en campo (Mini PC Gateway) |
| pyserial | Control del canal LoRa por USB-CDC |
| GitHub (privado) | Control de versiones |
| Notion | Gestión de tareas y wiki |
| Overleaf | Informes en LaTeX |

---

*Documento generado a partir de la sesión de diseño técnico del proyecto. Fecha de referencia: junio 2026.*
