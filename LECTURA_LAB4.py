import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import pywt

#----------------------------------FUNCIONES--------------------------------#
fm = 500 # Frecuencia de muestreo en Hz

# Definir funciones para los filtros
def butter_highpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_filters(data, fs):
    # Filtro pasa alto de 0.5 Hz
    b, a = butter_highpass(0.5, fs)
    filtered_data = filtfilt(b, a, data)
    
    # Filtro pasa bajo de 40 Hz
    b, a = butter_lowpass(40, fs)
    filtered_data = filtfilt(b, a, filtered_data)
    
    return filtered_data


#----------------------------------EXTRACCION DATOS--------------------------------#
# Nombre del archivo que contiene los datos
datos = 'datos_ecg_filtrados.txt'
# Leer los datos del archivo .txt
ecg_data = np.loadtxt(datos)
# Cantidad total de datos y duración total en segundos
num_samples = len(ecg_data)  # Número total de muestras
total_time_seconds = 5 * 60  # 5 minutos en segundos
# Crear un vector de tiempo para la gráfica
t = np.linspace(0, total_time_seconds, num_samples)  # Vector de tiempo desde 0 hasta 300 segundos

#----------------------------------APLICAR FILTROS--------------------------------#

# Aplicar los filtros
filtered_ecg = apply_filters(ecg_data, fm)

#----------------------------------DETECCIÓN PICOS--------------------------------#
# Detectar picos R
# Aquí usamos 'distance' para evitar detectar picos muy cercanos
peaks, _ = find_peaks(filtered_ecg, distance=0.5 * fm, height=np.mean(filtered_ecg) + 0.3 * np.std(filtered_ecg))
# Calcular intervalos R-R
r_r_intervals = np.diff(peaks) / fm  # Intervalos R-R en segundos


#----------------------------------REDIMENSIONAR--------------------------------#


# Transformar el eje de tiempo y seleccionar los primeros 30 segundos
time = np.arange(len(ecg_data)) / fm * 1000  # Convertir el eje de muestras a tiempo en milisegundos
time_30s = time[time <= 10000]  # Filtrar solo los primeros 30 segundos
ecg_30s = ecg_data[:len(time_30s)]
filtered_ecg_30s = filtered_ecg[:len(time_30s)]
peaks_30s = peaks[peaks <= len(time_30s)]



#----------------------------------PARÁMETROS HVR--------------------------------#

# lista de intervalos R-R en milisegundos
r_r_intervals_ms = r_r_intervals * 1000  # Convierte los intervalos R-R a milisegundos si es necesario

# 0. Promedio del Intervalo R-R (RRi)
RRi_mean = np.mean(r_r_intervals_ms)

# 1. Cálculo de la frecuencia cardíaca a partir de los intervalos R-R
heart_rate = 60 / r_r_intervals  # Frecuencia cardíaca en latidos por minuto

# 2. Desviación Estándar de los Intervalos R-R (SDNN)
SDNN = np.std(r_r_intervals_ms)

# 3. SDSD: Desviación estándar de las diferencias entre intervalos R-R adyacentes
r_r_diff = np.diff(r_r_intervals_ms)
SDSD = np.std(r_r_diff)

# 4. rMSSD: Raíz cuadrada de la media de las diferencias al cuadrado entre intervalos R-R adyacentes
rMSSD = np.sqrt(np.mean(r_r_diff ** 2))

# 5. pNN50: Porcentaje de intervalos R-R con una diferencia mayor a 50 ms
pNN50 = np.sum(np.abs(r_r_diff) > 50) / len(r_r_diff) * 100

# Imprimir resultados
print(f"Promedio del Intervalo R-R (RRi): {RRi_mean:.2f} ms")
print(f"Desviación Estándar de los Intervalos R-R (SDNN): {SDNN:.2f} ms")
print(f"Desviación Estándar de las Diferencias (SDSD): {SDSD:.2f} ms")
print(f"Raíz Cuadrada de la Media de las Diferencias Cuadradas (rMSSD): {rMSSD:.2f} ms")
print(f"Porcentaje de Intervalos R-R con diferencia > 50 ms (pNN50): {pNN50:.2f}%")

#----------------------------------TRANSFORMADA WAVELET--------------------------------#

# Definir los rangos de frecuencia para cada componente de HRV
frecuencia_ultra_baja = (0.0, 0.003)  # ULF (≤ 0.003 Hz)
frecuencia_muy_baja = (0.003, 0.04)   # VLF (0.003 - 0.04 Hz)
frecuencia_baja = (0.04, 0.15)        # LF (0.04 - 0.15 Hz)
frecuencia_alta = (0.15, 0.4)         # HF (0.15 - 0.4 Hz)
sampling_period = 1.0  # Ajuste según el intervalo de muestreo de r_r_intervals (1.0 si está en segundos)

# Ajuste de escalas para obtener frecuencias en el rango deseado
scales = np.arange(1, 200)  # Escalas bajas para obtener frecuencias más altas
coef, freqs = pywt.cwt(r_r_intervals, scales, 'cmor1.5-0.5', sampling_period)

# Verificación de frecuencias calculadas
print("Rango de frecuencias de TWC:", freqs.min(), "a", freqs.max(), "Hz")

# Identificar índices de las frecuencias de interés en las escalas
idx_ultra_baja = np.where((freqs >= frecuencia_ultra_baja[0]) & (freqs <= frecuencia_ultra_baja[1]))[0]
idx_muy_baja = np.where((freqs >= frecuencia_muy_baja[0]) & (freqs <= frecuencia_muy_baja[1]))[0]
idx_baja = np.where((freqs >= frecuencia_baja[0]) & (freqs <= frecuencia_baja[1]))[0]
idx_alta = np.where((freqs >= frecuencia_alta[0]) & (freqs <= frecuencia_alta[1]))[0]

# Calcular potencia espectral en cada banda
potencia_ultra_baja = np.mean(np.abs(coef[idx_ultra_baja, :])**2, axis=0) if idx_ultra_baja.size > 0 else np.nan
potencia_muy_baja = np.mean(np.abs(coef[idx_muy_baja, :])**2, axis=0) if idx_muy_baja.size > 0 else np.nan
potencia_baja = np.mean(np.abs(coef[idx_baja, :])**2, axis=0) if idx_baja.size > 0 else np.nan
potencia_alta = np.mean(np.abs(coef[idx_alta, :])**2, axis=0) if idx_alta.size > 0 else np.nan

# Verificar si los datos de potencia no son NaN antes de graficar
total_time_seconds = len(r_r_intervals) * sampling_period
time_vector = np.linspace(0, total_time_seconds, len(r_r_intervals))

plt.figure(figsize=(10, 6))

if not np.isnan(potencia_ultra_baja).all():
    plt.plot(time_vector, potencia_ultra_baja, color='purple', label='Potencia Ultra Baja (ULF)')
else:
    print("No se puede graficar Potencia Ultra Baja: todos los valores son NaN.")

if not np.isnan(potencia_muy_baja).all():
    plt.plot(time_vector, potencia_muy_baja, color='green', label='Potencia Muy Baja (VLF)')
else:
    print("No se puede graficar Potencia Muy Baja: todos los valores son NaN.")

if not np.isnan(potencia_baja).all():
    plt.plot(time_vector, potencia_baja, color='blue', label='Potencia Baja (LF)')
else:
    print("No se puede graficar Potencia Baja: todos los valores son NaN.")

if not np.isnan(potencia_alta).all():
    plt.plot(time_vector, potencia_alta, color='red', label='Potencia Alta (HF)')
else:
    print("No se puede graficar Potencia Alta: todos los valores son NaN.")

plt.xlabel('Tiempo (s)')
plt.ylabel('Potencia')
plt.legend()
plt.title('Potencia espectral en bandas ULF, VLF, LF y HF')
plt.show()

plt.figure(figsize=(10, 6))

if not np.isnan(potencia_baja).all():
    plt.plot(time_vector, potencia_baja, color='blue', label='Potencia Baja (LF)')
else:
    print("No se puede graficar Potencia Baja: todos los valores son NaN.")

if not np.isnan(potencia_alta).all():
    plt.plot(time_vector, potencia_alta, color='red', label='Potencia Alta (HF)')
else:
    print("No se puede graficar Potencia Alta: todos los valores son NaN.")

plt.xlabel('Tiempo (s)')
plt.ylabel('Potencia')
plt.legend()
plt.title('Potencia espectral en bandas LF y HF')
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(np.abs(coef)**2, extent=[0, total_time_seconds, freqs.min(), freqs.max()], cmap='jet', aspect='auto', vmax=np.percentile(np.abs(coef)**2, 99))
plt.colorbar(label='Potencia')
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia (Hz)')
plt.title('Espectrograma de la Transformada Wavelet (Morlet)')
plt.ylim(0, 0.5)  # Limitar para visualizar solo hasta 0.5 Hz (rango relevante)
plt.show()




#----------------------------------GRAFICAS--------------------------------#

# Primer conjunto de gráficos: Señal original, señal filtrada con picos R, e intervalos R-R en función de muestras
plt.figure(figsize=(15, 8))

# Señal original
plt.subplot(3, 1, 1)
plt.plot(ecg_data, color='Green')
plt.title("Señal ECG Original")
plt.xlabel("Muestras [1]")
plt.ylabel("Amplitud (mV)")

# Señal filtrada
plt.subplot(3, 1, 2)
plt.plot(filtered_ecg, color='blue')
plt.plot(peaks, filtered_ecg[peaks], "rx")  # Marcar picos R
plt.title("Señal ECG Filtrada con Picos R Detectados")
plt.xlabel("Muestras[1]")
plt.ylabel("Amplitud (mV)")

# Intervalos R-R
plt.subplot(3, 1, 3)
plt.plot(r_r_intervals, color='orange')
plt.title("Intervalos R-R")
plt.xlabel("Número de Intervalo")
plt.ylabel("Intervalo R-R (s)")

# Ajustar espacio entre subplots en el primer conjunto
plt.tight_layout()  # Ajuste automático
plt.subplots_adjust(hspace=0.5)  # Ajuste manual, incrementar si es necesario
plt.show()

# Segundo conjunto de gráficos: Señal original, señal filtrada con picos R, e intervalos R-R en función del tiempo (primeros 30 segundos)
plt.figure(figsize=(15, 8))

# Señal original en función del tiempo
plt.subplot(3, 1, 1)
plt.plot(time_30s, ecg_30s, color='gray')
plt.title("Señal ECG Original (Primeros 10 segundos en función del tiempo)")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud(mV)")

# Señal filtrada con picos R en función del tiempo
plt.subplot(3, 1, 2)
plt.plot(time_30s, filtered_ecg_30s, color='blue')
plt.plot(time_30s[peaks_30s], filtered_ecg_30s[peaks_30s], "rx")  # Marcar picos R
plt.title("Señal ECG Filtrada con Picos R Detectados (Primeros 10 segundos en función del tiempo)")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud(mV)")

# Intervalos R-R en función del tiempo
r_r_intervals_30s = np.diff(peaks_30s) / fm  # Intervalos R-R en segundos para los primeros 30 segundos
plt.subplot(3, 1, 3)
plt.plot(r_r_intervals_30s * 1000, color='green')  # Convertir a milisegundos para coherencia
plt.title("Intervalos R-R (Primeros 10 segundos en función del tiempo)")
plt.xlabel("Número de Intervalo")
plt.ylabel("Intervalo R-R (ms)")

# Ajustar espacio entre subplots en el segundo conjunto
plt.tight_layout()  # Ajuste automático
plt.subplots_adjust(hspace=0.5)  # Ajuste manual, incrementar si es necesario
plt.show()

# Configuración de la figura para dos subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico del intervalo R-R (RRi)
ax1.hist(r_r_intervals, bins=30, color='blue')
ax1.set_xlabel("RR (s)")
ax1.set_ylabel("Frecuencia (Hz)")
ax1.set_title("Distribución del Intervalo R-R")

# Gráfico de la frecuencia cardíaca (FC)
ax2.hist(heart_rate, bins=30, color='red')
ax2.set_xlabel("FC (latidos/min)")
ax2.set_ylabel("Frecuencia(Hz)")
ax2.set_title("Distribución de la Frecuencia Cardíaca")

# Mostrar las gráficas
plt.tight_layout()
plt.show()

