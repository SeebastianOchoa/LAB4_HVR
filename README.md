---
title: "README"
author: "Sebastian Ochoa"
date: "2024-09-31"
output: html_document
---

# Laboratorio 4 : Variabilidad de la Frecuencia Cardiaca usando la Transformada Wavelet

## Introducción: 

La variabilidad de la frecuencia cardíaca (HRV) es un indicador no invasivo de la función del sistema nervioso autónomo y refleja la modulación de la actividad cardíaca por factores neurohumorales. Su análisis proporciona información valiosa sobre el estado de salud cardiovascular y puede utilizarse para evaluar el riesgo de enfermedades cardiovasculares, así como para monitorear la respuesta a tratamientos.

En este laboratorio, se propone capturar una señal electrocardiográfica (ECG) utilizando un sensor AD8232 y un sistema de adquisición de datos basado en Arduino UNO R3. La señal capturada será procesada digitalmente en Python, con el objetivo de extraer características relevantes de la HRV.

## Objetivo:

El objetivo principal de este laboratorio es analizar la variabilidad de la frecuencia cardíaca a partir de una señal ECG adquirida experimentalmente. Para ello, se llevarán a cabo las siguientes etapas:

1.  Realizar una investigación para una base teorica solida respecto a posibles conceptos utilizados a lo largo del desarrollo.
2.  Adquisición y preprocesamiento de la señal ECG: Captura de la señal utilizando un sensor AD8232, filtrado digital para eliminar ruido y extracción de los picos.
3.  Análisis de la HRV en el dominio del tiempo: Cálculo de parámetros estadísticos de los intervalos R-R, como la media y la desviación estándar, para evaluar la variabilidad a corto plazo y su respectivo analisis.
4.  Análisis de la HRV en el dominio de la frecuencia: Obtención del espectrograma de la HRV utilizando la transformada wavelet continua, con el fin de identificar las componentes de frecuencia asociadas a diferentes mecanismos fisiológicos.
5.  Interpretación de los resultados: Análisis de las variaciones en la potencia espectral en las bandas de baja y alta frecuencia para obtener conclusiones sobre el estado del sistema nervioso autónomo y la respuesta del organismo a diferentes estímulos.

# Fundamento Teorico 

**Actividad simpática y parasimpática del sistema nervioso autónomo y efecto de su activiad en la frecuencia cardiaca:**

> El sistema nervioso autónomo (SNA), influye sobre la actividad cardíaca, quedando esto visible mediante el registro electrocardiográfico (ECG) (Malik, 1995; Estévez Báes y col., 2007).

De acuerdo a lo anterior y según (José M.G, 2011; Universidad Nacional de Buenos Aires, "Procesamiento de señales electrocardiográficas mediante transformada wavelet, para el estudio de variabilidad de la frecuencia cardíaca") Se puede decir que el balance entre la rama simpática y parasimpática produce una variación pequeña entre latidos en la señal del ECG, de un corazón normal, el parasimpático incrementa esta variación y el simpático la decrementa. Recordando que; el sistema simpático prepara al organismo para situaciones estresantes o de emergencia mientra que, el sistema parasimpático controla los procesos corporales durante situaciones ordinarias.

**Variabilidad de la frecuencia cardiaca (HRV) medida como fluctuaciones en el intervalo R-R, y las frecuencias de interés en este análisis:**

> El estudio de las variaciones en la duración del intervalo entre latidos sucesivos (intervalo RR) aporta información sobre la modulación que ejerce el sistema simpato-vagal sobre el corazón. Estas variaciones latido a latido se conocen con el nombre de variabilidad de la frecuencia cardíaca (VFC) (Malik, 1995)

La medición de la HVR se realiza a partir de un electrocardiograma (ECG), en el cual se analizan los intervalos RR, es decir, la variación que existe de un latido a otro. En la valoración de la VFC, se utilizan métodos que filtran los latidos prematuros ectópicos y artefactos, detectando y corrigiendo los intervalos RR anormales. Los parámetros estadísticos para la caracterización de la VFC se valoran a partir del dominio temporal o del frecuencial de acuerdo a:

> Alvarado F. V, Camacho V.S, Monge R.S; 2017 " Revista médica de la universidad de Costa Rica", ISSN: 1659-2441.

-   ***Dominio temporal:**Los parámetros de este dominio se expresan en unidades de tiempo (ms), ya que analizan los lapsos entre compos QRS. Las más utilizadas son: el promedio del intervalo RR (RRi), la desviación estándar de todos los RRi normales (SDNN), la desviación estándar de los promedios de los RRi (SDANN), el promedio de las desviaciones estándar de los RRi en 5 minutos (SNDDi), la raíz cuadrada de la diferencia entre RRi normales adyacentes (rMSSD) y el porcentaje de RRi adyacentes con una diferencia de duración mayor a 50 ms (pNN50). Estos parámetros sólo ofrecen aspectos muy generales de la variabilidad y no permiten estudiar los ritmos intrínsecos presentes en los latidos cardíacos*

-   ***Dominio frecuencial:** Estos parámetros permiten cuantificar las fluctuaciones cíclicas del RRi utilizando un análisis de la densidad espectral de potencia, es decir, una descomposición y cuantificación de la VFC en 4 componentes oscilatorios (tabla 1). Esta técnica de descomposición del registro en bandas de frecuencia es similar a la que se realiza con las señales eléctricas obtenidas con un electroencefalograma. La composición espectral de la VFC no es estática, varía constantemente debido a la interacción de distintos elementos fisiológicos*

![Tabla 1.](https://github.com/SeebastianOchoa/Imgs_Lab4/blob/eeb47d13fcdeb35de2707f691434c7a2decbe23a/Tabla_1.png).

Según el autor algunas de las **frecuencias de interes** son el **componente de alta frecuencia** del HVR mide el HVR debida a la arritmia sinusal respiratoria. El principal regulador de la arritmia sinusal respiratoria es el sistema parasimpatico (SP), por lo que el componente HF se ha utilizado experimental y clínicamente como marcador de la actividad del SP [11,12]. Esta afirmación se basa en estudios con modelos animales, en los cuales se ha medido directamente la actividad vagal y su relación con la arritmia sinusal respiratoria y en estudios farmacológicos en humanos en los que al realizar bloqueo colinérgico, se disminuye dicha variabilidad . No obstante, parece ser que la relación del HF con la arritmia sinusal respiratoria se da únicamente en condiciones fisiológicas específicas, las cuales son analizadas extensamente por Grossman et al.

Por otro lado, La **componente de baja frecuencia** del HVR El componente LF ocurre en sincronización con las fluctuaciones fisiológicas de la presión arterial. Inicialmente se postulaba que el componente LF era un marcador directo de la actividad del sistema simpatico (SS), debido a que aumentaba en registros de 24 horas bajo condiciones fisiológicas asociadas al incremento del SS como el estrés y el ejercicio físico. No obstante, luego se reconoció que su origen era más complejo, ya que representa un efecto combinado del SS y el SP.

**Transformada Wavelet: definición, usos y tipos de wavelet utilizadas en señales biológicas.**

> La transformada Wavelet es eficiente para el análisis local de señales no estacionarias y de rápida transitoriedad y, al igual que la Transformada de Fourier con Ventana, mapea la señal en una representación de tiempo-escala. El aspecto temporal de las señales es preservado. La diferencia está en que la Transformada Wavelet provee análisis de multiresolución con ventanas dilatadas. El análisis de las frecuencias de mayor rango se realiza usando ventanas angostas y el análisis de las frecuencias de menor rango se hace utilizando ventanas anchas

De manera muy general, la Transformada Wavelet de una función f(t) es la descomposición de f(t) en un conjunto de funciones ψs,τ (t), que forman una base y son llamadas las “Wavelets”. **(Libro "Transformada Wavelet "Descomposición de señales")**

Las Wavelets, funciones bases de la Transformada Wavelet, son generadas a partir de una función Wavelet básica, mediante traslaciones y dilataciones. Estas funciones permiten reconstruir la señal original a través de la Transformada Wavelet inversa. La Transformada Wavelet no es solamente local en tiempo, sino también en frecuencia. Dentro de los **usos** de esta poderosa herramienta podemos nombrar, además del análisis local de señales no estacionarias, el análisis de señales electrocardiográficas, sísmicas, de sonido, de radar, así como también es utilizada para la compresión y procesamiento de imágenes y reconocimiento de patrones.

*Tipos de Wavelets:*

1.  **Wavelet de Haar:** Es la wavelet más simple y se caracteriza por su forma rectangular. Es útil para detectar discontinuidades y cambios bruscos en la señal, pero puede no ser la mejor opción para señales biológicas que suelen tener componentes de frecuencia más altas. 2. **Wavelet de Daubechies:** Son un conjunto de wavelets ortogonales con diferentes órdenes. Se utilizan comúnmente en el procesamiento de imágenes y señales debido a su buena localización en tiempo y frecuencia.
2.  **Wavelet de Coiflets:** Similares a las Daubechies, pero con momentos nulos de mayor orden, lo que las hace más suaves y adecuadas para señales con menos ruido.
3.  **Wavelet de Symlets:** Otra familia de wavelets ortogonales, con simetría y suavidad variables.
4.  **Wavelet de Morlet:** Es una wavelet compleja que combina una oscilación sinusoidal con una gaussiana. Esta combinación le otorga una excelente localización tanto en tiempo como en frecuencia, lo que la hace muy adecuada para el análisis de señales no estacionarias como el ECG.

## Diagrama de flujo

![Diagrama de flujo:](https://github.com/SeebastianOchoa/Imgs_Lab4/blob/eeb47d13fcdeb35de2707f691434c7a2decbe23a/Diagrama%20de%20flujo.png)
Link diagrama de flujo:  https://www.canva.com/design/DAGT8Ga4GBU/MPZWtITwqG352pjyt9zQew/edit?utm_content=DAGT8Ga4GBU&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

## Adquisición de la señal 

Para la adquisición de la señal ECG, se utilizó el sensor AD8232 conectado a una tarjeta Arduino Uno R3, la cual realizó la conversión analógica-digital a través del pin A0. Los pines de 3.3V y tierra del Arduino se conectaron a sus respectivos pines del sensor para asegurar un circuito común.

![Configuracion Analogo-digital:](https://github.com/SeebastianOchoa/Imgs_Lab4/blob/eeb47d13fcdeb35de2707f691434c7a2decbe23a/Arduino_Configuracion.jpg).

Una vez configurados analogamente ambos componentes, se procedio a conectar el Arduino UNO R3 al computador para realizar su configuracion con el programa "Arduino IDE" con el codigo a continuacion:

```         
void setup() {
    Serial.begin(115200); // Configura la velocidad de baudios
}

void loop() {
    // Lee el valor del sensor directamente sin promediar
    int valorECG = analogRead(A0);
    
    // Envía el valor directamente al puerto serie para graficarlo
    Serial.println(valorECG);
    
    // Espera para ajustar la frecuencia de muestreo a 500 Hz (2 ms)
    delay(2); // Espera 2 ms entre lecturas
}
```

Inicialmente, se realizo la configuracion de electrodos en el paciente como se muestra en la siguiente figura y se verificó la señal obtenida mediante el plotter serial de Arduino, comprobando que la captura del ECG fuera correcta. Tanto en el Arduino IDE como en la interfaz gráfica de Python se configuró la **comunicación serial** a 115200 baudios,**la frecuencia de muestreo** se estableció en 500 Hz en ambos programas. Esta frecuencia fue seleccionada siguiendo el criterio del teorema de Nyquist, que indica que la frecuencia de muestreo debe ser al menos el doble de la frecuencia máxima de la señal para evitar aliasing. Dado que la frecuencia máxima de los filtros aplicados es de 40 Hz, la frecuencia de muestreo teóricamente necesaria sería entre 100 y 140 Hz; sin embargo, en la bibliografía se recomienda una frecuencia de 500-1000 Hz para señales de electrocardiogramas, ya que estas frecuencias más altas permiten capturar con mayor precisión detalles finos de la señal ECG, como las complejas morfologías de los picos QRS.

![Configuracion Electrodos:](https://github.com/SeebastianOchoa/Imgs_Lab4/blob/eeb47d13fcdeb35de2707f691434c7a2decbe23a/Electrodos_ubicacion.png).
![Señal Plotter Arduino:](https://github.com/SeebastianOchoa/Imgs_Lab4/blob/886da207ebc3131f97f87762bbb551dde618f937/Se%C3%B1al_arduino.gif)

A continuacion se encuentra el codigo y parametros utilzados en phyton para la adquisicion de la señal:

```         
from PyQt6 import uic
import pyqtgraph as pg
import serial.tools.list_ports
import os

class principal(QMainWindow):
    def __init__(self):
        super(principal, self).__init__()
        uic.loadUi("ecg_interfaz.ui", self)
        self.puertos_disponibles()
        self.ser = None
        self.connect.clicked.connect(self.conectar)

        # Parámetros de graficación
        self.fm = 500
        self.tiempo_muestreo = 5
        self.num_points = self.fm * self.tiempo_muestreo
        self.x = np.linspace(0, self.tiempo_muestreo, self.num_points)
        self.y = np.zeros(self.num_points)
        
      def conectar(self):
        if self.connect.text() == "CONECTAR":
            com = self.puertos.currentText()
            try:
                self.ser = serial.Serial(com, 115200)
                print("Conexión exitosa")
                self.connect.setText("DESCONECTAR")
                self.iniciar_guardado_datos_automatico()
            except serial.SerialException as e:
                print(f"Error: {e}")
        else:
            if self.ser:
                self.ser.close()
            self.connect.setText("CONECTAR")
        
          # Guardado de datos
        self.data_to_save = []
        self.saving_data = False
        self.start_time = None
        self.record_timer = QTimer()
        self.record_timer.setSingleShot(True)
        
    def iniciar_guardado_datos_automatico(self):
        """Inicia el temporizador de 30 segundos antes de empezar a guardar datos"""
        self.saving_data = False
        self.data_to_save = []
        self.start_time = QDateTime.currentDateTime()
        print("Esperando 30 segundos antes de iniciar el guardado automático...")
        self.record_timer.timeout.connect(self.comenzar_guardado_datos)
        self.record_timer.start(30000)

    def comenzar_guardado_datos(self):
        """Inicia el guardado de datos de la señal filtrada después de 30 segundos"""
        self.saving_data = True
        print("Iniciando guardado automático de datos por 5 minutos.")

    def finalizar_guardado_datos(self):
        """Guarda los datos de la señal filtrada en un archivo de texto después de 5 minutos"""
        self.saving_data = False
        file_name = "datos_ecg_filtrados.txt"
        file_path = os.path.join(os.getcwd(), file_name)
        with open(file_path, 'w') as file:
            for data in self.data_to_save:
                file.write(f"{data}\n")
        print(f"Datos guardados exitosamente en: {file_path}")
```

Luego de la validación, los datos de la señal ECG fueron enviados mediante comunicación serial hacia una interfaz gráfica desarrollada en Python. En esta interfaz, la señal fue recibida, filtrada como veremos en segmentos posteriores y graficada, mostrando tanto la señal original como la señal procesada. Finalmente, se realizo un **periodo de muestreo** de 5 minutos, durante los cuales cada muestra fue cuantificada en niveles de 10 bits (0-1023), proporcionando una resolución suficiente para captar variaciones de voltaje precisas en la señal de ECG. Los datos capturados fueron almacenados en un archivo .txt para su posterior procesamiento. En cuanto a las unidades y escalas, en el eje de tiempo se utilizó segundos (ms) para una interpretación temporal intuitiva, mientras que en el eje de amplitud se emplearon milivoltios (mV), una unidad estándar en ECG que permite cuantificar los potenciales eléctricos generados por la actividad cardíaca. Los valores estadísticos de la señal, tales como el promedio y la desviación estándar, fueron calculados posteriormente para analizar la estabilidad y consistencia de la señal capturada y se podran observar en segmentos posteriores.

![Señal Phyton:](https://github.com/SeebastianOchoa/Imgs_Lab4/blob/886da207ebc3131f97f87762bbb551dde618f937/se%C3%B1al_phyton.gif)

## Pre-procesamiento de la señal 

En esta etapa se implementaron dos tipos de filtros, un pasa-bajo y un pasa-alto, para mejorar la calidad de la señal ECG capturada. Los cálculos específicos de diseño se detallarán a continuación. Los filtros se aplicaron inicialmente en la interfaz gráfica de Python y luego se replicaron en el entorno Spyder (Anaconda) para el procesamiento posterior de los datos. Ambos filtros fueron diseñados siguiendo la bibliografía consultada, donde se recomienda el uso de filtros Butterworth de segundo orden para señales biomédicas, como el ECG, debido a su respuesta de amplitud plana en la banda de paso, que evita distorsionar la señal.

![Calculos de filtros:](https://github.com/SeebastianOchoa/Imgs_Lab4/blob/886da207ebc3131f97f87762bbb551dde618f937/Calculos_1.png)
![Calculos de filtros:](https://github.com/SeebastianOchoa/Imgs_Lab4/blob/886da207ebc3131f97f87762bbb551dde618f937/Calculo_2.png)


A continuacion se observa el codigo de los filtros en la interfaz grafica de Phyton:

```         
 # Configuración de las gráficas originales y filtradas
        self.plot_widget_original = pg.PlotWidget(title="Señal Original")
        layout = QVBoxLayout()
        layout.addWidget(self.plot_widget_original)
        self.graficaWidget.setLayout(layout)
        self.curve_original = self.plot_widget_original.plot(pen='y')
        self.plot_widget_original.setYRange(200, 600)

        self.plot_widget_filtrada = pg.PlotWidget(title="Señal Filtrada")
        layout1 = QVBoxLayout()
        layout1.addWidget(self.plot_widget_filtrada)
        self.FILTRO.setLayout(layout1)
        self.curve_filtrada = self.plot_widget_filtrada.plot(pen='g')
        self.plot_widget_filtrada.setYRange(-100, 300)

        # Filtros
        self.fc_high = 0.5
        self.fc_low = 40
        self.orden = 2  # Reducimos el orden del filtro para evitar sobrerrespuesta
        self.b, self.a = butter(self.orden, self.fc_high / (0.5 * self.fm), btype='high')
        self.c, self.d = butter(self.orden, self.fc_low / (0.5 * self.fm), btype='low')
```

Para el filtro pasa-bajo, se eligió una frecuencia de corte de 40 Hz, dentro del rango recomendado por distintos autores de 20-40 Hz, que permite eliminar el ruido de alta frecuencia sin perder la información esencial de la señal ECG. En el caso del filtro pasa-alto, se seleccionó una frecuencia de corte de 1 Hz (en el rango de 0.5-1 Hz), que permite suprimir las componentes de baja frecuencia, como la línea base, manteniendo los detalles relevantes de la señal. En este caso se descarto el filtro "Notch" debido a que el sensor AD8232 ya filtra esta frecuencia.

Los datos provenientes del Arduino fueron filtrados primero en la interfaz gráfica para su visualización, y luego se guardaron en un archivo .txt para ser nuevamente filtrados en Spyder, asegurando así una mayor precisión y limpieza en los datos procesados. El uso de un filtro de segundo orden fue adecuado como lo menciona la bibliografia, ya que ofrece un equilibrio óptimo entre precisión y simplicidad, proporcionando una atenuación adecuada de las frecuencias no deseadas sin agregar complejidad innecesaria al sistema o eliminar datos en exceso relevantes para nuestro laboratorio.

Una vez en Spyder (Anaconda) los datos se extrajeron en un arreglo y se replicaron los filtros ya aplicado en la interfaz grafica de phyton siguiendo los mismos parametros de la literatura y los calculos realizados. A continuacion podremos observas el codigo y la correspondiente grafica.

```         
 import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import pywt

#----------------------------------FUNCIONES--------------------------------#
fm = 500 # Frecuencia de muestreo en Hz

# Definir funciones para los filtros
def butter_highpass(cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass(cutoff, fs, order=2):
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

#----------------------------------GRAFICAR--------------------------------#

# Primer conjunto de gráficos: Señal original, señal filtrada con picos R, e intervalos R-R en función de muestras
plt.figure(figsize=(15, 8))

# Señal original
plt.subplot(3, 1, 1)
plt.plot(ecg_data, color='Green')
plt.title("Señal ECG Original")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")

# Señal filtrada
plt.subplot(3, 1, 2)
plt.plot(filtered_ecg, color='blue')
plt.plot(peaks, filtered_ecg[peaks], "rx")  # Marcar picos R
plt.title("Señal ECG Filtrada con Picos R Detectados")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")

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
plt.ylabel("Amplitud")

# Señal filtrada con picos R en función del tiempo
plt.subplot(3, 1, 2)
plt.plot(time_30s, filtered_ecg_30s, color='blue')
plt.plot(time_30s[peaks_30s], filtered_ecg_30s[peaks_30s], "rx")  # Marcar picos R
plt.title("Señal ECG Filtrada con Picos R Detectados (Primeros 10 segundos en función del tiempo)")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud")

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
```

![Señal Orginal y Filtros:](https://github.com/SeebastianOchoa/Imgs_Lab4/blob/9d0cb199f36c060feb1034c24195002f120b2e45/5minutos.png)

En la grafica podemos observar la señal original y la señal filtrada junto con la grafica de los calculos realizados para los intervalos R-R. En esta primera grafica se realiza en funcion del numero de muestras total (5 minutos), para darnos una visualizacion completa de los datos tomados en nuestra interfaz de phyton. Ademas podemos visualizar que no hay mucha diferencia entre la señal orginen y la señal con los filtros. Esto puede ser debido a que nuestros datos del archivo .txt ya vienen con un proceso de filtracion desde la interfaz.

Una vez extraidos los datos filtrados y graficados. Redimensionamos nuestro arreglo para graficar tan solo **10 segundos** de la señal y se realiza la debida conversion para que quede una grafica de amplitud (mV) vs Tiempo (ms). Ademas se realiza gracias a la biblioteca NCyPy la detección y graficación de los picos R-R como podemos observar en la siguiente grafica y codigo de programación.

```         
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
```

![Señal 10 Segundos](https://github.com/SeebastianOchoa/Imgs_Lab4/blob/9d0cb199f36c060feb1034c24195002f120b2e45/10segundos.png)

## Análisis de la HRV en el dominio del tiempo 

Continuando con nuestro procesamiento y respectivo analisis, se procedio a calcular los paramtros HVR mas comunes en funcion del tiempo.

> Segun el autor Alvarado F. V, Camacho V.S, Monge R.S; 2017 " Revista médica de la universidad de Costa Rica", ISSN: 1659-2441. y como en las secciones anteriores. Las caracteristicas mas medidas en el dominio del tiempo en un HVR son:

1.  RRi
2.  SDNN
3.  SDANN
4.  SNDDi
5.  rMSSD
6.  pNN50

De acuerdo a lo siguiente se realizo el siguiente codigo en el programa spyder (Anaconda) para calcular los parametros que nos menciona el autor.

```         

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
```

Adicionalmente, se calculo el inverso del promedio de intervalos RRi lo cual no otorga el valor de la frecuencia cardiaca y se obtuvieron los siguientes resultados, ademas de la grafica comparativa entre la distribución del intervalo R-R y la distribucion de la frecuencia cardiaca.

![Señal Phyton:](https://github.com/SeebastianOchoa/Imgs_Lab4/blob/9d0cb199f36c060feb1034c24195002f120b2e45/RESULTADOS.png)


De acuerdo con estos resultados los cuales nos permiten evaluar la modulación autónoma del corazón, específicamente la actividad de los sistemas simpático y parasimpático por lo cual vamos a realizar una pequeña definición de cada parametro y un analisis critico que nos brinde informacion acerca de nuestro paciente. A continuacion analisamos punto por punto:

1.  El promedio del intervalo R-R **(RRi)** refleja el tiempo medio entre latidos, y es inversamente proporcional a la frecuencia cardíaca (FC). Con un RRi de 634.04 ms, la frecuencia cardíaca promedio sería aproximadamente 94.6 latidos por minuto (bpm), indicando una frecuencia cardíaca elevada, posiblemente relacionada con factores como actividad física reciente, estrés o ansiedad, o menor influencia del sistema parasimpático (el cual reduce la frecuencia cardíaca). Este valor puede ser considerado en el rango alto, especialmente si la persona se encontraba en reposo durante la medición, lo cual sugiere una predominancia simpática o menor tono vagal.

2.  **SDNN** es una medida global de la variabilidad de la frecuencia cardíaca y refleja la influencia conjunta de los sistemas simpático y parasimpático. Un valor de 105.26 ms es moderado y sugiere un nivel aceptable de variabilidad, aunque es menor en comparación con individuos con buena salud cardíaca y mayor influencia parasimpática, donde valores de SDNN suelen ser más altos. Un SDNN bajo puede estar relacionado con un mayor riesgo cardiovascular y es común en personas con estrés crónico o con actividad simpática elevada.

3.  **SDSD** es un parámetro relacionado con las variaciones a corto plazo en los intervalos R-R y se utiliza para evaluar la actividad del sistema parasimpático. Un valor de 146.80 ms es elevado, lo cual indica una buena modulación vagal (actividad del sistema parasimpático), aunque puede parecer contradictorio con la alta frecuencia cardíaca promedio. Este valor sugiere que la persona tiene una capacidad considerable para modular la frecuencia cardíaca en intervalos cortos, a pesar de un promedio de RR bajo, lo cual podría interpretarse como un sistema autónomo que responde bien a estímulos repentinos.

4.  **RMSSD** también refleja la actividad del sistema parasimpático y es una medida robusta de variaciones de intervalo a corto plazo. Un RMSSD de 146.81 ms es alto y sugiere una buena capacidad del sistema parasimpático para influir en la frecuencia cardíaca, lo cual es positivo en términos de salud autonómica. Valores altos de RMSSD están asociados con una mayor flexibilidad autonómica y menor estrés. Este valor es consistente con el SDSD alto y sugiere una respuesta parasimpática sólida.

5.  **El pNN50** es el porcentaje de pares de intervalos R-R consecutivos con una diferencia mayor a 50 ms y también se asocia con la actividad del sistema parasimpático. Un valor de 37.79% es moderado-alto y sugiere una buena capacidad para cambios rápidos en el intervalo R-R. Esto se interpreta como una respuesta vagal saludable y es un buen indicador de un sistema parasimpático activo y capaz de modular la frecuencia cardíaca eficientemente. En individuos sanos, valores elevados de pNN50 están asociados con menor estrés y buena salud cardiovascular.

Ahora bien, teniendo en cuanta este analisis en conjunto, los parámetros RMSSD, SDSD y pNN50 indican una respuesta parasimpática adecuada y saludable, mientras que el SDNN moderado y el promedio RRi relativamente bajo (frecuencia cardíaca alta) sugieren una posible predominancia simpática o un menor tono vagal en reposo. Esta combinación de una frecuencia cardíaca promedio elevada y una variabilidad a corto plazo alta podría indicar un sistema autónomo equilibrado pero con una tendencia hacia la respuesta simpática, posiblemente como reacción a factores externos (como estrés o actividad física reciente).

![Señal Phyton:](https://github.com/SeebastianOchoa/Imgs_Lab4/blob/fc357539f1697ef6187d24e52f2ab83b91bd625e/frecuenciacardiaca.png)

Por otro lado, Las gráficas muestran la distribución de los intervalos R-R y la frecuencia cardíaca en una muestra de ECG. La mayoría de los intervalos R-R se concentran entre 0.6 y 0.7 segundos, indicando una frecuencia cardíaca regular en reposo, mientras que algunos valores dispersos sugieren variabilidad en el ritmo cardíaco. En la segunda gráfica, la frecuencia cardíaca se concentra entre 80 y 100 latidos por minuto (lpm), lo cual es típico en adultos en reposo, aunque también se observan valores extremos que podrían estar asociados con variaciones en los intervalos R-R. En conjunto, estos datos sugieren un ritmo cardíaco mayormente regular.

## Aplicación de transformada Wavelet

En la última fase del laboratorio, se aplica la Transformada Wavelet Continua (CWT) utilizando una wavelet de tipo Morlet a la señal ECG, conforme a los parámetros previamente definidos. La CWT permite descomponer la señal original en componentes de frecuencia y tiempo, mediante una suma ponderada de la señal f(t) con una función wavelet Ψ reescalada y desplazada a través del tiempo. Las wavelets, como la de Morlet, son formas de onda de duración limitada con un promedio de cero, lo que las hace adecuadas para analizar variaciones rápidas en señales biológicas. Específicamente, la wavelet de Morlet, que combina una onda sinusoidal con una ventana gaussiana, es biológicamente apropiada para analizar señales como el ECG, ya que permite capturar tanto las frecuencias bajas como las altas, ajustando la longitud de la ventana de tiempo en función de la frecuencia analizada.

Una vez entendido este concepto de la transformada Wavelet especificamente de tipo Morlet. se procedio programar la aplicación de la transformada seleccionada a nuestros datos.

```         

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
```

Con la transformada de Wavelet tipo Morlet aplicada a nuestros datos se procede a ser graficados para un posterior analisis.

```         

#----------------------------------GRAFICAS ESPECTOGRAMA--------------------------------#

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
```

![Espectograma Wavelet-Morlet:](https://github.com/SeebastianOchoa/Imgs_Lab4/blob/be4901c3979704ae3769bf8d300755e018422ab2/espectrograma.png).

Esta gráfica muestra el espectrograma resultante de aplicar la transformada wavelet de Morlet a la señal ECG, donde se observa la distribución de las frecuencias a lo largo del tiempo y su potencia espectral en distintas bandas. En el eje vertical se encuentran las frecuencias (en Hz), mientras que en el eje horizontal se representa el tiempo (en segundos). La potencia está codificada por color, donde los tonos rojos y amarillos representan niveles más altos de potencia, mientras que los tonos azules indican una menor potencia espectral.

En el espectrograma, se observa que las frecuencias más bajas (0 a 0.1 Hz) muestran una mayor potencia a lo largo de la señal, lo que es característico de las modulaciones en las bandas ultra baja frecuencia (ULF) y muy baja frecuencia (VLF). A medida que la frecuencia aumenta hacia el rango de baja frecuencia (LF) y alta frecuencia (HF), la potencia disminuye, lo cual es consistente con el comportamiento esperado en un análisis ECG, donde las componentes de muy baja frecuencia tienden a tener mayor potencia debido a la regulación autónoma. Se observa, además, que hay fluctuaciones en la potencia de LF y HF en ciertos intervalos de tiempo, lo cual podría estar vinculado a cambios en la actividad respiratoria y al sistema nervioso autónomo.

![Potencia espectral 2 Bandas:](https://github.com/SeebastianOchoa/Imgs_Lab4/blob/be4901c3979704ae3769bf8d300755e018422ab2/2bandas.png).

Esta gráfica muestra la potencia espectral específica en las bandas de baja frecuencia (LF, en azul) y alta frecuencia (HF, en rojo) a lo largo del tiempo. La banda LF se asocia principalmente con la regulación autónoma, mientras que la HF está ligada a la actividad respiratoria.

En el análisis de esta gráfica, se observa que la potencia en la banda LF se mantiene relativamente estable, con ligeros aumentos en ciertos intervalos de tiempo. Esto es indicativo de una modulación autónoma constante, sin variaciones abruptas. Por otro lado, la banda HF muestra picos más pronunciados, especialmente al inicio y al final del periodo analizado, sugiriendo una actividad respiratoria más fluctuante en estos intervalos. La variación en HF refleja que hay episodios de incremento de la actividad respiratoria, que coinciden con aumentos en la potencia espectral de HF.

![Potencia espectral 4 Bandas:](https://github.com/SeebastianOchoa/Imgs_Lab4/blob/be4901c3979704ae3769bf8d300755e018422ab2/4bandas.png).

En esta última gráfica, se presenta la potencia espectral en cuatro bandas de frecuencia: ultra baja frecuencia (ULF, en morado), muy baja frecuencia (VLF, en verde), baja frecuencia (LF, en azul), y alta frecuencia (HF, en rojo). Esta visualización permite observar cómo se distribuye la potencia en una escala completa de frecuencias para la señal ECG analizada.

El análisis muestra que la banda ULF presenta una potencia considerablemente mayor en comparación con las demás bandas, manteniendo una curva en forma de arco a lo largo del tiempo, lo cual es típico de esta banda, asociada a los procesos de regulación más lentos, como la termorregulación y la actividad neuroendocrina. La banda VLF también tiene una presencia notable en la señal, aunque con menor potencia que ULF, y presenta una tendencia relativamente constante, reflejando la actividad autónoma de baja frecuencia.

Las bandas LF y HF muestran potencias mucho menores en comparación con ULF y VLF. LF mantiene una ligera variabilidad a lo largo del tiempo, mientras que HF muestra variaciones más abruptas y picos en puntos específicos. Estos patrones en LF y HF confirman el comportamiento esperado en relación a la actividad autonómica y respiratoria, respectivamente, y muestran cómo los diferentes sistemas fisiológicos impactan en la señal ECG en distintas escalas de frecuencia.

## Bibliografia 

1.  Alvarado F.V , Camacho V.S , Monge R.S; 2017. "Reviste Médica de la universidad de Costa Rica", (Variabilidad de la frecuencia cardiaca como indicador de la actividad del sistema nervioso autonomo) Vol 11. Art 5. ISNN: 1659-2441

2.  Jose Manuel Gallardo; 2011. "Procesamiento de señales electrocardiograficas mediante transformada Wavelet para el estudio del HVR". Universidad Nacional de Buenos Aires. Vol 9 No. 1.

3.  Libro "Transformada Wavelet", 2006. "Descomposicion de señales"

4.  Jorge Muñoz; 1997. Universidad de Valencia, Tesis "Compresion del ECG en tiempo real". Facultad del departamento de electronica e informatica.
