import numpy as np
import serial
from scipy.signal import butter, filtfilt, medfilt
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QApplication
from PyQt6.QtCore import QTimer, QDateTime
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

        # Temporizador para actualizar la gráfica
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_grafica)
        self.timer.start(2)

        # Guardado de datos
        self.data_to_save = []
        self.saving_data = False
        self.start_time = None
        self.record_timer = QTimer()
        self.record_timer.setSingleShot(True)

    def puertos_disponibles(self):
        ports = serial.tools.list_ports.comports()
        self.puertos.clear()
        for port in ports:
            self.puertos.addItem(port.device)

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

    def actualizar_grafica(self):
        if self.ser and self.ser.is_open:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    ecg_value = int(line)
                    
                    # Validar si el valor está en un rango plausible para evitar picos
                    if 200 <= ecg_value <= 600:
                        self.y = np.roll(self.y, -1)
                        self.y[-1] = ecg_value

                        # Graficar señal original
                        self.curve_original.setData(self.y)

                        # Filtrado de la señal
                        dfH = filtfilt(self.b, self.a, self.y)
                        dfL = filtfilt(self.c, self.d, dfH)

                        # Aplicar filtro de mediana para reducir picos
                        dfL = medfilt(dfL, kernel_size=3)

                        # Graficar señal filtrada
                        self.curve_filtrada.setData(dfL)

                        # Guardar los datos en una lista
                        if self.saving_data:
                            self.data_to_save.append(dfL[-1])

                        # Verificar si han pasado 5 minutos para guardar
                        if self.saving_data and QDateTime.currentDateTime() >= self.start_time.addSecs(300):
                            self.finalizar_guardado_datos()
            except ValueError:
                print("Dato no válido recibido")

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

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = principal()
    window.show()
    sys.exit(app.exec())















    
