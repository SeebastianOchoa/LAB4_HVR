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



