import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import pyaudio
import numpy as np
import time
import torch.nn.functional as F
from scipy.signal import spectrogram

# Definir la arquitectura del modelo
class AudioModel(nn.Module):
    def __init__(self):
        super(AudioModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Conv with padding
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # MaxPool to reduce dimensions
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # MaxPool to reduce dimensions
        )
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)

    def forward(self, x):
        print("Input shape to CNN: ", x.shape)  # Check the shape before CNN
        x = self.cnn(x)
        print("Shape after CNN: ", x.shape)  # Check the shape after CNN
        x = x.view(x.size(0), -1, 64)  # Reshape to (batch, time, feature)
        x, _ = self.rnn(x)
        x, _ = self.lstm(x)
        return x

# Definir la función para grabar audio en streaming
def grabar_audio_streaming():
    chunk = 1024
    formato = pyaudio.paInt16
    canales = 1
    frecuencia_muestreo = 44100
    tiempo_grabacion = 1

    audio = pyaudio.PyAudio()
    stream = audio.open(format=formato, channels=canales, rate=frecuencia_muestreo, input=True, frames_per_buffer=chunk)

    frames = []
    for i in range(0, int(frecuencia_muestreo / chunk * tiempo_grabacion)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    return frames

# Definir la función para cargar el audio grabado y convertirlo a espectrograma
def cargar_audio_grabado(frames):
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    audio_data = audio_data[:44100]  # Cortar el audio a 1 segundo
    
    # Convertir a espectrograma
    f, t, Sxx = spectrogram(audio_data, fs=44100, nperseg=1024)
    
    # Convertir el espectrograma a tensor
    Sxx = np.log(Sxx + 1e-10)  # Aplicar log para compresión
    Sxx_tensor = torch.tensor(Sxx).float()  # Convertir a tensor flotante
    Sxx_tensor = Sxx_tensor.unsqueeze(0).unsqueeze(0)  # Añadir dimensiones de batch y canal

    return Sxx_tensor

# Definir la función para procesar el audio con el modelo
def procesar_audio_con_modelo(audio, modelo):
    salida = modelo(audio)
    return salida

# Instanciar el modelo una vez
modelo = AudioModel()

while True:
    # Grabar audio en streaming
    frames = grabar_audio_streaming()

    # Cargar el audio grabado y convertirlo a espectrograma
    audio_grabado = cargar_audio_grabado(frames)

    # Procesar el audio con el modelo
    salida = procesar_audio_con_modelo(audio_grabado, modelo)

    # Imprimir la salida del modelo
    print(salida)

    # Esperar un segundo antes de grabar de nuevo
    time.sleep(1)
