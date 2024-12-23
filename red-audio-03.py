import pyaudio
import wave
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# Definir la arquitectura de la CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128*10*10, 128)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 128*10*10)
        x = torch.relu(self.fc1(x))
        return x

# Definir la arquitectura de la LSTM
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 128).to(x.device)
        c0 = torch.zeros(1, x.size(0), 128).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return out[:, -1, :]

# Definir la función para grabar audio
def grabar_audio():
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

# Definir la función para procesar el audio
def procesar_audio(frames):
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    audio_data = audio_data[:44100]  # Cortar el audio a 1 segundo
    audio_data = audio_data.reshape(1, 1, 44100)  # Reshape para la CNN
    audio_data = torch.tensor(audio_data).float()  # Convertir a tensor flotante

    cnn = CNN()
    lstm = LSTM()

    caracteristicas = cnn(audio_data)
    caracteristicas = caracteristicas.unsqueeze(1)  # Agregar dimensión para la LSTM
    salida = lstm(caracteristicas)

    return salida

# Grabar y procesar audio cada x tiempo
x_tiempo = 5  # segundos
while True:
    frames = grabar_audio()
    salida = procesar_audio(frames)
    print(salida)
    time.sleep(x_tiempo)
