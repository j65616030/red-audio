import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pyaudio
import wave
import numpy as np

# Definir la arquitectura del modelo
class AudioModel(nn.Module):
    def __init__(self):
        super(AudioModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 64, 128)
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

# Definir la función para cargar el audio grabado
def cargar_audio_grabado(frames):
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    audio_data = audio_data[:44100]  # Cortar el audio a 1 segundo
    audio_data = torch.tensor(audio_data).unsqueeze(0).unsqueeze(0)
    return audio_data

# Definir la función para procesar el audio con el modelo
def procesar_audio_con_modelo(audio):
    modelo = AudioModel()
    salida = modelo(audio)
    return salida

while True:
    # Grabar audio en streaming
    frames = grabar_audio_streaming()

    # Cargar el audio grabado
    audio_grabado = cargar_audio_grabado(frames)

    # Procesar el audio con el modelo
    salida = procesar_audio_con_modelo(audio_grabado)

    # Reproducir la salida del modelo
    salida_reproducida = salida.squeeze(0).squeeze(0).numpy()
    salida_reproducida = salida_reproducida.astype(np.int16)
    wave_file = wave.open("salida.wav", "wb")
    wave_file.setnchannels(1)
    wave_file.setsampwidth(2)
    wave_file.setframerate(44100)
    wave_file.writeframes(salida_reproducida.tobytes())
    wave_file.close()

    # Reproducir el archivo de audio generado
    os.system("aplay salida.wav")