import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import librosa
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import io

# Função para carregar e processar o arquivo de áudio
def audio_to_spectrogram(file_path, max_length=128):
    y, sr = librosa.load(file_path, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    
    # Padronizar o comprimento do espectrograma
    if spectrogram.shape[1] > max_length:
        spectrogram = spectrogram[:, :max_length]
    else:
        padding = max_length - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), mode='constant')
    
    # Adicionar dimensão para o canal
    spectrogram = spectrogram[np.newaxis, ..., np.newaxis]
    
    # Normalizar o espectrograma
    spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)
    
    return spectrogram

# Função para criar e salvar a imagem do espectrograma
def create_spectrogram_image(file_path):
    y, sr = librosa.load(file_path, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    
    # Padronizar o comprimento do espectrograma
    if spectrogram.shape[1] > 128:
        spectrogram = spectrogram[:, :128]
    else:
        padding = 128 - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), mode='constant')
    
    # Criar a imagem do espectrograma
    plt.figure(figsize=(6, 3))
    plt.imshow(librosa.power_to_db(spectrogram, ref=np.max), aspect='auto', cmap='inferno')
    plt.axis('off')
    
    # Salvar a imagem em um objeto BytesIO
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    buf.seek(0)
    image = Image.open(buf)
    return ImageTk.PhotoImage(image)

# Função para fazer previsões com o modelo carregado
def predict_audio(file_path):
    try:
        spectrogram = audio_to_spectrogram(file_path)
        model = tf.keras.models.load_model('modelo_sirene.h5')
        prediction = model.predict(spectrogram)
        print(f'Predição bruta: {prediction}')  # Adicionado para depuração
        classes = ['ambulance', 'firetruck', 'traffic']
        predicted_class = classes[np.argmax(prediction)]
        return predicted_class
    except Exception as e:
        print(f"Erro ao processar o áudio: {e}")
        return None

# Função chamada quando o botão de abrir arquivo é clicado
def open_file():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if file_path:
            print(f'Arquivo selecionado: {file_path}')
            result = predict_audio(file_path)
            if result:
                result_label.config(text=f'Classificação: {result}')
                img = create_spectrogram_image(file_path)
                if img:
                    spectrogram_label.config(image=img)
                    spectrogram_label.image = img  # Mantenha uma referência da imagem
            else:
                result_label.config(text='Erro ao fazer a previsão.')
    except Exception as e:
        print(f"Erro ao abrir o arquivo: {e}")
        messagebox.showerror("Erro", f"Erro ao abrir o arquivo: {e}")

# Função principal para criar e exibir a interface Tkinter
def main():
    global result_label, spectrogram_label
    
    root = tk.Tk()
    root.title("Reconhecimento de Ruído")
    
    open_button = tk.Button(root, text="Selecionar Arquivo", command=open_file)
    open_button.pack(pady=20)
    
    result_label = tk.Label(root, text="Nenhuma previsão")
    result_label.pack(pady=10)
    
    spectrogram_label = tk.Label(root)
    spectrogram_label.pack(pady=20)
    
    root.mainloop()

if __name__ == '__main__':
    main()
