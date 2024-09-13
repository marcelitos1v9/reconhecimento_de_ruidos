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
def audio_to_spectrogram(file_path, target_shape=(128, 128)):
    y, sr = librosa.load(file_path, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    
    # Normalizar o espectrograma
    spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)
    
    # Redimensionar para o formato esperado
    if spectrogram.shape[0] > target_shape[0]:
        spectrogram = spectrogram[:target_shape[0], :]
    else:
        padding = target_shape[0] - spectrogram.shape[0]
        spectrogram = np.pad(spectrogram, ((0, padding), (0, 0)), mode='constant')
    
    if spectrogram.shape[1] > target_shape[1]:
        spectrogram = spectrogram[:, :target_shape[1]]
    else:
        padding = target_shape[1] - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), mode='constant')
    
    # Adicionar dimensão para o canal
    spectrogram = spectrogram[np.newaxis, ..., np.newaxis]
    
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

# Função para criar e salvar a imagem da forma de onda do áudio
def create_waveform_image(file_path):
    y, sr = librosa.load(file_path, sr=None)
    plt.figure(figsize=(6, 2))
    plt.plot(np.linspace(0, len(y) / sr, num=len(y)), y)
    plt.title('Forma de Onda do Áudio')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    
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
        # Carregar o modelo e obter a forma de entrada esperada
        model = tf.keras.models.load_model('modelos/modelo_final.h5')
        input_shape = model.input_shape[1:3]
        print(f"Forma de entrada esperada: {input_shape}")
        
        # Processar o áudio e ajustar o formato
        spectrogram = audio_to_spectrogram(file_path, target_shape=input_shape)
        
        prediction = model.predict(spectrogram)
        print(f'Predição bruta: {prediction}')  # Adicionado para depuração
        classes = ['ambulance', 'dog', 'firetruck', 'traffic']
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
                
                # Criar e exibir a imagem do espectrograma
                spectrogram_img = create_spectrogram_image(file_path)
                if spectrogram_img:
                    spectrogram_label.config(image=spectrogram_img)
                    spectrogram_label.image = spectrogram_img  # Mantenha uma referência da imagem
                
                # Criar e exibir a imagem da forma de onda
                waveform_img = create_waveform_image(file_path)
                if waveform_img:
                    waveform_label.config(image=waveform_img)
                    waveform_label.image = waveform_img  # Mantenha uma referência da imagem
            else:
                result_label.config(text='Erro ao fazer a previsão.')
    except Exception as e:
        print(f"Erro ao abrir o arquivo: {e}")
        messagebox.showerror("Erro", f"Erro ao abrir o arquivo: {e}")

# Função principal para criar e exibir a interface Tkinter
def main():
    global result_label, spectrogram_label, waveform_label
    
    root = tk.Tk()
    root.title("Reconhecimento de Ruído")
    
    open_button = tk.Button(root, text="Selecionar Arquivo", command=open_file)
    open_button.pack(pady=20)
    
    result_label = tk.Label(root, text="Nenhuma previsão")
    result_label.pack(pady=10)
    
    spectrogram_label = tk.Label(root)
    spectrogram_label.pack(pady=20)
    
    waveform_label = tk.Label(root)
    waveform_label.pack(pady=20)
    
    root.mainloop()

if __name__ == '__main__':
    main()
