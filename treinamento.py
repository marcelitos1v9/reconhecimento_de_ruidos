import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Ativar o uso de GPU, se disponível
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Treinamento utilizando GPU.")
else:
    print("GPU não detectada. Treinamento será feito com CPU.")

# Função para padronizar o comprimento dos espectrogramas
def pad_spectrogram(spectrogram, max_length):
    if spectrogram.shape[1] > max_length:
        return spectrogram[:, :max_length]
    else:
        padding = max_length - spectrogram.shape[1]
        return np.pad(spectrogram, ((0, 0), (0, padding)), mode='constant')

# Função para normalizar o espectrograma
def normalize_spectrogram(spectrogram):
    return (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)

# Função para carregar e processar o dataset
def load_data(data_dir, max_length=128):
    labels = []
    features = []
    label_map = {'ambulance': 0, 'construction': 1, 'dog': 2, 'firetruck': 3, 'traffic': 4}
    
    for category in os.listdir(data_dir):
        category_dir = os.path.join(data_dir, category)
        if os.path.isdir(category_dir):
            for audio_file in os.listdir(category_dir):
                file_path = os.path.join(category_dir, audio_file)
                y, sr = librosa.load(file_path, sr=None)
                spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
                
                # Normalizar o espectrograma
                spectrogram = normalize_spectrogram(spectrogram)
                
                # Padronizar o tamanho do espectrograma
                spectrogram = pad_spectrogram(spectrogram, max_length)
                
                features.append(spectrogram)
                labels.append(label_map[category])
    
    # Convertendo listas para arrays
    features = np.array(features)
    labels = np.array(labels)
    
    # Ajuste de dimensões para o modelo convolucional
    features = features[..., np.newaxis]
    
    return features, labels

# Função para criar o modelo CNN com dropout e regularização L2
def create_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer='l2', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer='l2'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer='l2'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),  # Adiciona dropout para evitar overfitting
        layers.Dense(128, activation='relu', kernel_regularizer='l2'),
        layers.Dense(5, activation='softmax')  # Ajustado para 5 classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Função para treinar o modelo usando validação cruzada
def train_model_with_cross_validation(features, labels):
    skf = StratifiedKFold(n_splits=3)
    fold_no = 1
    accuracies = []
    val_accuracies = []
    
    for train_idx, val_idx in skf.split(features, labels):
        print(f'Treinando o Fold {fold_no}...')
        train_X, val_X = features[train_idx], features[val_idx]
        train_y, val_y = labels[train_idx], labels[val_idx]

        model = create_cnn_model(input_shape=(features.shape[1], features.shape[2], 1))

        # Early stopping para evitar overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        history = model.fit(train_X, train_y, epochs=15, batch_size=32, 
                            validation_data=(val_X, val_y), callbacks=[early_stopping])

        # Salvar acurácias para análise geral ao final
        accuracies.append(history.history['accuracy'])
        val_accuracies.append(history.history['val_accuracy'])
        
        fold_no += 1

    # Gráfico geral de desempenho
    plt.figure(figsize=(10, 6))
    for i in range(len(accuracies)):
        plt.plot(accuracies[i], label=f'Fold {i+1} - Acurácia de Treinamento')
        plt.plot(val_accuracies[i], label=f'Fold {i+1} - Acurácia de Validação')
    
    plt.title('Acurácia de Treinamento e Validação para todos os Folds')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.show()

    # Criar diretório para salvar o modelo, se não existir
    if not os.path.exists('modelos'):
        os.makedirs('modelos')
    
    # Salvar o modelo treinado no diretório 'modelos/'
    model.save('modelos/modelo_sirene_v1.0.1.h5')
    print("Modelo salvo como 'modelos/modelo_sirene_v1.0.1.h5'.")

# Executando o treinamento
features, labels = load_data('dataset')
train_model_with_cross_validation(features, labels)
